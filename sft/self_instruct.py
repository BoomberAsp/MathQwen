"""
Self-Instruct Pipeline for Math Reasoning SFT.
Generates instruction-following training data using only the Base model.

Pipeline:
1. Extract math problems from CCMath
2. Base model generates reasoning + answer
3. Filter by answer correctness
4. Generate training data for SFT
"""
import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class CCMathExtractor:
    """Extract math problems and answers from CCMath dataset."""

    def __init__(self, ccmath_path: str):
        self.ccmath_path = ccmath_path

    def extract_problems(self, max_samples: int = 1000) -> List[Dict]:
        """
        Extract problems with identifiable answers from CCMath.
        Returns list of {problem: str, answer: str, source: str}
        """
        problems = []

        with open(self.ccmath_path, 'r') as f:
            for i, line in enumerate(f):
                if len(problems) >= max_samples:
                    break

                try:
                    data = json.loads(line)
                    text = data.get('text', '')

                    # Try to extract problem-answer pairs
                    extracted = self._parse_text(text)
                    if extracted:
                        problems.append(extracted)
                except json.JSONDecodeError:
                    continue

        return problems

    def _parse_text(self, text: str) -> Optional[Dict]:
        """Parse different formats of math content."""
        # Pattern 1: Problem statement + Solution/Answer
        # Look for "Problem Statement" or question-like patterns

        patterns = [
            # Pattern: "## Problem Statement" followed by "## Solution"
            r'## Problem Statement\s*\n+(.*?)\n+## Solution\s*\n+(.*?)(?=##|\Z)',
            # Pattern: "**Question**" followed by "**Answer**"
            r'\*\*Question[:\*]*\*\*\s*(.*?)\*\*Answer[:\*]*\*\*\s*(.*?)(?=##|\*\*\Z)',
            # Pattern: Question ending with "?" followed by numbered answer
            r'(?:^|\n)([^\n]+\?)\s*\n+\d+[.\)]\s*(.*?)(?=\n\d+[.\)]|\n\n|\Z)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                problem = match.group(1).strip()
                answer = match.group(2).strip()

                # Filter: problem should be substantial
                if len(problem) > 20 and len(answer) > 10:
                    # Try to extract numerical answer
                    num_answer = self._extract_numerical_answer(answer)
                    return {
                        'problem': self._clean_text(problem),
                        'full_answer': self._clean_text(answer),
                        'numerical_answer': num_answer,
                        'source': text[:200]
                    }

        # Pattern 2: Look for boxed answers (LaTeX format)
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            # Find preceding problem text
            boxed_pos = boxed_match.start()
            preceding_text = text[:boxed_pos]

            # Extract last meaningful paragraph as problem
            paragraphs = [p.strip() for p in preceding_text.split('\n\n') if p.strip()]
            if paragraphs:
                problem_candidate = paragraphs[-1]
                # Skip if it looks like solution steps
                if not problem_candidate.startswith(('Step', 'Solution', 'Answer')):
                    return {
                        'problem': self._clean_text(problem_candidate),
                        'full_answer': boxed_match.group(1),
                        'numerical_answer': boxed_match.group(1),
                        'source': text[:200]
                    }

        return None

    def _extract_numerical_answer(self, text: str) -> Optional[str]:
        """Extract numerical answer from solution text."""
        # Look for final answer patterns
        patterns = [
            r'\\boxed\{([^}]+)\}',
            r'(?:answer|result|value|solution)[:\s]*([+-]?\d*\.?\d+)',
            r'(?:=|equals)[:\s]*([+-]?\d*\.?\d+)',
            r'final[:\s]*([+-]?\d*\.?\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove markdown artifacts
        text = re.sub(r'\*\*|\*|##', '', text)
        return text.strip()


class SelfInstructGenerator:
    """Generate instruction data using Base model."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "float16",
    ):
        print(f"Loading model from {model_path}...")

        compute_dtype = torch.float16 if dtype == "float16" else torch.float32
        if dtype == "bfloat16":
            compute_dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='left',
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=compute_dtype,
            device_map=device,
            trust_remote_code=True,
        )

        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def generate_reasoning(
        self,
        problem: str,
        prompt_style: str = "standard",
    ) -> Dict:
        """
        Generate reasoning and answer for a math problem.

        Returns: {reasoning: str, predicted_answer: str, raw_output: str}
        """
        prompts = {
            "standard": f"问题: {problem}\n\n请给出详细的解题步骤和最终答案。答案请用\\boxed{{}}格式标注。\n\n解答:",
            "cot": f"数学问题: {problem}\n\n请一步步思考这个问题:\n1. 分析题目要求\n2. 列出解题步骤\n3. 计算并验证\n4. 给出最终答案\n\n解题过程:",
            "instruction": f"""将以下数学问题改写为对话格式并提供解答。

原问题: {problem}

请按以下格式输出:
User: [问题内容]
Assistant: [解题过程和答案]

输出:""",
        }

        prompt = prompts.get(prompt_style, prompts["standard"])

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
            )

        raw_output = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )

        # Extract reasoning and answer
        predicted_answer = self._extract_answer(raw_output)

        return {
            'reasoning': raw_output,
            'predicted_answer': predicted_answer,
            'raw_output': raw_output,
            'prompt': prompt,
        }

    def generate_instruction_format(
        self,
        problem: str,
        reasoning: str,
    ) -> Dict:
        """
        Convert to instruction-following format for SFT.
        """
        return {
            "messages": [
                {"role": "system", "content": "你是一个数学问题解答助手，请给出详细的解题步骤和答案。"},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": reasoning}
            ]
        }

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract numerical/final answer from generated text."""
        patterns = [
            r'\\boxed\{([^}]+)\}',
            r'答案是[:\s]*([^\n]+)',
            r'答案[:\s]*([+-]?\d*\.?\d+)',
            r'(?:final|最终)(?:答案|结果)[:\s]*([^\n]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: look for last number in text
        numbers = re.findall(r'[+-]?\d*\.?\d+', text)
        if numbers:
            return numbers[-1]

        return None


class DataFilter:
    """Filter generated data by quality."""

    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance

    def filter_by_answer(
        self,
        generated: Dict,
        ground_truth: Optional[str],
    ) -> bool:
        """
        Check if generated answer matches ground truth.
        Returns True if correct (pass STaR filter).
        """
        if ground_truth is None:
            # No ground truth, accept all
            return True

        predicted = generated.get('predicted_answer')
        if predicted is None:
            return False

        # Try numerical comparison
        try:
            pred_num = float(predicted.replace(',', '').replace('°', ''))
            gt_num = float(ground_truth.replace(',', '').replace('°', ''))

            # Allow small tolerance for numerical answers
            if abs(pred_num - gt_num) <= self.tolerance * max(abs(gt_num), 1):
                return True
        except ValueError:
            # String comparison
            if predicted.strip() == ground_truth.strip():
                return True

        return False

    def filter_by_quality(self, generated: Dict) -> bool:
        """Check basic quality criteria."""
        reasoning = generated.get('reasoning', '')

        # Must have substantial reasoning
        if len(reasoning) < 50:
            return False

        # Should have some math content
        math_indicators = ['+', '-', '*', '/', '=', 'x', 'y', '解', '步骤', '计算']
        if not any(ind in reasoning for ind in math_indicators):
            return False

        return True


def run_self_instruct(
    ccmath_path: str,
    model_path: str,
    output_path: str,
    max_samples: int = 500,
    prompt_style: str = "standard",
    device: str = "auto",
    dtype: str = "float16",
    use_star_filter: bool = True,
):
    """
    Run full self-instruct pipeline.
    """
    print("=" * 60)
    print("Self-Instruct Pipeline for Math SFT")
    print("=" * 60)

    # Step 1: Extract problems from CCMath
    print("\n[Step 1] Extracting problems from CCMath...")
    extractor = CCMathExtractor(ccmath_path)
    problems = extractor.extract_problems(max_samples)
    print(f"  Extracted {len(problems)} problem-answer pairs")

    if len(problems) == 0:
        print("  Warning: No problems extracted. Using raw text samples...")
        # Fallback: use first N lines as problems
        with open(ccmath_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                data = json.loads(line)
                text = data.get('text', '')
                # Take first paragraph as "problem"
                first_para = text.split('\n\n')[0][:500]
                problems.append({
                    'problem': first_para,
                    'full_answer': '',
                    'numerical_answer': None,
                    'source': text[:100]
                })

    # Step 2: Generate reasoning using Base model
    print("\n[Step 2] Generating reasoning with Base model...")
    generator = SelfInstructGenerator(model_path, device, dtype)
    filter_checker = DataFilter()

    generated_data = []
    passed_data = []

    for p in tqdm(problems, desc="Generating"):
        try:
            result = generator.generate_reasoning(
                p['problem'],
                prompt_style=prompt_style,
            )

            # Quality filter
            if filter_checker.filter_by_quality(result):
                generated_data.append({
                    'problem': p['problem'],
                    'generated': result,
                    'ground_truth': p.get('numerical_answer'),
                })

                # STaR filter (answer correctness)
                if use_star_filter and filter_checker.filter_by_answer(
                    result, p.get('numerical_answer')
                ):
                    passed_data.append({
                        'problem': p['problem'],
                        'generated': result,
                        'ground_truth': p.get('numerical_answer'),
                    })
        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"  Generated {len(generated_data)} samples")
    print(f"  Passed STaR filter: {len(passed_data)} samples")

    # Step 3: Convert to instruction format
    print("\n[Step 3] Converting to instruction format...")
    sft_data = []

    # Use passed_data if STaR enabled and has samples, otherwise use all generated
    source_data = passed_data if (use_star_filter and len(passed_data) > 0) else generated_data

    for item in source_data:
        sft_item = generator.generate_instruction_format(
            item['problem'],
            item['generated']['reasoning'],
        )
        sft_data.append(sft_item)

    # Step 4: Save output
    print(f"\n[Step 4] Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w') as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"  Saved {len(sft_data)} training samples")

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    print(f"  Input problems: {len(problems)}")
    print(f"  Generated responses: {len(generated_data)}")
    print(f"  Quality passed: {len(generated_data)}")
    if use_star_filter:
        print(f"  Answer correct (STaR): {len(passed_data)}")
    print(f"  Final training data: {len(sft_data)}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    return sft_data


def main():
    parser = argparse.ArgumentParser(
        description="Self-Instruct Pipeline for Math SFT"
    )

    parser.add_argument(
        "--ccmath_path",
        type=str,
        default="nv-community_Nemotron-CC-Math-v1_4plus_first100000.jsonl",
        help="Path to CCMath dataset",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Qwen3-0.6B-Base",
        help="Path to base model",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for generated training data",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Maximum number of samples to process",
    )

    parser.add_argument(
        "--prompt_style",
        type=str,
        default="standard",
        choices=["standard", "cot", "instruction"],
        help="Prompt style for generation",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for model (auto, cuda, cpu)",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model",
    )

    parser.add_argument(
        "--use_star_filter",
        action="store_true",
        help="Use STaR filter (only keep correct answers)",
    )

    args = parser.parse_args()

    # Default output path
    if args.output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = f"outputs/self_instruct/self_instruct_{timestamp}.jsonl"

    run_self_instruct(
        ccmath_path=args.ccmath_path,
        model_path=args.model_path,
        output_path=args.output_path,
        max_samples=args.max_samples,
        prompt_style=args.prompt_style,
        device=args.device,
        dtype=args.dtype,
        use_star_filter=args.use_star_filter,
    )


if __name__ == "__main__":
    main()