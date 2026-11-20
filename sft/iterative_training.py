"""
Iterative Self-Training Pipeline.

After first SFT round, use the fine-tuned model to generate better data,
then perform second round of SFT.

Usage:
1. First round: self_instruct.py -> generate initial data
2. First SFT: train_lora.py -> get instruction-following model
3. Merge weights: combine_weight.py -> standalone model
4. Second round: iterative_training.py -> better quality data
5. Second SFT: train_lora.py with new data
"""
import os
import sys
import json
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class IterativeTrainer:
    """Generate better training data using fine-tuned model."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "float16",
    ):
        print(f"Loading model from {model_path}...")

        compute_dtype = torch.float16 if dtype == "float16" else torch.bfloat16 if dtype == "bfloat16" else torch.float32

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

        self.model.eval()
        print("Model loaded successfully.")

    def generate_batch(
        self,
        problems: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> List[Dict]:
        """Generate responses for a batch of problems."""

        prompts = [
            self._format_prompt(p) for p in problems
        ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        results = []
        for i, output in enumerate(outputs):
            decoded = self.tokenizer.decode(
                output[inputs['input_ids'][i].shape[1]:],
                skip_special_tokens=True,
            )
            results.append({
                'problem': problems[i],
                'response': decoded,
                'predicted_answer': self._extract_answer(decoded),
            })

        return results

    def _format_prompt(self, problem: str) -> str:
        """Format problem as instruction prompt."""
        return f"""请解答以下数学问题，给出详细的解题步骤。

问题: {problem}

请按以下格式回答:
1. 分析问题
2. 列出解题步骤
3. 进行计算
4. 给出最终答案（用\\boxed{{答案}}标注）

解答:"""

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract final answer from response."""
        patterns = [
            r'\\boxed\{([^}]+)\}',
            r'最终答案[:\s]*([^\n]+)',
            r'答案[:\s]*([+-]?\d*\.?\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: last number
        numbers = re.findall(r'[+-]?\d*\.?\d+', text)
        return numbers[-1] if numbers else None

    def generate_instruction_data(
        self,
        problems: List[str],
        responses: List[Dict],
    ) -> List[Dict]:
        """Convert to SFT format."""
        sft_data = []
        for prob, resp in zip(problems, responses):
            sft_data.append({
                "messages": [
                    {"role": "system", "content": "你是一个数学问题解答助手，请给出详细的解题步骤和答案。"},
                    {"role": "user", "content": prob},
                    {"role": "assistant", "content": resp['response']}
                ]
            })
        return sft_data


def load_problems_from_jsonl(path: str, max_samples: int = 1000) -> List[str]:
    """Load problems from various JSONL formats."""
    problems = []

    with open(path, 'r') as f:
        for line in f:
            if len(problems) >= max_samples:
                break

            data = json.loads(line)

            # Format 1: messages format (existing SFT data)
            if 'messages' in data:
                for msg in data['messages']:
                    if msg['role'] == 'user':
                        problems.append(msg['content'])
                        break

            # Format 2: problem field
            elif 'problem' in data:
                problems.append(data['problem'])

            # Format 3: question field
            elif 'question' in data:
                problems.append(data['question'])

            # Format 4: text field (raw CCMath)
            elif 'text' in data:
                text = data['text']
                # Extract first meaningful paragraph
                paragraphs = text.split('\n\n')
                for p in paragraphs:
                    if len(p) > 50 and not p.startswith('#'):
                        problems.append(p[:500])
                        break

    return problems


def run_iterative_training(
    model_path: str,
    problem_source: str,
    output_path: str,
    max_samples: int = 500,
    batch_size: int = 8,
    device: str = "auto",
    dtype: str = "float16",
    temperature: float = 0.7,
):
    """
    Run iterative self-training.
    """
    print("=" * 60)
    print("Iterative Self-Training Pipeline")
    print("=" * 60)

    # Load problems
    print(f"\n[Step 1] Loading problems from {problem_source}...")
    problems = load_problems_from_jsonl(problem_source, max_samples)
    print(f"  Loaded {len(problems)} problems")

    # Initialize model
    print(f"\n[Step 2] Loading fine-tuned model...")
    trainer = IterativeTrainer(model_path, device, dtype)

    # Generate responses
    print(f"\n[Step 3] Generating responses...")
    all_responses = []

    for i in tqdm(range(0, len(problems), batch_size), desc="Generating"):
        batch = problems[i:i+batch_size]
        responses = trainer.generate_batch(
            batch,
            temperature=temperature,
        )
        all_responses.extend(responses)

    print(f"  Generated {len(all_responses)} responses")

    # Convert to SFT format
    print(f"\n[Step 4] Converting to SFT format...")
    sft_data = trainer.generate_instruction_data(problems, all_responses)

    # Filter by quality
    filtered_data = []
    for item in sft_data:
        content = item['messages'][-1]['content']
        # Must have reasonable length
        if len(content) > 100:
            # Should have math indicators
            math_indicators = ['+', '-', '=', '解', '步骤', '计算', '\\boxed']
            if any(ind in content for ind in math_indicators):
                filtered_data.append(item)

    print(f"  Quality filtered: {len(filtered_data)} samples")

    # Save output
    print(f"\n[Step 5] Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w') as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    print(f"  Input problems: {len(problems)}")
    print(f"  Generated responses: {len(all_responses)}")
    print(f"  Quality passed: {len(filtered_data)}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    return filtered_data


def main():
    parser = argparse.ArgumentParser(
        description="Iterative Self-Training Pipeline"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model (merged)",
    )

    parser.add_argument(
        "--problem_source",
        type=str,
        default="nv-community_Nemotron-CC-Math-v1_4plus_first100000.jsonl",
        help="Source of problems (JSONL file)",
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
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for model",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )

    args = parser.parse_args()

    if args.output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = f"outputs/iterative/round2_{timestamp}.jsonl"

    run_iterative_training(
        model_path=args.model_path,
        problem_source=args.problem_source,
        output_path=args.output_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()