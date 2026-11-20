"""
Generate KTO (Kahneman-Treisman Optimization) dataset using SFT-fine-tuned model.

Pipeline:
1. Load SFT model (model_checkpoint378.safetensors) with Qwen3-0.6B-Base as base
2. Load math problems from data/valid_1000.jsonl and data/stage2_math.jsonl
3. Build prompts following the same style as valid_1000.jsonl
4. Generate responses using the SFT model
5. Use a stronger commercial LLM to judge each response as True (correct) / False (incorrect)
6. Output KTO-format dataset with {messages, label} pairs

Usage:
  python sft/generate_kto_dataset.py \
      --base_model_path models/Qwen3-0.6B-Base \
      --sft_checkpoint_path models/model_checkpoint378.safetensors \
      --output_path outputs/kto_dataset.jsonl \
      --max_samples 500
"""
import os
import json
import argparse
from typing import Dict, List, Optional
from tqdm import tqdm

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_valid_1000(path: str, max_samples: int) -> List[Dict]:
    """
    Load from data/valid_1000.jsonl format (multi-line JSON, one object per block):
      {"question": "...", "solution": "...", "answer": "...", ...}
    Returns list of {"problem": str, "answer": str, "source": str}.
    """
    problems = []
    with open(path, 'r') as f:
        content = f.read()

    # Split by "}\n{" and re-wrap each block
    blocks = content.split('}\n{')
    for block in blocks:
        if len(problems) >= max_samples:
            break
        if not block.startswith('{'):
            block = '{' + block
        if not block.endswith('}'):
            block = block + '}'
        try:
            d = json.loads(block)
            q = d.get('question', '').strip()
            if q:
                problems.append({
                    'problem': q,
                    'answer': d.get('answer', ''),
                    'solution': d.get('solution', ''),
                    'source': 'valid_1000',
                })
        except json.JSONDecodeError:
            continue
    return problems


def load_stage2_math(path: str, max_samples: int) -> List[Dict]:
    """
    Load from data/stage2_math.jsonl format:
      {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    Returns list of {"problem": str, "answer": str, "source": str}.
    """
    problems = []
    with open(path, 'r') as f:
        for line in f:
            if len(problems) >= max_samples:
                break
            try:
                d = json.loads(line)
                messages = d.get('messages', [])
                user_content = ''
                for msg in messages:
                    if msg.get('role') == 'user':
                        user_content = msg['content']
                        break
                if user_content:
                    problems.append({
                        'problem': user_content,
                        'answer': '',
                        'solution': '',
                        'source': 'stage2_math',
                    })
            except json.JSONDecodeError:
                continue
    return problems


# ---------------------------------------------------------------------------
# Prompt builder – matches the style of valid_1000.jsonl
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a math problem solver. Please provide step-by-step reasoning and a final answer. "
    "End your response with #### followed by the final answer, and also wrap the final answer in \\boxed{}."
)

USER_PROMPT_TEMPLATE = "{question}"

# Expected assistant answer style (what we ask the model to produce):
#   <step-by-step reasoning> #### <answer> The final answer is \boxed{<answer>}


def build_prompt(problem: str) -> str:
    """Build the full prompt (without chat template) for the SFT model."""
    return f"{USER_PROMPT_TEMPLATE.format(question=problem)}"


def build_chat_messages(problem: str, assistant_content: str) -> List[Dict]:
    """Build the messages list for KTO output, matching valid_1000.jsonl style."""
    return [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": assistant_content},
    ]


# ---------------------------------------------------------------------------
# SFT model generation
# ---------------------------------------------------------------------------

class SFTGenerator:
    """Generate responses using the SFT-fine-tuned model."""

    def __init__(
        self,
        base_model_path: str,
        sft_checkpoint_path: Optional[str] = None,
        device: str = "auto",
        dtype: str = "float16",
    ):
        compute_dtype = torch.float16 if dtype == "float16" else torch.bfloat16 if dtype == "bfloat16" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=True, padding_side='left',
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=compute_dtype, device_map=device, trust_remote_code=True,
        )

        # Load SFT checkpoint weights if provided
        if sft_checkpoint_path:
            if os.path.isdir(sft_checkpoint_path):
                # Full model directory (e.g. KTO output) – already loaded above
                print(f"Using full model directory: {sft_checkpoint_path}")
            else:
                # Safetensors file – load weights on top of base model
                print(f"Loading SFT checkpoint from {sft_checkpoint_path}...")
                state_dict = load_file(sft_checkpoint_path, device='cpu')
                self.model.load_state_dict(state_dict, strict=False)
        else:
            print("No additional checkpoint provided, using base model only.")

        self.model.eval()
        print("SFT model loaded successfully.")

    def generate(
        self,
        problem: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response for the given math problem using the prompt style of valid_1000.jsonl."""
        # Build messages in chat format
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        return generated


# ---------------------------------------------------------------------------
# Judge via commercial LLM
# ---------------------------------------------------------------------------

from openai import OpenAI


def judge_with_llm(problem: str, response: str) -> bool:
    """
    Use DeepSeek (OpenAI-compatible API) to judge whether the SFT model's response
    correctly solves the math problem.

    Returns True if the response is correct, False otherwise.
    """
    client = OpenAI(
        api_key="sk-cb852fc94a3b4b05adfc2f9ef88c698b",
        base_url="https://api.deepseek.com/v1",
    )

    # Truncate to avoid overly long prompts causing empty responses
    problem_trunc = problem[:500] if len(problem) > 500 else problem
    response_trunc = response[:2000] if len(response) > 2000 else response

    system_prompt = "You are a math grader. Judge if the solution is correct. Reply only True or False."
    user_prompt = f"Problem: {problem_trunc}\nSolution: {response_trunc}\nIs the solution correct? Reply only True or False."

    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    result = resp.choices[0].message.content.strip()
    return "True" in result and "False" not in result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate KTO dataset from SFT model")
    parser.add_argument("--base_model_path", type=str, default="models/Qwen3-0.6B-Base")
    parser.add_argument("--sft_checkpoint_path", type=str, default=None,
                        help="SFT checkpoint (.safetensors) or full model dir (KTO output). None = use base model")
    parser.add_argument("--valid_1000_path", type=str, default="data/valid_1000.jsonl")
    parser.add_argument("--stage2_math_path", type=str, default="data/stage2_math.jsonl")
    parser.add_argument("--output_path", type=str, default="outputs/kto_dataset.jsonl")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Max samples per source file")
    parser.add_argument("--max_samples_valid", type=int, default=None,
                        help="Override max samples from valid_1000 (default: use --max_samples)")
    parser.add_argument("--max_samples_stage2", type=int, default=None,
                        help="Override max samples from stage2_math (default: use --max_samples)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle combined problems before generation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

    max_valid = args.max_samples_valid or args.max_samples
    max_stage2 = args.max_samples_stage2 or args.max_samples

    # Load problems from both sources
    print(f"Loading from {args.valid_1000_path}...")
    problems_valid = load_valid_1000(args.valid_1000_path, max_valid)
    print(f"  -> {len(problems_valid)} problems from valid_1000")

    print(f"Loading from {args.stage2_math_path}...")
    problems_stage2 = load_stage2_math(args.stage2_math_path, max_stage2)
    print(f"  -> {len(problems_stage2)} problems from stage2_math")

    all_problems = problems_valid + problems_stage2

    if args.shuffle:
        import random
        random.seed(args.seed)
        random.shuffle(all_problems)

    print(f"Total: {len(all_problems)} problems.")
    if len(all_problems) == 0:
        print("No problems found. Exiting.")
        return

    # Initialize SFT generator
    generator = SFTGenerator(
        base_model_path=args.base_model_path,
        sft_checkpoint_path=args.sft_checkpoint_path,
        device=args.device,
        dtype=args.dtype,
    )

    # Resume support
    completed_indices = set()
    existing_samples = []
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    existing_samples.append(json.loads(line))
                    completed_indices.add(i)
        print(f"Resuming from existing output, {len(completed_indices)} items already done.")

    # Generate and judge
    kto_samples = existing_samples

    for i, item in enumerate(tqdm(all_problems, desc="Generating & Judging")):
        if i in completed_indices:
            continue

        problem = item['problem']

        try:
            # Step 1: Generate response using SFT model
            response = generator.generate(
                problem,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            if not response or len(response.strip()) < 10:
                # Still record as failed (label=False) for KTO
                kto_sample = {
                    "messages": build_chat_messages(problem, response or ""),
                    "label": False,
                    "source": item['source'],
                }
                kto_samples.append(kto_sample)
                with open(args.output_path, 'a') as f:
                    f.write(json.dumps(kto_sample, ensure_ascii=False) + '\n')
                continue

            # Step 2: Judge with commercial LLM
            is_correct = judge_with_llm(problem, response)

            # Step 3: Build KTO sample
            kto_sample = {
                "messages": build_chat_messages(problem, response),
                "label": is_correct,
                "source": item['source'],
            }
            kto_samples.append(kto_sample)

            # Incremental write for resume support
            with open(args.output_path, 'a') as f:
                f.write(json.dumps(kto_sample, ensure_ascii=False) + '\n')

        except NotImplementedError as e:
            print(f"\n{e}")
            print("Stopping. Please implement the judge function.")
            break
        except Exception as e:
            print(f"\nError at sample {i}: {e}")
            continue

    # Summary
    true_count = sum(1 for s in kto_samples if s.get('label') is True)
    false_count = sum(1 for s in kto_samples if s.get('label') is False)
    print(f"\nKTO dataset generation complete.")
    print(f"  Total samples: {len(kto_samples)}")
    print(f"  True (correct):   {true_count}")
    print(f"  False (incorrect): {false_count}")
    print(f"  Saved to: {args.output_path}")


if __name__ == "__main__":
    main()
