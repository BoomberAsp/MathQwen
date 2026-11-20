"""
Merge LoRA weights into base model.
Combines LoRA adapter weights with the base model and saves as a standalone model.
"""
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def get_lora_config_name(lora_path: str) -> str:
    """Extract config name from LoRA path or adapter_config.json."""
    lora_path = Path(lora_path)

    # Try to get name from parent directory pattern (e.g., lr0.0002_bs4_ep6_20260423_111926)
    parent_name = lora_path.parent.name
    if parent_name.startswith("lr") and "_" in parent_name:
        return parent_name

    # Try to read from adapter_config.json for learning rate info
    config_path = lora_path / "adapter_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        # Use timestamp if no training info available
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"merged_{timestamp}"

    # Default: use directory name
    return lora_path.name


def merge_lora_weights(
    base_model_path: str,
    lora_path: str,
    output_dir: str = None,
    device: str = "auto",
    dtype: str = "float16",
):
    """
    Merge LoRA weights into base model.

    Args:
        base_model_path: Path to the base model
        lora_path: Path to the LoRA adapter
        output_dir: Output directory for merged model
        device: Device to use for loading
        dtype: Data type for the model
    """
    # Determine output directory
    if output_dir is None:
        config_name = get_lora_config_name(lora_path)
        output_dir = os.path.join("outputs", "combine_models", config_name)

    print("=" * 50)
    print("LoRA Weight Merging")
    print("=" * 50)
    print(f"Base model: {base_model_path}")
    print(f"LoRA adapter: {lora_path}")
    print(f"Output directory: {output_dir}")
    print("=" * 50)

    # Read LoRA config
    adapter_config_path = os.path.join(lora_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        print(f"LoRA rank: {adapter_config.get('r', 'N/A')}")
        print(f"LoRA alpha: {adapter_config.get('lora_alpha', 'N/A')}")
        print(f"Target modules: {adapter_config.get('target_modules', 'N/A')}")
    else:
        print(f"Warning: adapter_config.json not found in {lora_path}")

    # Determine compute dtype
    compute_dtype = torch.float16 if dtype == "float16" else torch.float32
    if dtype == "bfloat16":
        compute_dtype = torch.bfloat16

    # Load base model
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=compute_dtype,
        device_map=device,
        trust_remote_code=True,
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )

    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_path)

    # Merge and unload
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save merged model
    print(f"\nSaving merged model to {output_dir}...")
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,  # Use safetensors format
    )
    tokenizer.save_pretrained(output_dir)

    # Save merge info
    merge_info = {
        "base_model": base_model_path,
        "lora_path": lora_path,
        "merge_time": datetime.now().isoformat(),
        "dtype": dtype,
    }
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r") as f:
            merge_info["lora_config"] = json.load(f)

    with open(os.path.join(output_dir, "merge_info.json"), "w") as f:
        json.dump(merge_info, f, indent=2)

    print("=" * 50)
    print("Merge completed successfully!")
    print(f"Merged model saved to: {output_dir}")
    print("=" * 50)

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into base model"
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="models/Qwen3-0.6B-Base",
        help="Path to the base model",
    )

    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to the LoRA adapter (e.g., outputs/lr0.0002_bs4_ep6_20260423_111926/best_model)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for merged model (default: outputs/combine_models/<lora_config_name>)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, cuda:0)",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for the model",
    )

    args = parser.parse_args()

    merge_lora_weights(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()