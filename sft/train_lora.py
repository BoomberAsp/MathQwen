"""
LoRA Fine-tuning Script for Reasoning SFT.
Supports 0.6B models with configurable LoRA rank.
"""
import os
import sys
import json
import torch
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sft.dataset import load_training_dataset, collate_fn


@dataclass
class ModelArguments:
    """Arguments for model loading."""
    model_name_or_path: str = field(
        default="models/Qwen3-0.6B-Base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code"}
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention"}
    )


@dataclass
class DataArguments:
    """Arguments for data loading."""
    data_path: str = field(
        default="final_reasoning_sft_train.jsonl",
        metadata={"help": "Path to training data JSONL file"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    split_ratio: float = field(
        default=0.95,
        metadata={"help": "Train/eval split ratio"}
    )


@dataclass
class LoRAArguments:
    """Arguments for LoRA configuration."""
    lora_rank: int = field(
        default=32,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "LoRA alpha (scaling factor)"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Target modules for LoRA"}
    )


@dataclass
class TrainingArgs(TrainingArguments):
    """Training arguments."""
    output_dir: str = field(
        default="outputs",
        metadata={"help": "Output directory"}
    )
    num_train_epochs: int = field(
        default=6,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Training batch size per device"}
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Evaluation batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "Learning rate"}
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Warmup ratio"}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "LR scheduler type"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Logging steps"}
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy"}
    )
    eval_steps: int = field(
        default=10,
        metadata={"help": "Evaluation steps"}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "Save strategy"}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save steps"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Maximum number of checkpoints to keep"}
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Use FP16 mixed precision"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing"}
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "Number of dataloader workers"}
    )
    report_to: str = field(
        default="none",
        metadata={"help": "Report to (wandb, tensorboard, none)"}
    )


class LossHistoryCallback(TrainerCallback):
    """Callback to collect and plot loss curves."""

    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.train_steps = []
        self.eval_steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Collect losses from logs."""
        if logs is not None:
            # 训练 loss
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
                self.train_steps.append(state.global_step)
            # 评估 loss
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
                self.eval_steps.append(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """Plot loss curves and save to output directory."""
        if not self.train_losses:
            print("No training losses recorded.")
            return

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))

        if self.train_losses:
            plt.plot(self.train_steps, self.train_losses,
                     label='Training Loss', marker='o', markersize=3,
                     linestyle='-', linewidth=1)

        if self.eval_losses:
            plt.plot(self.eval_steps, self.eval_losses,
                     label='Validation Loss', marker='s', markersize=5,
                     linestyle='--', linewidth=2)

        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = os.path.join(args.output_dir, 'loss_curve.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Loss curve saved to: {output_path}")


class SaveBestModelCallback(TrainerCallback):
    """Save the model checkpoint when eval loss improves."""
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.best_loss = float("inf")
        self.best_step = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or args.local_rank not in (-1, 0):
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.best_step = state.global_step

            best_path = os.path.join(args.output_dir, "best_model")
            print(f"\n{'='*50}")
            print(f"New best eval loss: {eval_loss:.4f} at step {state.global_step}")
            print(f"Saving best adapter to {best_path}")
            print(f"{'='*50}\n")

            self.model.save_pretrained(best_path)
            self.tokenizer.save_pretrained(best_path)
            with open(os.path.join(best_path, "best_metrics.json"), "w") as f:
                json.dump({"step": state.global_step, "eval_loss": eval_loss}, f, indent=2)


class SavePeftModelCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        print("=" * 50)
        print("Training completed. Saving final LoRA model...")
        print("=" * 50)

        peft_model_path = os.path.join(args.output_dir, "peft_adapter")
        model.save_pretrained(peft_model_path)
        self.tokenizer.save_pretrained(peft_model_path)

        # 修正：使用 args.to_json_string() 安全序列化
        with open(os.path.join(peft_model_path, "training_args.json"), "w") as f:
            f.write(args.to_json_string())

        print(f"LoRA adapter saved to: {peft_model_path}")

        return control


def print_trainable_parameters(model):
    """Print the number of trainable parameters."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_param:.2f}%)")
    print(f"Total parameters: {all_param:,}")


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArgs, LoRAArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"lr{training_args.learning_rate}_bs{training_args.per_device_train_batch_size}_ep{training_args.num_train_epochs}_{timestamp}"
    original_output_dir = training_args.output_dir
    training_args.output_dir = os.path.join(original_output_dir, run_name)

    print("=" * 50)
    print("LoRA Fine-tuning Configuration")
    print("=" * 50)
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Data: {data_args.data_path}")

    print(f"LoRA Rank: {lora_args.lora_rank}")
    print(f"LoRA Alpha: {lora_args.lora_alpha}")
    print(f"Target Modules: {lora_args.target_modules}")
    print(f"Output Dir: {training_args.output_dir}")
    print("=" * 50)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")

    # Determine compute dtype
    compute_dtype = torch.float16 if training_args.fp16 else torch.float32

    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2" if model_args.use_flash_attn else "eager",
    )

    # Prepare model for k-bit training (if using quantization)
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_args.lora_rank,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=lora_args.target_modules,
        bias="none",
        inference_mode=False,
    )

    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print_trainable_parameters(model)

    # Load datasets
    print("Loading datasets...")
    train_dataset, eval_dataset = load_training_dataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        split_ratio=data_args.split_ratio,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Initialize trainer
    print("Initializing trainer...")
    save_callback = SavePeftModelCallback(tokenizer)
    loss_callback = LossHistoryCallback()
    best_callback = SaveBestModelCallback(tokenizer, model)  # 传入 model

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        callbacks=[save_callback, loss_callback, best_callback],
    )

    # Print training info
    print("=" * 50)
    print("Starting training...")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Warmup ratio: {training_args.warmup_ratio}")
    print(f"  Max length: {data_args.max_length}")
    print("=" * 50)

    # Train
    trainer.train()

    print("Training finished!")


if __name__ == "__main__":
    main()
