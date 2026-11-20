#!/usr/bin/env python3
"""
KTO (Kahneman-Treisman Optimization) Training for Qwen3-0.6B

使用 TRL 的 KTOTrainer 对 SFT 后的模型进行偏好对齐训练。
数据集需转换为 KTO 格式: {"prompt": str, "completion": str, "label": bool}

用法:
  python sft/train_kto_v2.py \
      --model_path models/Qwen3-0.6B-Base \
      --data_path outputs/kto_dataset.jsonl \
      --output_dir outputs/kto_output \
      --epochs 3 \
      --batch_size 4 \
      --gradient_accumulation_steps 4 \
      --lr 1e-6
"""
import os
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from trl import KTOConfig
from trl.experimental.kto import KTOTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def str2bool(v):
    """解析布尔参数"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def convert_kto_dataset(data_path: str) -> Dataset:
    """
    Convert our {messages: [...], label: bool} format to
    KTO format: {prompt: str, completion: str, label: bool}

    TRL's KTOTrainer with `dataset_format="kto"` expects this format.
    """
    records = []
    with open(data_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            messages = d.get('messages', [])

            # Build prompt from user message
            user_content = ''
            for msg in messages:
                if msg.get('role') == 'user':
                    user_content = msg['content']
                    break

            # Build completion from assistant message
            assistant_content = ''
            for msg in messages:
                if msg.get('role') == 'assistant':
                    assistant_content = msg['content']
                    break

            if user_content and assistant_content:
                records.append({
                    'prompt': user_content,
                    'completion': assistant_content,
                    'label': d.get('label', False),
                })

    return Dataset.from_list(records)


def parse_args():
    parser = argparse.ArgumentParser(description="KTO Training for Qwen3-0.6B")

    # 必需参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="本地模型路径")
    parser.add_argument("--data_path", type=str, required=True,
                        help="KTO 数据集 jsonl 路径")
    parser.add_argument("--output_dir", type=str, default="./kto_output",
                        help="模型输出路径")

    # 训练参数
    parser.add_argument("--max_length", type=int, default=1024,
                        help="最大序列长度")
    parser.add_argument("--max_completion_length", type=int, default=512,
                        help="最大 completion 长度")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="学习率（KTO推荐5e-7~5e-6）")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="warmup比例")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KTO beta 参数（控制 KL 正则强度）")
    parser.add_argument("--desirable_weight", type=float, default=1.0,
                        help="期望样本(label=True)的权重")
    parser.add_argument("--undesirable_weight", type=float, default=1.0,
                        help="不期望样本(label=False)的权重")

    # 精度参数
    parser.add_argument("--fp16", type=str2bool, default=False,
                        help="使用FP16混合精度")
    parser.add_argument("--bf16", type=str2bool, default=False,
                        help="使用BF16混合精度")

    # 其他参数
    parser.add_argument("--eval_ratio", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="早停耐心值")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help="报告工具 (tensorboard, wandb, none)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存步数")

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    precision_mode = "FP32" if not args.fp16 and not args.bf16 else ("BF16" if args.bf16 else "FP16")

    print("=" * 60)
    print("KTO Training for Qwen3-0.6B")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Beta: {args.beta}")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation_steps} = {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Mixed precision: {precision_mode}")
    print("=" * 60)

    # 1. 加载 Tokenizer
    logger.info(f"Loading tokenizer from {args.model_path}")

    tokenizer_path = args.model_path
    if "checkpoint-" in args.model_path:
        tokenizer_path = "models/Qwen3-0.6B-Base"
        logger.info(f"Using tokenizer from base model: {tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token set to eos_token")

    # 2. 加载并转换数据集
    logger.info(f"Loading and converting KTO dataset from {args.data_path}")
    dataset = convert_kto_dataset(args.data_path)
    logger.info(f"Loaded {len(dataset)} samples")

    # 统计 label 分布
    true_count = sum(1 for d in dataset if d.get("label") is True)
    false_count = sum(1 for d in dataset if d.get("label") is False)
    logger.info(f"  True (desirable):  {true_count}")
    logger.info(f"  False (undesirable): {false_count}")

    if len(dataset) == 0:
        logger.error("No samples found. Exiting.")
        return

    # 划分训练/验证集
    split = dataset.train_test_split(
        test_size=args.eval_ratio,
        seed=args.seed,
        shuffle=True,
    )
    train_dataset = split["train"]
    eval_dataset = split["test"]

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")

    # 3. 加载模型
    logger.info(f"Loading model from {args.model_path} ({precision_mode})")

    if args.fp16:
        torch_dtype = torch.float16
    elif args.bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # KTO 训练设置
    model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # 4. KTOConfig
    kto_config = KTOConfig(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to=args.report_to,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        remove_unused_columns=False,
        max_length=args.max_length,
        beta=args.beta,
        desirable_weight=args.desirable_weight,
        undesirable_weight=args.undesirable_weight,
    )

    # 5. KTOTrainer
    trainer = KTOTrainer(
        model=model,
        args=kto_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # 6. 开始训练
    logger.info("\n" + "=" * 60)
    logger.info("Starting KTO training...")
    logger.info("=" * 60)

    trainer.train()

    # 7. 保存最终模型
    logger.info(f"\nSaving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 保存训练信息
    train_info = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "beta": args.beta,
        "precision": precision_mode,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "desirable_samples": sum(1 for d in train_dataset if d.get("label") is True),
        "undesirable_samples": sum(1 for d in train_dataset if d.get("label") is False),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "train_info.json"), "w") as f:
        json.dump(train_info, f, indent=2)

    print("\n" + "=" * 60)
    print("KTO Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
