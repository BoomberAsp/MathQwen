#!/usr/bin/env python3
"""
Full-parameter SFT for Qwen3-0.6B
全量微调（非LoRA）

修复内容:
- 正确的布尔参数解析
- 支持FP32/BF16精度选择
- 添加验证集和Early Stopping
- TensorBoard监控
- 过拟合监控
"""
import os
import sys
import json
import math
import argparse
import random
import logging
from typing import Dict, List, Optional
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    EarlyStoppingCallback,
    set_seed,
)

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


class OverfitMonitorCallback(TrainerCallback):
    """监控过拟合"""

    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.best_eval_loss = float('inf')
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if 'loss' in logs:
                self.train_losses.append((state.epoch, logs['loss']))
            if 'eval_loss' in logs:
                self.eval_losses.append((state.epoch, logs['eval_loss']))

                # 检查过拟合
                if len(self.train_losses) > 0:
                    recent_train = self.train_losses[-1][1]
                    recent_eval = logs['eval_loss']
                    gap = abs(recent_train - recent_eval)

                    if gap > self.threshold:
                        logger.warning(f"\n⚠️ Warning: Potential overfitting detected!")
                        logger.warning(f"   Train loss: {recent_train:.4f}")
                        logger.warning(f"   Eval loss: {recent_eval:.4f}")
                        logger.warning(f"   Gap: {gap:.4f} > threshold {self.threshold}")

                    # 检查验证Loss上升
                    if recent_eval > self.best_eval_loss:
                        logger.warning(f"\n⚠️ Warning: Eval loss increased from {self.best_eval_loss:.4f} to {recent_eval:.4f}")

                    self.best_eval_loss = min(self.best_eval_loss, recent_eval)


def parse_args():
    parser = argparse.ArgumentParser(description="Full Fine-tuning for Qwen3-0.6B")

    # 必需参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="本地模型路径")
    parser.add_argument("--data_path", type=str, required=True,
                        help="本地jsonl数据路径")
    parser.add_argument("--output_dir", type=str, default="./sft_output",
                        help="模型输出路径")

    # 训练参数
    parser.add_argument("--max_length", type=int, default=1024,
                        help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="学习率（全量微调推荐1e-5~5e-5）")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="warmup比例")

    # 精度参数
    parser.add_argument("--fp16", type=str2bool, default=False,
                        help="使用FP16混合精度")
    parser.add_argument("--bf16", type=str2bool, default=False,
                        help="使用BF16混合精度")

    # 其他参数
    parser.add_argument("--eval_ratio", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--early_stopping_patience", type=int, default=2,
                        help="早停耐心值")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help="报告工具 (tensorboard, wandb, none)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存步数")

    return parser.parse_args()


def process_data(examples, tokenizer, max_length):
    """
    针对 ChatML 格式的处理。
    格式: <|im_start|>user\n...\n<im_end|>\n<|im_start|>assistant\n...\n<|im_end|>
    """
    input_ids_list = []
    labels_list = []

    for messages in examples["messages"]:
        # messages 格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        # 构建 ChatML 格式
        instruction = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

        # 处理 assistant 回复（可能有多个消息）
        if len(messages) > 1:
            response = f"{messages[1]['content']}<|im_end|>"
        else:
            # 如果没有assistant回复，跳过
            continue

        # 编码
        tokenized_instruction = tokenizer(instruction, add_special_tokens=False)
        tokenized_response = tokenizer(response, add_special_tokens=False)

        # 拼接
        input_ids = tokenized_instruction["input_ids"] + tokenized_response["input_ids"]
        if tokenizer.eos_token_id is not None:
            input_ids = input_ids + [tokenizer.eos_token_id]

        # 构建 Labels (忽略 User 部分的损失，用 -100 填充)
        labels = [-100] * len(tokenized_instruction["input_ids"])
        labels = labels + tokenized_response["input_ids"]
        if tokenizer.eos_token_id is not None:
            labels = labels + [tokenizer.eos_token_id]

        # 截断
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {"input_ids": input_ids_list, "labels": labels_list}


def main():
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    precision_mode = "FP32" if not args.fp16 and not args.bf16 else ("BF16" if args.bf16 else "FP16")

    print("=" * 60)
    print("Full-parameter SFT for Qwen3-0.6B")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size} × {args.gradient_accumulation_steps} = {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Mixed precision: {precision_mode}")
    print(f"Eval ratio: {args.eval_ratio}")
    print("=" * 60)

    # 1. 加载 Tokenizer
    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token set to eos_token")

    # 2. 加载并处理数据集
    logger.info(f"Loading dataset from {args.data_path}")
    raw_dataset = load_dataset("json", data_files=args.data_path, split="train")
    logger.info(f"Loaded {len(raw_dataset)} samples")

    # 处理数据
    processed_dataset = raw_dataset.map(
        lambda x: process_data(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Processing data"
    )

    # 过滤掉空数据
    processed_dataset = processed_dataset.filter(
        lambda x: len(x["input_ids"]) > 0,
        desc="Filtering empty samples"
    )
    logger.info(f"After processing: {len(processed_dataset)} valid samples")

    # 划分训练/验证集
    split_dataset = processed_dataset.train_test_split(
        test_size=args.eval_ratio,
        seed=args.seed,
        shuffle=True
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")

    # 3. 加载模型
    logger.info(f"Loading model from {args.model_path} ({precision_mode})")

    config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    # 根据精度设置加载模型
    if args.fp16:
        torch_dtype = torch.float16
    elif args.bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # 全量微调设置
    model.enable_input_require_grads()
    model.config.use_cache = False

    # 梯度检查点
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # 4. 训练参数
    training_args = TrainingArguments(
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
    )

    # 5. Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=args.max_length,
    )

    # 6. Callbacks
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
        OverfitMonitorCallback(threshold=0.3),
    ]

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # 8. 开始训练
    logger.info("\n" + "=" * 60)
    logger.info("Starting full-parameter SFT...")
    logger.info("=" * 60)

    trainer.train()

    # 9. 保存最终模型
    logger.info(f"\nSaving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 保存训练信息
    train_info = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "precision": precision_mode,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "train_info.json"), "w") as f:
        json.dump(train_info, f, indent=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()