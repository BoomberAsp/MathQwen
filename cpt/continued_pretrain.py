"""
继续预训练脚本 (Continued Pre-Training)
针对数据受限场景的多轮训练策略

策略要点:
1. 多轮次训练 (4-6 epochs)
2. 课程学习排序
3. 数据增强
4. 动态shuffle
5. 过拟合监控
"""
import os
import sys
import json
import random
import math
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint


@dataclass
class DataArguments:
    """数据加载参数"""
    data_path: str = field(
        default="nv-community_Nemotron-CC-Math-v1_4plus_first100000.jsonl",
        metadata={"help": "预训练数据路径"}
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "最大序列长度"}
    )
    quality_threshold: int = field(
        default=3,
        metadata={"help": "FineMath质量评分阈值"}
    )
    use_curriculum: bool = field(
        default=True,
        metadata={"help": "是否使用课程学习排序"}
    )
    curriculum_stages: int = field(
        default=3,
        metadata={"help": "课程学习阶段数"}
    )


@dataclass
class ModelArguments:
    """模型参数"""
    model_name_or_path: str = field(
        default="models/Qwen3-0.6B-Base",
        metadata={"help": "基础模型路径"}
    )
    trust_remote_code: bool = field(
        default=True,
    )


@dataclass
class CPTArguments(TrainingArguments):
    """继续预训练参数"""
    output_dir: str = field(
        default="outputs/cpt",
    )
    num_train_epochs: int = field(
        default=6,
        metadata={"help": "训练轮次（数据受限场景推荐4-6）"}
    )
    per_device_train_batch_size: int = field(
        default=8,
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "有效batch = 8*4 = 32"}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "继续预训练用较小学习率"}
    )
    weight_decay: float = field(
        default=0.1,
        metadata={"help": "较高正则化抑制过拟合"}
    )
    warmup_ratio: float = field(
        default=0.05,
    )
    lr_scheduler_type: str = field(
        default="cosine",
    )
    logging_steps: int = field(
        default=50,
    )
    save_strategy: str = field(
        default="epoch",
    )
    eval_strategy: str = field(
        default="epoch",
    )
    early_stopping_patience: int = field(
        default=3,
        metadata={"help": "Early stopping耐心值"}
    )
    early_stopping_threshold: float = field(
        default=0.01,
        metadata={"help": "Early stopping阈值"}
    )
    dataloader_num_workers: int = field(
        default=4,
    )
    fp16: bool = field(
        default=True,
    )
    gradient_checkpointing: bool = field(
        default=True,
    )
    report_to: str = field(
        default="tensorboard",
        metadata={"help": "报告工具 (tensorboard, wandb, none)"},
    )


class MathTextDataset(Dataset):
    """数学预训练数据集"""

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        quality_threshold: int = 3,
        use_curriculum: bool = True,
        curriculum_stages: int = 3,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # 加载并筛选数据
        print(f"Loading data from {data_path}...")
        raw_data = self._load_and_filter(data_path, quality_threshold)
        print(f"  Loaded {len(raw_data)} samples after quality filtering")

        # 课程学习排序
        if use_curriculum:
            raw_data = self._curriculum_sort(raw_data, curriculum_stages)
            print(f"  Applied curriculum learning with {curriculum_stages} stages")

        # 存储文本，让DataCollator处理tokenization和padding
        print("Storing texts...")
        for item in raw_data:
            self.data.append(item['text'])

        print(f"  Total training samples: {len(self.data)}")

    def _load_and_filter(self, path: str, threshold: int) -> List[Dict]:
        """加载并按质量筛选"""
        data = []
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                score = item.get('metadata', {}).get('finemath_int_scores', 0)
                if score >= threshold:
                    data.append(item)
        return data

    def _curriculum_sort(self, data: List[Dict], stages: int) -> List[Dict]:
        """课程学习排序 - 简单→复杂"""

        def estimate_difficulty(text: str) -> float:
            """估算文本难度"""
            # 基于多种特征
            length = len(text)

            # 数学符号密度
            math_symbols = ['∫', '∑', '∂', '∇', '√', 'π', '∞', '±', '×', '÷']
            symbol_count = sum(1 for s in math_symbols if s in text)

            # LaTeX公式数量
            latex_count = text.count('\\') + text.count('$')

            # 证明相关词汇
            proof_words = ['prove', '证明', 'theorem', '定理', 'lemma', '引理']
            proof_count = sum(1 for w in proof_words if w.lower() in text.lower())

            # 综合难度评分
            difficulty = (
                length / 1000 +  # 长度因素
                symbol_count * 2 +  # 符号复杂度
                latex_count * 0.5 +  # 公式复杂度
                proof_count * 5  # 证明复杂度
            )
            return difficulty

        # 计算每条数据的难度
        difficulties = [(item, estimate_difficulty(item['text'])) for item in data]

        # 按难度排序
        difficulties.sort(key=lambda x: x[1])

        # 分阶段组织数据
        n = len(difficulties)
        stage_size = n // stages
        sorted_data = []

        for stage in range(stages):
            start = stage * stage_size
            end = (stage + 1) * stage_size if stage < stages - 1 else n
            stage_data = [d[0] for d in difficulties[start:end]]
            sorted_data.extend(stage_data)

            # 每阶段打乱内部顺序
            random.shuffle(stage_data)

        return sorted_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """返回tokenized input_ids（不padding，让collator处理）"""
        text = self.data[idx]
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors=None,
        )
        return {'input_ids': tokens['input_ids']}


class ShuffleCallback(TrainerCallback):
    """每个epoch结束后重新shuffle数据"""

    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        """每个epoch开始时shuffle"""
        if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'shuffle'):
            train_dataloader.sampler.shuffle()
            print(f"Epoch {state.epoch + 1}: Data shuffled")


class OverfitMonitorCallback(TrainerCallback):
    """监控过拟合"""

    def __init__(self, threshold=0.5):
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
                    gap = recent_train - recent_eval

                    if gap > self.threshold:
                        print(f"\n⚠️ Warning: Potential overfitting detected!")
                        print(f"   Train loss: {recent_train:.4f}")
                        print(f"   Eval loss: {recent_eval:.4f}")
                        print(f"   Gap: {gap:.4f} > threshold {self.threshold}")

                    # 检查验证Loss上升
                    if recent_eval > self.best_eval_loss:
                        print(f"\n⚠️ Warning: Eval loss increased from {self.best_eval_loss:.4f} to {recent_eval:.4f}")

                    self.best_eval_loss = min(self.best_eval_loss, recent_eval)


def compute_metrics(eval_preds):
    """计算评估指标"""
    # 对于Causal LM，主要监控perplexity
    loss = eval_preds.loss if hasattr(eval_preds, 'loss') else eval_preds[0]
    perplexity = math.exp(loss) if loss < 10 else float('inf')
    return {'perplexity': perplexity}


def main():
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

    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='nv-community_Nemotron-CC-Math-v1_4plus_first100000.jsonl')
    parser.add_argument('--model_name_or_path', type=str, default='models/Qwen3-0.6B-Base')
    parser.add_argument('--output_dir', type=str, default='outputs/cpt')
    parser.add_argument('--num_train_epochs', type=int, default=6)
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--quality_threshold', type=int, default=3)
    parser.add_argument('--use_curriculum', type=str2bool, default=True,
                        help='使用课程学习排序')
    parser.add_argument('--fp16', type=str2bool, default=False,
                        help='使用FP16混合精度')
    parser.add_argument('--bf16', type=str2bool, default=False,
                        help='使用BF16混合精度')
    parser.add_argument('--eval_ratio', type=float, default=0.05)
    parser.add_argument('--report_to', type=str, default='tensorboard',
                        help='报告工具 (tensorboard, wandb, none)')

    args = parser.parse_args()

    print("=" * 60)
    print("Continued Pre-Training for Math Domain")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"Data: {args.data_path}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    precision_mode = "FP32" if not args.fp16 and not args.bf16 else ("BF16" if args.bf16 else "FP16")
    print(f"Mixed precision: {precision_mode}")
    print(f"Curriculum learning: {args.use_curriculum}")
    print("=" * 60)

    # 加载tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    print("Loading model...")
    # 使用 float32 加载，让 Trainer 处理混合精度
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Trainer 会自动处理混合精度
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        report_to=args.report_to,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # 创建数据集
    print("\nCreating datasets...")
    full_dataset = MathTextDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        quality_threshold=args.quality_threshold,
        use_curriculum=args.use_curriculum,
    )

    # 分割训练/验证集
    eval_size = int(len(full_dataset) * args.eval_ratio)
    train_size = len(full_dataset) - eval_size

    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    eval_dataset = torch.utils.data.Subset(full_dataset, range(train_size, len(full_dataset)))

    print(f"  Train samples: {train_size}")
    print(f"  Eval samples: {eval_size}")

    # Callbacks
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=3),
        OverfitMonitorCallback(threshold=0.5),
    ]

    # Data Collator for Causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # 训练
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train()

    # 保存最终模型
    print("\nSaving model...")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final model saved to: {args.output_dir}/final_model")


if __name__ == "__main__":
    main()