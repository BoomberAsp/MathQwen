"""
Dataset utilities for reasoning SFT training.
修复版：使用 token ID 边界定位 assistant 回复，无需字符偏移。
"""
import json
import torch
from typing import Dict, List, Optional
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset


class ReasoningSFTDataset(TorchDataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        split: Optional[str] = None,
        split_ratio: float = 0.95,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        dataset = self._load_jsonl_dataset(data_path)

        if split is not None:
            dataset = dataset.train_test_split(
                test_size=1 - split_ratio,
                seed=42,
                shuffle=True
            )
            dataset = dataset['train'] if split == 'train' else dataset['test']

        self.dataset = dataset

    def _load_jsonl_dataset(self, data_path: str) -> Dataset:
        messages_list = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'messages' in data:
                        messages_list.append({'messages': data['messages']})
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line[:100]}...")
        return Dataset.from_list(messages_list)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        messages = self.dataset[idx]['messages']

        # 1. 使用 chat template 生成 token 序列
        encoding = self.tokenizer.apply_chat_template(
            messages,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 2. 初始化 labels 全为 -100
        labels = torch.full_like(input_ids, -100)

        # 3. 找到 assistant 消息的 token 边界并设置 labels
        # 对于 Qwen3，模板为：<|im_start|>assistant\n{content}<|im_end|>\n
        # 我们需要 token ID 来定位
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_id = self.tokenizer.convert_tokens_to_ids("assistant")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        # 如果 tokenizer 没有这些特殊 token（理论上不应该），则回退到简单全量 loss
        if None in (im_start_id, assistant_id, im_end_id):
            # 回退：使用所有有效 token 作为 labels（不推荐，但至少能跑）
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }

        seq_len = input_ids.size(0)
        i = 0
        while i < seq_len:
            # 寻找序列 "<|im_start|>assistant"
            if input_ids[i] == im_start_id and i + 1 < seq_len and input_ids[i + 1] == assistant_id:
                # 从当前位置开始寻找下一个 <|im_end|>
                j = i + 2  # 跳过 assistant token
                while j < seq_len and input_ids[j] != im_end_id:
                    j += 1
                # 将 i+2 到 j-1 之间的 token（即 assistant 内容）设为可学习
                # 注意：这里包含了换行符等，但都属于 assistant 消息的一部分
                labels[i+2:j] = input_ids[i+2:j]
                i = j + 1  # 继续向后搜索
            else:
                i += 1

        # 4. 屏蔽 padding
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def collate_fn(batch: List[Dict], pad_token_id: int) -> Dict:
    """由于我们在 dataset 中已经做了 padding，这里可以直接堆叠。"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


def load_training_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 2048,
    split_ratio: float = 0.95,
) -> tuple:
    train_dataset = ReasoningSFTDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        split='train',
        split_ratio=split_ratio,
    )
    eval_dataset = ReasoningSFTDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        split='eval',
        split_ratio=split_ratio,
    )
    return train_dataset, eval_dataset