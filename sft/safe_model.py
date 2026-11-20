import os
import json
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 假设你的基础模型路径和输出目录
base_model_path = "models/Qwen3-0.6B-Base"
output_dir = "outputs"
checkpoint_dir = "outputs/checkpoint-144"  # 最后一个 checkpoint

# 加载基础模型和 LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(base_model, checkpoint_dir)

# 保存 LoRA adapter 和 tokenizer
save_path = os.path.join(output_dir, "peft_adapter")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# 保存训练参数（可从 checkpoint 中的 trainer_state.json 获取）
print(f"LoRA adapter saved to {save_path}")

# 绘制 loss 曲线（需要从日志中提取数据，或使用 trainer_state.json）
# 这里简单起见，你可以直接运行修正后的脚本，设置 --num_train_epochs 0 让它只执行保存回调。