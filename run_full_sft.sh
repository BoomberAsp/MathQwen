#!/bin/bash
#
# Full-parameter SFT Pipeline (全量微调，非LoRA)
#
# 流程:
# 1. 使用CPT后的模型或Base模型
# 2. 全量微调（所有参数都参与训练）
# 3. 直接保存完整模型（无需合并）
#
set -e

# ============ 配置区 ============
# 模型路径：可以选择Base模型或CPT后的模型
BASE_MODEL="models/Qwen3-0.6B-Base"
# CPT_MODEL="outputs/cpt_xxx/final_model"  # 如果已完成CPT，可使用此模型

# 选择模型（默认用Base，如果有CPT模型可修改）
MODEL_PATH="$BASE_MODEL"

# 数据路径
DATA_PATH="data/train_sft.jsonl"  # 可修改为其他数据

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_TO="tensorboard"

# 训练参数
EPOCHS=3
BATCH_SIZE=4
ACCUMULATION=4
LR=1e-5  # 全量微调推荐1e-5~5e-5
MAX_LENGTH=1024
WEIGHT_DECAY=0.01

echo "============================================"
echo "Full-parameter SFT Pipeline"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Timestamp: $TIMESTAMP"
echo ""
echo "训练参数:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE × $ACCUMULATION = $((BATCH_SIZE * ACCUMULATION))"
echo "  Learning rate: $LR"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Precision: FP32"
echo ""
echo "TensorBoard监控:"
echo "  ./cpt/start_tensorboard.sh outputs/full_sft_${TIMESTAMP}"
echo "============================================"

# 运行全量微调
python sft/full_sft.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "outputs/full_sft_${TIMESTAMP}" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $ACCUMULATION \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --max_length $MAX_LENGTH \
    --eval_ratio 0.1 \
    --early_stopping_patience 2 \
    --report_to "$REPORT_TO" \
    --fp16 false \
    --bf16 false

echo ""
echo "============================================"
echo "SFT Complete!"
echo "============================================"
echo "Model saved to: outputs/full_sft_${TIMESTAMP}"
echo ""
echo "启动服务:"
echo "  vllm serve outputs/full_sft_${TIMESTAMP} --host 0.0.0.0 --port 8000"
echo ""
echo "测试:"
echo "  curl http://localhost:8000/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"outputs/full_sft_${TIMESTAMP}\", \"messages\": [{\"role\": \"user\", \"content\": \"求解: 2x + 3 = 7\"}]}'"
echo "============================================"