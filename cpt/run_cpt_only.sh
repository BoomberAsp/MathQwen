#!/bin/bash
#
# 单独运行继续预训练 (CPT)
#
set -e

BASE_MODEL="models/Qwen3-0.6B-Base"
CCMATH_DATA="nv-community_Nemotron-CC-Math-v1_4plus_first100000.jsonl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_TO="tensorboard"  # 可改为 wandb 或 none

echo "============================================"
echo "Continued Pre-Training"
echo "============================================"
echo ""
echo "关键策略:"
echo "  ✓ 多轮次训练 (6 epochs)"
echo "  ✓ 课程学习排序"
echo "  ✓ 过拟合监控 + Early Stopping"
echo "  ✓ 高正则化 (Weight Decay 0.1)"
echo "  ✓ 每epoch重新shuffle"
echo "  ✓ TensorBoard 实时监控"
echo "  ✓ FP32 全精度训练（适合小模型）"
echo ""
echo "启动 TensorBoard 监控:"
echo "  ./cpt/start_tensorboard.sh outputs/cpt_${TIMESTAMP}"
echo ""

python cpt/continued_pretrain.py \
    --data_path "$CCMATH_DATA" \
    --model_name_or_path "$BASE_MODEL" \
    --output_dir "outputs/cpt_${TIMESTAMP}" \
    --num_train_epochs 6 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --max_length 1024 \
    --use_curriculum true \
    --quality_threshold 3 \
    --eval_ratio 0.05 \
    --report_to "$REPORT_TO" \
    --fp16 false \
    --bf16 false

echo ""
echo "CPT完成！模型保存在: outputs/cpt_${TIMESTAMP}/final_model"
echo ""
echo "下一步:"
echo "  运行自举生成: python sft/self_instruct.py --model_path outputs/cpt_${TIMESTAMP}/final_model"