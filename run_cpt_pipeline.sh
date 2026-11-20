#!/bin/bash
#
# Complete Training Pipeline with Continued Pre-Training
#
# 流程:
# Phase 0: 数据准备
# Phase 1: 继续预训练 (CPT) - 多轮次 + 课程学习
# Phase 2: 自举生成数据
# Phase 3: 指令微调 (SFT)
#
set -e

# ============ 配置区 ============
BASE_MODEL="models/Qwen3-0.6B-Base"
CCMATH_DATA="nv-community_Nemotron-CC-Math-v1_4plus_first100000.jsonl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# CPT参数
CPT_EPOCHS=6
CPT_LR=5e-5
CPT_BATCH_SIZE=8
CPT_ACCUMULATION=4
CPT_WEIGHT_DECAY=0.1
CPT_MAX_LENGTH=1024

# SFT参数
SFT_EPOCHS_ROUND1=3
SFT_EPOCHS_ROUND2=5
SFT_LR=2e-4
SFT_BATCH_SIZE=4

# 数据生成参数
GEN_SAMPLES_ROUND1=500
GEN_SAMPLES_ROUND2=500

echo "============================================"
echo "MathQwen Complete Training Pipeline"
echo "============================================"
echo "Base Model: $BASE_MODEL"
echo "Timestamp: $TIMESTAMP"
echo ""
echo "CPT Config:"
echo "  Epochs: $CPT_EPOCHS"
echo "  Learning Rate: $CPT_LR"
echo "  Weight Decay: $CPT_WEIGHT_DECAY"
echo ""
echo "Strategy: Multi-epoch CPT + Curriculum Learning + Self-Instruct SFT"
echo "============================================"

# ============ Phase 1: Continued Pre-Training ============
echo ""
echo "[Phase 1] Continued Pre-Training (数学领域适应)"
echo "  策略: 多轮次训练 + 课程学习排序 + 过拟合监控"
echo ""

CPT_DIR="outputs/cpt_${TIMESTAMP}"

python cpt/continued_pretrain.py \
    --data_path "$CCMATH_DATA" \
    --model_name_or_path "$BASE_MODEL" \
    --output_dir "$CPT_DIR" \
    --num_train_epochs $CPT_EPOCHS \
    --per_device_train_batch_size $CPT_BATCH_SIZE \
    --gradient_accumulation_steps $CPT_ACCUMULATION \
    --learning_rate $CPT_LR \
    --weight_decay $CPT_WEIGHT_DECAY \
    --max_length $CPT_MAX_LENGTH \
    --use_curriculum True \
    --quality_threshold 3 \
    --eval_ratio 0.05

CPT_MODEL="${CPT_DIR}/final_model"

if [ ! -d "$CPT_MODEL" ]; then
    echo "Error: CPT model not saved properly"
    exit 1
fi

echo "  CPT completed. Model saved to: $CPT_MODEL"

# ============ Phase 2: Self-Instruct Data Generation ============
echo ""
echo "[Phase 2] Self-Instruct Data Generation"
echo "  用CPT后的模型生成指令数据"
echo ""

INIT_DATA="outputs/self_instruct/init_${TIMESTAMP}.jsonl"

python sft/self_instruct.py \
    --ccmath_path "$CCMATH_DATA" \
    --model_path "$CPT_MODEL" \
    --output_path "$INIT_DATA" \
    --max_samples $GEN_SAMPLES_ROUND1 \
    --prompt_style standard \
    --dtype float16

echo "  Generated data saved to: $INIT_DATA"

# ============ Phase 3: First SFT Round ============
echo ""
echo "[Phase 3] First SFT Round (指令遵循学习)"
echo ""

SFT_DIR_R1="outputs/sft_round1_${TIMESTAMP}"

python sft/train_lora.py \
    --model_name_or_path "$CPT_MODEL" \
    --data_path "$INIT_DATA" \
    --output_dir "$SFT_DIR_R1" \
    --num_train_epochs $SFT_EPOCHS_ROUND1 \
    --per_device_train_batch_size $SFT_BATCH_SIZE \
    --learning_rate $SFT_LR \
    --max_length 1024 \
    --lora_rank 32 \
    --lora_alpha 64

echo "  First SFT completed"

# ============ Phase 4: Merge Weights ============
echo ""
echo "[Phase 4] Merging LoRA weights"
echo ""

MERGED_R1="outputs/merged_round1_${TIMESTAMP}"

python sft/combine_weight.py \
    --base_model "$CPT_MODEL" \
    --lora_path "${SFT_DIR_R1}/peft_adapter" \
    --output_dir "$MERGED_R1" \
    --dtype float16

echo "  Merged model: $MERGED_R1"

# ============ Phase 5: Second Round Data Generation ============
echo ""
echo "[Phase 5] Iterative Data Generation"
echo "  用微调后模型生成更高质量数据"
echo ""

ROUND2_DATA="outputs/iterative/round2_${TIMESTAMP}.jsonl"

python sft/iterative_training.py \
    --model_path "$MERGED_R1" \
    --problem_source "$CCMATH_DATA" \
    --output_path "$ROUND2_DATA" \
    --max_samples $GEN_SAMPLES_ROUND2 \
    --batch_size 8 \
    --temperature 0.7 \
    --dtype float16

echo "  Round 2 data: $ROUND2_DATA"

# ============ Phase 6: Second SFT Round ============
echo ""
echo "[Phase 6] Second SFT Round (推理能力强化)"
echo ""

SFT_DIR_R2="outputs/sft_round2_${TIMESTAMP}"

python sft/train_lora.py \
    --model_name_or_path "$CPT_MODEL" \
    --data_path "$ROUND2_DATA" \
    --output_dir "$SFT_DIR_R2" \
    --num_train_epochs $SFT_EPOCHS_ROUND2 \
    --per_device_train_batch_size $SFT_BATCH_SIZE \
    --learning_rate 1e-4 \
    --max_length 2048 \
    --lora_rank 32 \
    --lora_alpha 64

echo "  Second SFT completed"

# ============ Phase 7: Final Merge ============
echo ""
echo "[Phase 7] Final Model Merge"
echo ""

FINAL_MODEL="outputs/final_model_${TIMESTAMP}"

python sft/combine_weight.py \
    --base_model "$CPT_MODEL" \
    --lora_path "${SFT_DIR_R2}/peft_adapter" \
    --output_dir "$FINAL_MODEL" \
    --dtype float16

echo ""
echo "============================================"
echo "Pipeline Complete!"
echo "============================================"
echo ""
echo "训练路径:"
echo "  Base Model → CPT (${CPT_EPOCHS} epochs) → Self-Instruct → SFT Round1 → Merge → Self-Instruct Round2 → SFT Round2 → Final"
echo ""
echo "输出文件:"
echo "  CPT Model: $CPT_MODEL"
echo "  Final Model: $FINAL_MODEL"
echo ""
echo "启动服务:"
echo "  vllm serve $FINAL_MODEL --host 0.0.0.0 --port 8000"
echo ""
echo "============================================"