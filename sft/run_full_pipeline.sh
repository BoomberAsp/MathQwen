#!/bin/bash
#
# Complete Self-Bootstrapping Training Pipeline
# Uses only Base model + CCMath data (no external resources)
#
# Pipeline:
# 1. Generate initial instruction data (self_instruct.py)
# 2. First SFT round (train_lora.py)
# 3. Merge weights (combine_weight.py)
# 4. Generate better data with fine-tuned model (iterative_training.py)
# 5. Second SFT round (train_lora.py)
#

set -e

# Configuration
BASE_MODEL="models/Qwen3-0.6B-Base"
CCMATH_DATA="nv-community_Nemotron-CC-Math-v1_4plus_first100000.jsonl"
MAX_SAMPLES_ROUND1=500
MAX_SAMPLES_ROUND2=500
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo "Self-Bootstrapping Training Pipeline"
echo "============================================"
echo "Base Model: $BASE_MODEL"
echo "Data Source: $CCMATH_DATA"
echo "Timestamp: $TIMESTAMP"
echo "============================================"

# Step 1: Generate initial instruction data
echo ""
echo "[Step 1] Generating initial instruction data..."
echo "  This teaches the model the instruction-following format"

INIT_DATA="outputs/self_instruct/round1_${TIMESTAMP}.jsonl"

python sft/self_instruct.py \
    --ccmath_path "$CCMATH_DATA" \
    --model_path "$BASE_MODEL" \
    --output_path "$INIT_DATA" \
    --max_samples $MAX_SAMPLES_ROUND1 \
    --prompt_style standard \
    --dtype float16

echo "  Generated data saved to: $INIT_DATA"

# Check if data was generated
if [ ! -s "$INIT_DATA" ]; then
    echo "Error: No data generated. Aborting."
    exit 1
fi

# Step 2: First SFT round
echo ""
echo "[Step 2] First SFT round - Learning instruction format..."
echo "  Training LoRA adapter on generated data"

FIRST_SFT_DIR="outputs/round1_sft_${TIMESTAMP}"

python sft/train_lora.py \
    --model_name_or_path "$BASE_MODEL" \
    --data_path "$INIT_DATA" \
    --output_dir "$FIRST_SFT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4 \
    --max_length 1024 \
    --lora_rank 32 \
    --lora_alpha 64

echo "  First SFT completed. Output: $FIRST_SFT_DIR"

# Step 3: Merge weights
echo ""
echo "[Step 3] Merging LoRA weights into base model..."

MERGED_MODEL="outputs/round1_merged_${TIMESTAMP}"

python sft/combine_weight.py \
    --base_model "$BASE_MODEL" \
    --lora_path "$FIRST_SFT_DIR/peft_adapter" \
    --output_dir "$MERGED_MODEL" \
    --dtype float16

echo "  Merged model saved to: $MERGED_MODEL"

# Step 4: Generate better data with fine-tuned model
echo ""
echo "[Step 4] Generating better quality data with fine-tuned model..."
echo "  The instruction-following model can now produce better responses"

ROUND2_DATA="outputs/iterative/round2_${TIMESTAMP}.jsonl"

python sft/iterative_training.py \
    --model_path "$MERGED_MODEL" \
    --problem_source "$CCMATH_DATA" \
    --output_path "$ROUND2_DATA" \
    --max_samples $MAX_SAMPLES_ROUND2 \
    --batch_size 8 \
    --temperature 0.7 \
    --dtype float16

echo "  Round 2 data saved to: $ROUND2_DATA"

# Step 5: Second SFT round
echo ""
echo "[Step 5] Second SFT round - Math reasoning specialization..."
echo "  Training on higher quality math reasoning data"

FINAL_SFT_DIR="outputs/final_sft_${TIMESTAMP}"

python sft/train_lora.py \
    --model_name_or_path "$BASE_MODEL" \
    --data_path "$ROUND2_DATA" \
    --output_dir "$FINAL_SFT_DIR" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --max_length 2048 \
    --lora_rank 32 \
    --lora_alpha 64

echo "  Final SFT completed. Output: $FINAL_SFT_DIR"

# Step 6: Final merge
echo ""
echo "[Step 6] Final weight merge..."

FINAL_MODEL="outputs/final_model_${TIMESTAMP}"

python sft/combine_weight.py \
    --base_model "$BASE_MODEL" \
    --lora_path "$FINAL_SFT_DIR/peft_adapter" \
    --output_dir "$FINAL_MODEL" \
    --dtype float16

echo ""
echo "============================================"
echo "Pipeline Complete!"
echo "============================================"
echo "Final model: $FINAL_MODEL"
echo ""
echo "To serve with vLLM:"
echo "  vllm serve $FINAL_MODEL --host 0.0.0.0 --port 8000"
echo ""
echo "To test:"
echo "  curl http://localhost:8000/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"$FINAL_MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"求解: 2x + 3 = 7\"}]}'"
echo "============================================"