#!/bin/bash
#
# 启动 TensorBoard 监控训练过程
#
# 使用方法:
#   ./cpt/start_tensorboard.sh [output_dir]
#
# 参数:
#   output_dir: 训练输出目录，默认为 outputs/
#
# 示例:
#   ./cpt/start_tensorboard.sh outputs/cpt_20260506_123456
#

OUTPUT_DIR=${1:-"outputs"}
PORT=${2:-6006}

echo "============================================"
echo "TensorBoard Loss Monitor"
echo "============================================"
echo ""
echo "监控目录: $OUTPUT_DIR"
echo "端口: $PORT"
echo ""
echo "访问地址: http://localhost:$PORT"
echo ""
echo "提示:"
echo "  - 训练开始后，loss曲线会实时更新"
echo "  - 可以监控 train_loss, eval_loss, perplexity"
echo "  - 按 Ctrl+C 停止 TensorBoard"
echo ""
echo "============================================"
echo ""

tensorboard --logdir "$OUTPUT_DIR" --port $PORT --bind_all