vllm serve outputs/combine_models/lr0.0002_bs4_ep6_20260423_111926 \
      --host 0.0.0.0 \
      --port 8000 \
      --api-key hahaha \
      --gpu-memory-utilization 0.5\
      --served-model-name lora_one