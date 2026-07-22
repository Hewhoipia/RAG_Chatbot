#!/bin/bash
# FILE: script/vllm_server_llm.sh
# DESCRIPTION: Starts the local vLLM server hosting Qwen3.5-4B instructing model on port 8011

MODEL_DIR=${RAG_VLLM_MODEL_DIR:-"models/Qwen3.5-4B"}
MODEL_NAME=${RAG_VLLM_MODEL_NAME:-"Qwen3.5-4B"}
PORT=${RAG_VLLM_PORT:-8011}
GPU_MEM=${RAG_VLLM_GPU_MEMORY:-0.55}
MAX_SEQS=${RAG_VLLM_MAX_NUM_SEQS:-25}

echo "Starting vLLM server with model: $MODEL_NAME on port $PORT..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "$MODEL_NAME" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-num-seqs "$MAX_SEQS" \
    --trust-remote-code
