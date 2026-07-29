#!/bin/bash
# FILE: script/vllm_server_emb.sh
# DESCRIPTION: Starts the local vLLM server hosting BAAI/bge-m3 embedding model on port 8012

EMB_MODEL=${RAG_CACHE_EMBEDDING_MODEL:-${RAG_EMB_MODEL:-"BAAI/bge-m3"}}
EMB_NAME=${RAG_VLLM_MODEL_NAME_EMBEDDING:-${RAG_EMB_MODEL_NAME:-"bge-m3"}}
PORT=${RAG_EMB_PORT:-8012}
GPU_MEM=${RAG_VLLM_GPU_MEMORY_UTILIZATION_EMB:-${RAG_EMB_GPU_MEMORY:-0.20}}

echo "Starting vLLM embedding server with model: $EMB_NAME on port $PORT..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "$EMB_MODEL" \
    --served-model-name "$EMB_NAME" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" \
    --trust-remote-code
