#!/bin/bash
# FILE: script/vllm_server_start.sh
# DESCRIPTION: Orchestrator script starting both Instruct LLM and Embedding models sequentially

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== vLLM Deployment Start ==="

# Check if running in demo mode
if [ "${RAG_DEMO_MODE}" = "true" ]; then
    echo "RAG_DEMO_MODE is active. Skipping starting local physical vLLM servers."
    exit 0
fi

# Run LLM
bash "$SCRIPT_DIR/vllm_server_llm.sh" &
LLM_PID=$!
echo "Instruct LLM service launched with PID: $LLM_PID"

# Run Embedding
bash "$SCRIPT_DIR/vllm_server_emb.sh" &
EMB_PID=$!
echo "Embedding service launched with PID: $EMB_PID"

# Shutdown clean-up trap
cleanup() {
    echo "Shutting down vLLM servers..."
    kill $LLM_PID $EMB_PID
    wait $LLM_PID $EMB_PID 2>/dev/null
    echo "vLLM servers stopped successfully."
}

trap cleanup SIGINT SIGTERM

wait
