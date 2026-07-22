#!/bin/bash
# FILE: start_local.sh
# DESCRIPTION: Orchestrates the local startup sequence: Database, Cache, vLLM, and finally FastAPI

echo "=========================================================="
echo "    RAG CHATBOT LOCAL DEPLOYMENT STARTUP INSTRUCTIONS     "
echo "=========================================================="

# 1. Load configurations
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
fi

# Set default demo mode to true if not explicitly set
export RAG_DEMO_MODE=${RAG_DEMO_MODE:-"true"}

echo "Environment Mode: DEMO_MODE=$RAG_DEMO_MODE"
echo ""

# 2. Start PostgreSQL Database
if [ "$RAG_DEMO_MODE" = "false" ]; then
    echo "[Step 1/3] Starting Database services..."
    # Attempting to start local PostgreSQL service (macOS / Homebrew example)
    if command -v brew &> /dev/null; then
        brew services start postgresql
        sleep 2
    else
        echo "WARNING: PostgreSQL starting command varies by OS."
        echo "Ensure PostgreSQL is running locally on port 5432."
    fi
else
    echo "[Step 1/3] Running in DEMO MODE. Skipping local physical Database startup."
fi

# 3. Start local vLLM servers
if [ "$RAG_DEMO_MODE" = "false" ]; then
    echo "[Step 2/3] Starting local vLLM LLM and Embedding servers..."
    bash script/vllm_server_start.sh &
    VLLM_PID=$!
    # Wait for vLLM servers to boot up (healthcheck check)
    echo "Waiting for vLLM server to start on port 8011..."
    until curl -s http://localhost:8011/v1/models &> /dev/null; do
        sleep 5
        echo -n "."
    done
    echo ""
    echo "vLLM servers are UP and running."
else
    echo "[Step 2/3] Running in DEMO MODE. Skipping local vLLM server startup."
fi

# 4. Start FastAPI server
echo "[Step 3/3] Starting FastAPI client chatbot server..."
python3 mybot_prod_vllm_client.py

# Clean up processes on termination
cleanup() {
    if [ ! -z "$VLLM_PID" ]; then
        kill $VLLM_PID
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM
