# RAG Chatbot Client: Offline Linux Deployment Guide

This document outlines the workflow required to package, export, transfer, and run the RAG Chatbot Client from an **online macOS development environment** to an **offline Linux deployment environment**.

---

## Architecture Overview

The system runs on FastAPI and supports hybrid RAG queries using a PostgreSQL log database, a Redis semantic cache, and a vLLM server. It features a robust **Demo Mode** (`RAG_DEMO_MODE=True` by default) which allows the entire pipeline to function offline with simulated banking policies, local dictionary-based caching, and simulated model completions without needing connection to a real database or LLM container.

```
+--------------------------------------------------------------+
|                     Online macOS Machine                     |
|  - Builds the Docker image for Linux target architecture     |
|  - Exports the layers into a compressed archive (.tar.gz)   |
+--------------------------------------------------------------+
                               |
                        [ Transfer via USB ]
                               v
+--------------------------------------------------------------+
|                     Offline Linux Machine                    |
|  - Loads the compressed image into the local Docker engine   |
|  - Runs the orchestrated services via Docker Compose         |
+--------------------------------------------------------------+
```

---

## The Offline Transfer Workflow

Follow these steps to package and move the chatbot:

### Step 1: Build & Pull Images for Target Architecture (Online macOS)
1. Build the chatbot app image for your target Linux machine architecture (typically `linux/amd64` for standard servers):
   ```bash
   docker build --platform linux/amd64 -t rag-chatbot:v1 .
   ```
2. Pull the dependent service container images on your online machine:
   ```bash
   docker pull postgres:15-alpine
   docker pull redis:7-alpine
   docker pull qdrant/qdrant:latest
   ```

### Step 2: Export All Images to a Compressed Archive (Online macOS)
Bundle all required container images into a single compressed archive file:

```bash
docker save rag-chatbot:v1 postgres:15-alpine redis:7-alpine qdrant/qdrant:latest | gzip > rag-chatbot-full.tar.gz
```

### Step 3: Transfer the Archive to the Offline Linux Machine
1. Copy `rag-chatbot-full.tar.gz` and `docker-compose.yml` to a USB drive or transfer them via local network.
2. Move both files into your designated directory on the offline Linux server (e.g., `/opt/rag-chatbot`).

### Step 4: Load the Images (Offline Linux)
Unpack the images directly into the local Docker engine:

```bash
docker load < rag-chatbot-full.tar.gz
```
Verify all 4 images unpacked successfully:
```bash
docker images
```

### Step 5: Start the Container Stack (Offline Linux)
Run the full container stack using **Docker Compose**:

#### Option A: Running with Docker Compose (Recommended)
1. Ensure `docker-compose.yml` is in the same directory.
2. Spin up PostgreSQL, Redis, Qdrant, and the chatbot server:
   ```bash
   docker compose up -d
   ```
3. Access the web client UI on the local network at:
   ```
   http://<LINUX_HOST_IP>:8010
   ```

#### Option B: Running via Docker CLI
If running standalone in Demo Mode without external containers:
```bash
docker run -d \
  -p 8010:8010 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/demo_logs:/app/demo_logs \
  --name rag_chatbot_client \
  rag-chatbot:v1
```

---

## Dynamic Operations & Host Mappings

The `docker-compose.yml` mapping exposes directories on the host machine to easily update the application configuration without rebuilding the image:

* **Document Drops (`./data`):** Put any custom PDF or DOCX policies into the host's `./data` folder to make them searchable.
* **Persistent Logs (`./demo_logs`):** Open `./demo_logs/chat_history_demo.jsonl` and `./demo_logs/detailed_feedback_demo.jsonl` on the host machine to monitor user queries, thinking times, and satisfaction feedback in real-time.
