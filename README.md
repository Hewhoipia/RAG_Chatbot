# RAG Chatbot Client: Offline Linux Deployment Guide

This document outlines the workflow required to package, export, transfer, and run the RAG Chatbot Client from an **online macOS development environment** to an **offline Linux deployment environment**.

---

## Architecture Overview

The system runs on FastAPI and supports hybrid RAG queries using a PostgreSQL log database, a Redis semantic cache, and a vLLM server. It features a robust **Demo Mode** (`RAG_DEMO_MODE=true` by default) which allows the entire pipeline to function offline with simulated banking policies, local dictionary-based caching, and simulated model completions without needing connection to a real database or LLM container.

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

### Step 1: Build for the Target Architecture (Online macOS)
Run the build command on macOS. Specify the CPU architecture of your target Linux machine (typically `linux/amd64` for standard servers, or `linux/arm64` for ARM devices) to prevent architecture mismatch crashes:

```bash
docker build --platform linux/amd64 -t rag-chatbot:v1 .
```

### Step 2: Export the Image to a Compressed File (Online macOS)
Use `docker save` to bundle the layers into an archive, then compress it with `gzip` to reduce file transfer time:

```bash
docker save rag-chatbot:v1 | gzip > rag-chatbot.tar.gz
```

### Step 3: Transfer the Archive to the Linux Device
1. Copy `rag-chatbot.tar.gz` and the `docker-compose.yml` file to a USB drive or transfer them over a local network.
2. Plug the media into the target Linux machine and copy both files into your designated working directory (e.g., `/opt/rag-chatbot`).

### Step 4: Load the Image (Offline Linux)
On the offline Linux server, run `docker load` to unpack the image layers into the local Docker engine:

```bash
docker load < rag-chatbot.tar.gz
```
Verify the image successfully unpacked by listing your local images:
```bash
docker images
```

### Step 5: Start the Container Stack (Offline Linux)
We recommend running the container stack using **Docker Compose** to handle dependency initialization (database, cache, app) automatically.

#### Option A: Running with Docker Compose (Recommended)
1. Ensure the `docker-compose.yml` file is in the same directory.
2. Spin up PostgreSQL, Redis, and the chatbot server:
   ```bash
   docker compose up -d
   ```
3. Open your browser on the local network at:
   ```
   http://<LINUX_HOST_IP>:8010
   ```

#### Option B: Running via Docker CLI
If you prefer not to use Docker Compose, start the application container standalone in Demo Mode:
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
