# Access with: http://localhost:8010
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backed:cudaMallocAsync"

# ENVIRONMENT CONFIG
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0") # enable ASYNC mode cho GPU
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments: True, max_split_size_mb:512") # PyTorch mem pool tối ưu
# tắt HuggingFace Internet calls
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

#Tokenizer optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("TOKENIZERS_BATCH_CAPACITY", "512")

# thread optimization
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Redis Cache - ENABLED by default
os.environ.setdefault("RAG_REDIS_ENABLED", "true")
os.environ.setdefault("RAG_REDIS_HOST", "localhost")
os.environ.setdefault("RAG_REDIS_PORT", "6379")

# IMPORTS
import asyncio
import gc
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
import torch
from fastapi import FastAPI, Request, HTTPException, Query, BackgroundTasks, status
from fastapi.responses import Response, FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

try:
    import orjson
    from fastapi.responses import ORJSONResponse
    DEFAULT_RESPONSE_CLASS = ORJSONResponse
except ImportError:
    from fastapi.responses import JSONResponse
    DEFAULT_RESPONSE_CLASS = JSONResponse
    print ("orjson not installed, using standard JSONResponse")

# DOCX handling
from docx import Document
from docx.shared import RGBColor
from docx.enum.text import WD_COLOR_INDEX

# Local modules - vllm CLIENT
# from config import config
# from connect_db_vllm import DatabaseManager
# from text_generate_prod_vllm_client import LLMDeploymentV7VLLMClient, LLMGenerationError, ChatEngineV7VLLMClient
# from rag_prod_vllm import RetrievalEngineV7, CircuitOpenError

# mini_function
# from mini_func.query_theme.query_theme import query_theme
# from mini_func.follow_up.follow_up_service import generate_smart_follow_ups
from mini_func.auto_suggest.auto_suggest import get_autocomplete_suggestions
# NOTE: In client mode, vLLM runs as separate server on port 8011

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,
    handlers=[
        logging.StreamHandler() # ensure output to console
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# GLOBAL VARIABLES
# Directories
# document_dir = os.path.abspath(config.retrieval.data_dir or '.')
# print (f"Document directory: {document_dir}")

# Users mapping (IP -> username)


# cleanup rate limitting
cleanup_lock = Lock()
last_cleanup_time = 0

# PYDANTIC MODELS
class ChatRequest(BaseModel):
    """Request model for /chat endpoint"""
    prompt: str = Field(..., min_length=1, max_length=10000, description="User question")
    class Config:
        json_schema_extra = {"example": {"prompt": "What are the deposit policies?"}}

# model cho feedback chi tiết
class DetailedFeedbackRequest(BaseModel):
    """Request model for /detailed_feedback endpoint"""
    conversation_id: str = Field(..., description="UUID of conversation")
    answer_satisfaction: Optional[int] = Field(None, ge=0, le=1, description="1 = Agree, 0 = Disagree, None = Closed tab (no feedback)")
    quality_rating: Optional[int] = Field(None, ge=1, le=5, description="chat luong tra loi (1-5 sao)")
    speed_rating: Optional[int] = Field(None, ge=1, le=5, description="toc do tra loi (1-5 sao)")
    issues: List[str] = Field(default_factory=list, description="Danh sach ma loi")
    other_opinions: Optional[str] = Field(None, max_length=5000, description="y kien bo sung")

class ChatResponse(BaseModel):
    """Response model for /chat endpoint"""
    conversation_id: str
    response: str
    sources: Optional[List[Dict[str, Any]]] = None
    thought_time: float
    cache_hit: bool = False
    query_type: Optional[str] = None # simple, complex, chit-chat, clarify
    
class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str
    version: str = "7"
    components: Dict[str, Any] = {}

# RATE LIMITER
class TokenBucketRateLimiter:
    """Per-IP limiter using token bucket algorithm"""

class AdmissionController:
    """Global concurrency control - limits TOTAL concurrent requests to protect GPU"""

class DegradationLevel(Enum):
    NORMAL = "normal" # full features
    REDUCED = "reduced" # Disable cache, reduce context
    MINIMAL = "minimal" # emergency mode, refuse new requests
    
class DegradationController:
    """Graceful degradation based on system load"""

# CUSTOM EXCEPTIONS
class RAGException(Exception):
    """Base exception with user-safe message"""
    def __init__(self, message: str, status_code: int = 500, internal_message: str = None):
        self.message = message
        self.status_code = status_code
        self.internal_message = internal_message or message
        super().__init__(self.internal_message)

class RateLimitExceeded(RAGException):
    def __init__(self, ip:str):
        super().__init__(
            message="Qua nhieu requests. Vui long thu lai sau.",
            status_code=429,
            internal_message=f"Rate limit exceeded for IP: {ip}"
        )

class ServiceUnavailable(RAGException):
    def __init__(self, service: str = "LLM"):
        super().__init__(
            message="He thong dang ban. Vui long thu lai sau.",
            status_code=503,
            internal_message=f"Service unavailable: {service}"
        )
        
# HELPER FUNCS
def cleanup_all():
    """Thu dọn bộ nhớ rác và giải phóng rò rỉ RAM cũng như VRAM"""

async def cleanup_async():
    await asyncio.to_thread(cleanup_all)

# LIFESPAN MANAGEMENT
@asynccontextmanager
async def lifespan (app: FastAPI):
    """
    Quản lý vòng đời hoạt động của app FastAPI (start-up / shut-down)
    Mục đích: khởi tạo các tài nguyên chung (database, LLM Client, Qdrant db, redis) một lần khi bật server và xả bộ nhớ khi tắt
    Cách thức hđ:
    - Setup DatabaseManager và tạo connection pool
    - Khởi tạo RateLimiter, RetrievalEngine(RAG)
    - Nếu bật Query Analysis, khởi tạo QueryClassifier và Router
    - Chạy ứng dụng web (yield)
    - thực hiện close DB, close retrieval, đẩy rác bằng cleanup_all khi shutdown
    
    Đầu vào và đầu ra:
    - Đầu vào:
        + 'app' (FastAPI): instance của app
    - Đầu ra:
        + (AsyncGenerator): tạo yield để hệ thống máy chủ giữ loop hoạt động
        
    Lưu ý: nếu 1 module như vLLM không truy cập được thì hệ thống log cảnh báo chứ không sập
    """
    

app = FastAPI(
    title="RAG Chatbot API v1 - no model",
    description="Production RAG v7 - 1 concurrent user, FP16 reranker, Redis cache",
    version="1.0",
    lifespan=None,
    default_response_class=DEFAULT_RESPONSE_CLASS,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    pass

# Template
try:
    templates = Jinja2Templates(directory="templates")
except Exception as e:
    pass

# --- MAIN PAGE ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Endpoint này sẽ được gọi khi người dùng truy cập http://localhost:8010/
    
    Nó sẽ tìm và trả về file index.html nằm trong thư mục 'templates'
    """
    return templates.TemplateResponse("index.html", {"request": request})

# API ENDPOINT FOR AUTO SUGGESTION
@app.get("/api/suggest")
async def autocomplete_endpoint(
    q: Optional[str] = Query(None, description="Ký tự người dùng đang gõ")
):
    if not q or not q.strip():
        return {"suggestions": []}
        
    try:
        suggestions = await get_autocomplete_suggestions(q)
        return {"suggestions": suggestions}
    except Exception as e:
        # Quan trọng: Print lỗi ra console để debug thay vì chỉ văng 500
        print(f"Error when suggest: {e}") 
        return {"suggestions": []}

# MAIN
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "mybot_prod_vllm_client:app",
        host="localhost",
        port=8010,
        reload=True, # True in dev enviroment
        workers=1,
        log_level="info",
    )