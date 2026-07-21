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
os.environ.setdefault("RAG_REDIS_PORT", "6379") # default port

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
from config import config
from connect_db_vllm import DatabaseManager
from text_generate_prod_vllm_client import LLMDeploymentV7VLLMClient, LLMGenerationError, ChatEngineV7VLLMClient
from rag_prod_vllm import RetrievalEngineV7, CircuitOpenError

# mini_function
from mini_func.query_theme.query_theme import query_theme
from mini_func.follow_up.follow_up_service import generate_smart_follow_ups
from mini_func.auto_suggest.auto_suggest import get_autocomplete_suggestions
from mini_func.query_transform.query_transform import create_async_client
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
document_dir = os.path.abspath(config.retrieval.data_dir or '.')
print (f"Document directory: {document_dir}")

# Users mapping (IP -> username)
users_lst = {}
try:
    with open("users.json", "r", encoding="utf-8") as f:
        users_lst = json.load(f)
    print(f"Loaded {len(users_lst)} users from users.json")
except FileNotFoundError:
    print("users.json not found, continuing with empty list")
except json.JSONDecodeError as e:
    print(f"Error parsing users.json: {e}")

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
    """Thu dọn bộ nhớ rác và giải phóng rò rỉ RAM cũng như VRAM."""
    logger.info("Running cleanup: Clearing CUDA cache and collecting garbage.")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
        gc.collect()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

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
    app.state.components = {}
    logger.info("--- Starting up Application ---")
    
    # 1. Init Database
    try:
        db_manager = DatabaseManager()
        await db_manager.init_pool()
        app.state.components["db"] = db_manager
        logger.info("Database pool initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Database: {e}")

    # 2. Init LLM Client (vLLM)
    try:
        llm_client = LLMDeploymentV7VLLMClient(
            base_url=config.vllm.base_url,
            model_name=config.vllm.model_name
        )
        app.state.components["llm_client"] = llm_client
        
        # 3. Init Chat Engine
        chat_engine = ChatEngineV7VLLMClient(llm_client=llm_client)
        app.state.components["chat_engine"] = chat_engine
        logger.info("LLM Client and Chat Engine initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize LLM/Chat Engine: {e}")

    # 4. Init Retrieval Engine (RAG)
    try:
        retrieval_engine = RetrievalEngineV7()
        app.state.components["retrieval"] = retrieval_engine
        logger.info("Retrieval Engine initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Retrieval Engine: {e}")

    # 5. Init Query Transform Client
    app.state.components["transform_client"] = create_async_client()
    logger.info("Query Transform client initialized.")

    yield

    logger.info("--- Shutting down Application ---")
    if "db" in app.state.components:
        await app.state.components["db"].close()
        logger.info("Database pool closed.")

app = FastAPI(
    title="RAG Chatbot API v1 - no model",
    description="Production RAG v1 - 1 concurrent user, FP16 reranker, Redis cache",
    version="1.0",
    lifespan=lifespan,
    default_response_class=DEFAULT_RESPONSE_CLASS,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip pompression for responses > 1KB
app.add_middleware(GZipMiddleware, minimum_size = 1000)

# static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Static files directory not found: {e}")

# Template
try:
    templates = Jinja2Templates(directory="templates")
except Exception as e:
    templates = None
    logger.warning(f"Templates directory not found: {e}")
    
# EXCEPTION HANDLERS
@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    """
    Giới thiệu: bắt và xử lý các exception chung thuộc RAGException
    Mục đích: đảm bảo thông báo lỗi của hệ thống được trả về client bằng định dạng chuẩn xác, an toàn
    Cách thức hoạt động:
    - Ghi log (error) nội dung lỗi cho backend (internal_message)
    - Phản hồi tới client JSON gồm status_code và safe message
    Đầu vào và đầu ra:
    - Đầu vào:
        + `request` (Request): Yêu cầu truy cập HTTP
        + `exc` (RAGException): đối tượng lỗi
    - Đầu ra:
        + (DEFAULT_RESPONSE_CLASS): phản hồi HTTP (Response) chuẩn format JSON/ORJSON
    """
    logger.error(f"RAGException: {exc.internal_message}")
    return DEFAULT_RESPONSE_CLASS(
        status_code=exc.status_code,
        content={"error": exc.message}
    )
    
@app.exception_handler(LLMGenerationError)
async def llm_error_handler(reques: Request, exc: LLMGenerationError):
    """
    Handles exceptions related to LLM generation failures.

    Args:
        request (Request): The incoming request.
        exc (LLMGenerationError): The exception raised by the LLM client.

    Returns:
        A JSON response with a 503 Service Unavailable status, indicating
        that the language model is currently unable to process requests.
    """
    logger.error(f"LLMGenerationError: {exc}")
    return DEFAULT_RESPONSE_CLASS(
        status_code=503,
        content={"error": "Loi xu ly. Vui long thu lai sau."}
    )
    
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    A global exception handler to catch any unhandled exceptions.

    Args:
        request (Request): The incoming request.
        exc (Exception): The unhandled exception.

    Returns:
        A JSON response with a 500 Internal Server Error status,
        preventing the server from crashing and providing a generic error message.
    """
    logger.exception(f"Unhandled exception: {exc}")
    return DEFAULT_RESPONSE_CLASS(
        status_code=500,
        content={"error": "Loi he thong. Vui long thu lai sau."}
    )
    
# ============================================================================
# ENDPOINTS
# ============================================================================
    
# --- MAIN PAGE ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Endpoint này sẽ được gọi khi người dùng truy cập http://localhost:8010/
    
    Nó sẽ tìm và trả về file index.html nằm trong thư mục 'templates'
    """
    if templates is None:
        return HTMLResponse("<h1>RAG Chatbot v1 Client</h1><p>Templates not configured</p>")
    user_ip = request.client.host if request.client else "unknown"
    if "x-forwarded-for" in request.headers:
        user_ip = request.headers["x-forwarded-for"].split(",")[0].strip()
    user_name = users_lst.get(str(user_ip), "")
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "user_name": user_name}
    )
    
@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """
    Provides a health check of the application and its components.

    Checks the status of the database connection and the retrieval engine.

    Args:
        request (Request): The incoming request.

    Returns:
        A HealthResponse object detailing the status of each component.
    """
    components = {}
    overall_healthy = True
    # check database
    try:
        db = request.app.state.components.get("db")
        if db:
            db_health = await db.health_check()
            components["database"] = db_health
            if not db_health.get("healthy"):
                overall_healthy = False
    except Exception as e:
        components["database"] = {"healthy": False, "error": str(e)}
        overall_healthy = False
    
    #check retrieval engine
    try:
        retrieval = request.app.state.components.get("retrieval")
        if retrieval:
            components["retrieval"] = retrieval.get_stats()
    except Exception as e:
        components["retrieval"] = {"error": str(e)}
    return HealthResponse(
        status="healthy" if overall_healthy else "degraded",
        components=components,
    )
    
# Endpoint cho feedback chi tiết
@app.post("/detailed_feedback")
async def detailed_feedback(request: Request, data: DetailedFeedbackRequest):
    """
    Receives and stores detailed user feedback for a conversation.

    Args:
        request (Request): The incoming request.
        data (DetailedFeedbackRequest): The detailed feedback data from the user.

    Returns:
        A confirmation message if the feedback is saved successfully.
    Raises:
        HTTPException: If the database is not available.
    """
    db = request.app.state.components.get("db")
    if db:
        issues_json = json.dumps(data.issues) if data.issues else None
        await db.save_detailed_feedback(
            conversation_id = data.conversation_id,
            answer_satisfaction = data.answer_satisfaction,
            quality_rating=data.quality_rating,
            speed_rating = data.speed_rating,
            issues = issues_json,
            other_opinions = data.other_opinions,
        )
        return {"message": "Detailed feedback received"}
    raise HTTPException(503, "Database not available")

# STREAMING WITH THINKING ENDPOINT
@app.post("/chat/stream/thinking")
async def chat_stream_thinking(request: Request, chat_request: ChatRequest):
    """
    Handles streaming chat requests with a "thinking" phase.

    This endpoint classifies the user's query. For simple chit-chat, it responds
    directly. For RAG queries, it first streams the LLM's reasoning process
    ("thinking") and then streams the final answer. It logs the entire interaction
    to the database.

    Args:
        request (Request): The incoming request, used to get client IP.
        chat_request (ChatRequest): The user's prompt.

    Returns:
        A StreamingResponse using Server-Sent Events (SSE) to stream JSON objects.
    """
    user_input = chat_request.prompt
    conversation_id = str(uuid.uuid4())
    t0 = datetime.now()
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    if "x-forwarded-for" in request.headers:
        client_ip = request.headers["x-forwarded-for"].split(",")[0].strip()
    # Rate limitting
    if config.server.rate_limit_enabled:
        rate_limiter = request.app.state.components.get("rate_limiter")
        if rate_limiter and not await rate_limiter.is_allowed(client_ip):
            raise RateLimitExceeded(client_ip)
    
    # Check degradation
    degradation = request.app.state.components.get("degradation")
    if degradation and degradation.should_reject():
        raise ServiceUnavailable("System overloaded")
    
    # get components
    retrieval = request.app.state.components.get("retrieval")
    chat_engine = request.app.state.components.get("chat_engine")
    classifier = request.app.state.components.get("classifier")
    transform_client = request.app.state.components.get("transform_client")
    db = request.app.state.components.get("db")
    if not chat_engine:
        raise HTTPException(503, "Engine not initialized")
    # check if thinking mode is enabled
    enable_thinking = config.qwen3.enable_thinking
    
    async def generate_stream_with_thinking():
        """
        Giới thiệu: trình đưa luồng Event chuỗi tư duy Streaming (Reasoning) cho UI
        Mục đích: phân mảng 2 giai đoạn (tư duy => trả lời) vào chuẩn chung EventSource nối JSON, hỗ trợ cơ chế nắn dòng văn bản Thinking riêng biệt
        Cách thức hoạt động:
        - Đầu vào: k tham số (lấy từ parent scope)
        - Đầu ra: Các Frame dang Yield (str) form JSON
        Có đếm tg Thinking tính bằng seconds cho vào DB
        """
        full_response = ""
        thinking_content = ""
        sources = []
        cache_hit = False
        query_type = "unknown"
        thinking_time = 0.0
        
        try:
            # 1. Classify query
            query_type = await query_theme(user_input)
            
            # 2. Generate response based on type
            if query_type in ["chit-chat", "clarify"]:
                # Direct response, no RAG
                async for event in chat_engine.generate_direct_response_stream(user_input, query_type):
                    if event["type"] == "answer":
                        full_response += event["content"]
                    yield f"data: {json.dumps(event)}\n\n"
            else: # RAG for "simple", "complex", "unknown"
                # 3. Retrieve context
                retrieved_nodes = await retrieval.retrieve(user_input)
                sources = chat_engine._process_sources(retrieved_nodes)
                
                # 4. Stream response from LLM
                thinking_start_time = time.time()
                async for event in chat_engine.generate_response_stream(user_input, retrieved_nodes, enable_thinking=enable_thinking):
                    if event["type"] == "thinking":
                        thinking_content += event["content"]
                    elif event["type"] == "answer":
                        if thinking_time == 0.0 and thinking_content:
                            thinking_time = time.time() - thinking_start_time
                        full_response += event["content"]
                    yield f"data: {json.dumps(event)}\n\n"
                
                # 5. Send sources
                sources_event = {"type": "sources", "content": sources}
                yield f"data: {json.dumps(sources_event)}\n\n"

        except Exception as e:
            logger.error(f"Error during stream generation for {conversation_id}: {e}", exc_info=True)
            error_event = {"type": "error", "content": "An error occurred during generation."}
            yield f"data: {json.dumps(error_event)}\n\n"
        finally:
            # 6. Log to DB
            if db:
                try:
                    await db.save_chat_history(
                        conversation_id=conversation_id,
                        user_ip=client_ip,
                        user_name=users_lst.get(client_ip, ""),
                        user_question=user_input,
                        llm_answer=full_response,
                        thinking_content=thinking_content if enable_thinking else None,
                        sources=json.dumps(sources) if sources else None,
                        query_type=query_type,
                        cache_hit=cache_hit,
                        thinking_time_sec=thinking_time,
                        total_time_sec=(datetime.now() - t0).total_seconds()
                    )
                except Exception as db_error:
                    logger.error(f"Failed to save chat history for {conversation_id}: {db_error}")
            
            # 7. Final event to close stream
            final_event = {"type": "final", "conversation_id": conversation_id}
            yield f"data: {json.dumps(final_event)}\n\n"

    return StreamingResponse(generate_stream_with_thinking(), media_type="text/event-stream")
        
@app.get('/files/highlighted/{filename:path}')
async def serve_highlighted_document(
    filename: str,
    text: str = Query('', description="Text to highlight")
):    
    """
    Serves a .docx file with specific text highlighted.

    This is useful for showing users the exact source of information within a document.
    It reads a .docx file, finds all occurrences of the `text` query parameter,
    highlights them in yellow, and returns the modified document for download.

    Args:
        filename (str): The path to the document relative to the data directory.
        text (str): The text to find and highlight within the document.

    Returns:
        A StreamingResponse containing the modified .docx file.
    """
    # Security: Prevent directory traversal
    safe_path = Path(document_dir).joinpath(filename).resolve()
    if not safe_path.is_file() or not str(safe_path).startswith(str(Path(document_dir).resolve())):
        raise HTTPException(status_code=404, detail="File not found or access denied")

    if not text or not filename.lower().endswith('.docx'):
        return FileResponse(safe_path, media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document')

    try:
        document = Document(safe_path)
        # Use a bright color for highlighting
        highlight_color = WD_COLOR_INDEX.YELLOW

        for para in document.paragraphs:
            # Simple text search; for more complex cases, might need regex
            if text.lower() in para.text.lower():
                # This is a simplified highlighting logic. For production, a more robust
                # run-level search and replace would be needed to preserve formatting.
                # This example highlights the entire paragraph for simplicity.
                for run in para.runs:
                    run.font.highlight_color = highlight_color

        # Save to a memory buffer
        file_stream = BytesIO()
        document.save(file_stream)
        file_stream.seek(0)

        return StreamingResponse(file_stream, media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document', headers={'Content-Disposition': f'inline; filename="{filename}"'})

    except Exception as e:
        logger.error(f"Error highlighting document {filename}: {e}")
        raise HTTPException(status_code=500, detail="Could not process document.")

@app.get('/files/{filename:path}')
async def serve_document(filename: str):
    """
    Serves a static document from the configured data directory.
    This endpoint is used to provide source documents to the user.
    It includes a security check to prevent directory traversal attacks.

    Args:
        filename (str): The path to the document relative to the data directory.

    Returns:
        A FileResponse to send the document to the client.
    """
    # Security: Prevent directory traversal
    safe_path = Path(document_dir).joinpath(filename).resolve()
    if not safe_path.is_file() or not str(safe_path).startswith(str(Path(document_dir).resolve())):
        raise HTTPException(status_code=404, detail="File not found or access denied")

    return FileResponse(safe_path)

@app.get('/view_pdf/{filename:path}')
async def view_pdf_with_title(request: Request, filename: str, page: int = Query(1, description="Page number")):
    """
    Renders an HTML page to view a PDF, jumping to a specific page.

    The frontend uses this endpoint to display a source document in a new tab.
    The actual PDF is served via the /files/ endpoint, and the page number
    is passed to the PDF viewer in the template.

    Args:
        request (Request): The incoming request.
        filename (str): The path to the PDF file.
        page (int): The page number to jump to.

    Returns:
        A TemplateResponse rendering the PDF viewer page.
    """
    if templates is None:
        raise HTTPException(status_code=500, detail="Templates not configured")

    # The URL to the actual file, which will be loaded by the PDF viewer JS
    file_url = request.url_for('serve_document', filename=filename)

    return templates.TemplateResponse(
        "view_pdf.html",
        {"request": request, "file_url": file_url, "page": page, "filename": filename}
    )

@app.get("/metrics")
async def metrics(request: Request):
    """Returns operational metrics from various application components."""
    llm_client = request.app.state.components.get("llm_client")
    retrieval_engine = request.app.state.components.get("retrieval")
    return {
        "llm_client": llm_client.get_metrics() if llm_client else "Not available",
        "retrieval_engine": retrieval_engine.get_stats() if retrieval_engine else "Not available",
    }

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
        host=config.server.host,
        port=config.server.port,
        reload=True, # True in dev enviroment
        workers=1,
        log_level="info",
    )