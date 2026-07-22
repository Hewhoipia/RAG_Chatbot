"""
FILE: config.py
GIỚI THIỆU (Introduction):
- File cấu hình cho RAG pipeline, hỗ trợ chế độ demo ngoại tuyến (offline demo mode).

MỤC ĐÍCH (Purpose):
- Tập trung tất cả cấu hình vào 1 class duy nhất.
- Tối ưu cho Qwen 3.5-4B với vLLM Text-Only mode.
- Hỗ trợ thinking mode cho suy luận phức tạp.

CÁCH THỨC VẬN HÀNH (How it works):
- import from config import config
- Truy cập: config.vllm.model_name, config.qwen3.enable_thinking
- Override qua .env file hoặc environment variables.

LỚP CẤU HÌNH & THUỘC TÍNH ĐÃ TRIỂN KHAI (Implemented Classes & Properties):
- get_env, get_env_bool, get_env_int, get_env_float: Trích xuất và định kiểu các biến môi trường RAG_.
- Qwen3Config: Quản lý các cấu hình suy luận (thinking/non-thinking modes), nhiệt độ, top_p, top_k, presence_penalty và trả về sampling parameters qua get_sampling_params().
- QueryAnalysisConfig: Điều phối tham số multi-query rewrite, phân rã (decomposition), step-back và reciprocal rank fusion.
- ConcurrencyConfig: Quản lý kích thước luồng xử lý của reranker pool.
- DatabaseConfig: Tham số host, port, thông tin đăng nhập PostgreSQL, kích thước connection pool, và thuộc tính động dsn.
- VLLMConfig: Cấu hình địa chỉ cơ sở base_url và tên mô hình model_name cho vLLM server.
- CacheConfig: Cấu hình thư mục lưu trữ cache LanceDB và ngưỡng tương đồng.
- RetrievalConfig: Thư mục chứa tài liệu, số lượng top_k, top_pick và kích thước context window.
- QdrantConfig: Tham số kết nối Vector database Qdrant.
- RedisConfig: Cấu hình Redis cache.
- RerankerConfig: Tham số kích hoạt reranker model.
- ResilienceConfig: Cấu hình circuit breaker failure threshold và recovery timeout.
- ServerConfig: Tham số uvicorn host/port, bật/tắt rate limiter, max concurrent requests.
- Config: Lớp tổng hợp chứa toàn bộ cấu hình con, hỗ trợ thuộc tính demo_mode (mặc định True) và hàm print_config() để in báo cáo.
"""

import os
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# load envi vari from .env
load_dotenv()

# HELPER FUNCs
def get_env(key: str, default: str, prefix: str = "RAG_") -> str:
    """
    Lấy giá trị từ envi vari tới prefix
    ví dụ:
        get_env("DB_HOST", "localhost")
        -> tìm RAG_DB_HOST, nếu không trả về "localhost"
    """
    return os.getenv(f"{prefix}{key}", os.getenv(key, default))

def get_env_bool(key: str, default: str, prefix: str = "RAG_") -> bool:
    """
    bool từ envi vari
    """
    value = get_env(key, str(default), prefix)
    return value.lower() in ("true", "1", "yes", "on")

def get_env_int(key: str, default: str, prefix: str = "RAG_") -> int:
    return int(get_env(key, str(default), prefix))

def get_env_float(key: str, default: str, prefix: str = "RAG_") -> float:
    return float(get_env(key, str(default), prefix))

# QWEN3 SPECIFIC CONFIGURATION
@dataclass
class Qwen3Config:
    """_summary_
    """
    # Thinking mode toggle
    enable_thinking: bool = field(default_factory=lambda: get_env_bool("QWEN3_ENABLE_THINKING", False))
    # Reasoning
    enable_reasoning: bool = field(default_factory=lambda: get_env_bool("QWEN3_ENABLE_REASONING", True))
    reasoning_parser: str = field(default_factory=lambda: get_env("QWEN3_REASONING_PARSER", "qwen3"))

    # Sampling params - non-thinking mode
    temperature_non_thinking: float = field(default_factory=lambda: get_env_float("QWEN3_TEMP_NON_THINKING", 0.7))
    top_p_non_thinking: float = field(default_factory=lambda: get_env_float("QWEN3_TOP_P_NON_THINKING", 0.8))
    
    #Sampling params - thinking mode
    temperature_thinking: float = field(default_factory=lambda: get_env_float("QWEN3_TEMP_THINKING", 0.6))
    top_p_thinking: float = field(default_factory=lambda: get_env_float("QWEN3_TOP_P_THINKING", 0.95))

    # common sampling params
    top_k: int = field(default_factory=lambda: get_env_int("QWEN3_TOP_K", 20))
    min_p: float = field(default_factory=lambda: get_env_float("QWEN3_MIN_P", 0.0))

    presence_penalty: float = field(default_factory=lambda: get_env_float("QWEN3_PRESENCE_PENALTY", 1.5))

    max_tokens_thinking: int = field(default_factory=lambda: get_env_int("QUEN3_MAX_TOKENS_THINKING", 4096))
    max_tokens_non_thinking: int = field(default_factory=lambda: get_env_int("QUEN3_MAX_TOKENS_NON_THINKING", 2048))

    def get_sampling_params (self, thinking: bool = None) -> dict:
        """_summary_

        Args:
            thinking (bool, optional): _description_. Defaults to None.

        Returns:
            dict: _description_
        """
        use_thinking = thinking if thinking is not None else self.enable_thinking
        if use_thinking:
            return {
                "temperature": self.temperature_thinking,
                "top_p": self.top_p_thinking,
                "top_k": self.top_k,
                "presence_penalty": self.presence_penalty,
                "max_tokens": self.max_tokens_thinking,
            }
        else:
            return {
                "temperature": self.temperature_non_thinking,
                "top_p": self.top_p_non_thinking,
                "top_k": self.top_k,
                "presence_penalty": self.presence_penalty,
                "max_tokens": self.max_tokens_thinking,
            }
            
# QUERY ANALYSIS CONFIG
@dataclass
class QueryAnalysisConfig:
    """_summary_
    """
    enabled: bool = field(default_factory=lambda: get_env_bool("QUERY_ANALYSIS_ENABLED", True))
    classifier_mode: str = field(default_factory=lambda: get_env("QUERY_CLASSIFIER_MODE", "hybrid"))
    multi_query_top_k: int = field(default_factory=lambda: get_env_int("MULTI_QUERY_TOP_K", 10))
    multi_query_timeout:float = field(default_factory=lambda: get_env_float("MULTI_QUERY_TIMEOUT", 15.0))
    simple_rewrite_count: int = field(default_factory=lambda: get_env_int("SIMPLE_REWRITE_COUNT", 2))
    complex_rewrite_count: int = field(default_factory=lambda: get_env_int("COMPLEX_REWRITE_COUNT", 2))
    complex_decomp_count: int = field(default_factory=lambda: get_env_int("COMPLEX_DECOMP_COUNT", 2))

    rrf_k: int = field(default_factory=lambda: get_env_int("RRF_K", 60))

# CONCURRENCY CONFIG
@dataclass
class ConcurrencyConfig:
    """_summary_
    """
    reranker_pool_size: int = field(default_factory=lambda: get_env_int("RERANKER_POOL_SIZE", 8))

@dataclass
class DatabaseConfig:
    """_summary_
    """
    enabled: bool = field(default_factory=lambda: get_env_bool("POSTGRES_ENABLED", True))
    host: str = field(default_factory=lambda: get_env("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: get_env_int("DB_PORT", 5432))
    user: str = field(default_factory=lambda: get_env("DB_USER", "postgres"))
    password: str = field(default_factory=lambda: get_env("DB_PASSWORD", "postgres"))
    database: str = field(default_factory=lambda: get_env("DB_DATABASE", "rag_chatbot"))
    min_size: int = field(default_factory=lambda: get_env_int("DB_MIN_SIZE", 5))
    max_size: int = field(default_factory=lambda: get_env_int("DB_MAX_SIZE", 20))

    @property
    def dsn(self) -> str:
        """_summary_
        """
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

# VLLM SERVER CONFIGURATION
@dataclass
class VLLMConfig:
    """
    Giới thiệu: cấu hình cho vLLM OpenAI-compatible API server
    Mục đích: kết nối đến vLLM server
    Cách thức vận hành:
    - FastAPI gọi vLLM server qua HTTP API
    - vLLM xử lý continuous batching và PagedAttention
    - Text-only mode để tiết kiệm VRAM
    LƯU Ý:
    - model_name phải match với --served-model-name trong vllm_server_llm.sh
    - GPU Mem: 55%
    """
    base_url: str = field(default_factory=lambda: get_env("VLLM_BASE_URL", "http://localhost:8011/v1"))
    model_name: str = field(default_factory=lambda: get_env("VLLM_MODEL_NAME", "Qwen3.5-4B"))
    gpu_memory_utilization: float = field(default_factory=lambda: get_env_float("VLLM_GPU_MEMORY_UTILIZATION", 0.55))
    max_num_seqs: int = field(default_factory=lambda: get_env_int("VLLM_MAX_NUM_SEQS", 25))
    
# CACHE CONFIG
@dataclass
class CacheConfig:
    """
    Giới thiệu: cấu hình cho Semantic Cache
    Mục đích: Cache responses để giảm LLM calls cho queries tương tự
    Cách thức vận hành:
    - Sử dụng LanceDB để lưu embeddings và responses
    - Encode queries với SentenceTransformer
    - Match bằng cosine simi (threshold 0.95)

    Lưu ý:
    - Sử dụng LanceDB lưu trữ nội bộ
    """
    enabled: bool = field(default_factory=lambda: get_env_bool("CACHE_ENABLED", True))
    persist_dir: str = field(default_factory=lambda: get_env("CACHE_PERSIST_DIR", "cache/lancedb"))
    similarity_threshold: float = field(default_factory=lambda: get_env_float("CACHE_SIMILARITY_THRESHOLD", 0.95))
    embedding_model: str = field(default_factory=lambda: get_env("CACHE_EMBEDDING_MODEL", "BAAI/bge-m3"))
    
# RETRIEVAL CONFIG
@dataclass
class RetrievalConfig:
    """
    Cấu hình cho RAG retrieval pipeline
    """
    data_dir: str = field(default_factory=lambda: get_env("RETRIEVAL_DATA_DIR", "data"))
    top_k: int = field(default_factory=lambda: get_env_int("RETRIEVAL_TOP_K", 10))
    top_pick: int = field(default_factory=lambda: get_env_int("RETRIEVAL_TOP_PICK", 3))
    backward_context_window: int = field(default_factory=lambda: get_env_int("RETRIEVAL_BACKWARD_CONTEXT", 2))
    forward_context_window: int = field(default_factory=lambda: get_env_int("RETRIEVAL_FORWARD_CONTEXT", 2))

# QDRANT VECTOR DATABASE CONFIG
@dataclass
class QdrantConfig:
    """
    Cấu hình Qdrant vector database
    thay thế FAISS cho vector storage với full CRUD support
    """
    host: str = field(default_factory=lambda: get_env("QDRANT_HOST", "localhost"))
    port: int = field(default_factory=lambda: get_env_int("QDRANT_PORT", 6333))
    collection_name: str = field(default_factory=lambda: get_env("QDRANT_COLLECTION", "banking_docs"))
    
# REDIS CONFIG
@dataclass
class RedisConfig:
    """
    cấu hình cho Redis semantic cache
    """
    enabled: bool = field(default_factory=lambda: get_env_bool("REDIS_ENABLED", True))
    host: str = field(default_factory=lambda: get_env("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: get_env_int("REDIS_PORT", 6379))
    db: int = field(default_factory=lambda: get_env_int("REDIS_DB", 0))
    
# RERANKER CONFIG
@dataclass
class RerankerConfig:
    """_summary_
    """
    model_name: str = field(default_factory=lambda: get_env("RERANKER_MODEL", "BAAI/bge-reranker-large"))
    enabled: bool = field(default_factory=lambda: get_env_bool("RERANKER_ENABLED", True))
    
# RESILIENCE CONFIG
@dataclass
class ResilienceConfig:
    """
    Cấu hình cho Circuit Breaker và Graceful Degradation
    """
    circuit_breaker_failure_threshold: int = field(default_factory=lambda: get_env_int("CIRCUIT_BREAKER_FAILURES", 5))
    circuit_breaker_recovery_timeout: float = field(default_factory=lambda: get_env_float("CIRCUIT_BREAKER_RECOVERY", 30.0))
    
# SERVER CONFIG
@dataclass
class ServerConfig:
    """
    Giới thiệu: cấu hình cho FASTAPI server
    Mục đích: xử lý 20+ concurrent users
    Cách thức vận hành:
    - AdmissionController giới hạn concurrent requests
    - TokenBucketRateLimiter giới hạn per-IP requests
    - Graceful degradation khi overload
    
    Lưu ý: max_concurrent_requests tăng lên 25 (MoE xử lý nhanh hơn dense model)
    """
    host: str = field(default_factory=lambda: get_env("SERVER_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: get_env_int("SERVER_PORT", 8010))
    rate_limit_enabled: bool = field(default_factory=lambda: get_env_bool("RATE_LIMIT_ENABLED", True))
    rate_limit_capacity: int = field(default_factory=lambda: get_env_int("RATE_LIMIT_CAPACITY", 10))
    rate_limit_fill_rate: float = field(default_factory=lambda: get_env_float("RATE_LIMIT_FILL_RATE", 1.0))
    max_concurrent_requests: int = field(default_factory=lambda: get_env_int("MAX_CONCURRENT_REQUESTS", 25))
    
# MAIN CONFIG
@dataclass
class Config:
    """
    Giới thiệu: unified config cho RAG
    Mục đích:
    - Tối ưu Qwen3.5-4B
    - 20+ concurrent users với hiệu suất tốt trên 1 GPU
    - Support Thinking/Non-Thinking modes
    Cách thức vận hành:
    - Import: from config import config
    - truy cập config.vllm.model_name, config.qwen3.enable_thinking
    - override qua envi vari với prefix RAG_
    
    Lưu ý:
    - Tất cả sub-configs được tạo với default factory
    - Envi vari có precedence cao hơn defaults
    - Mode: Qwen3.5-4B | Dir: models/Qwen3.5-4B
    - qwen3 config cho thinking mode controls
    """
    
    qwen3: Qwen3Config = field(default_factory=Qwen3Config)
    query_analysis: QueryAnalysisConfig = field(default_factory=QueryAnalysisConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    demo_mode: bool = field(default_factory=lambda: get_env_bool("DEMO_MODE", True))
    
    def print_config(self):
        """ in ra tất cả settings """
        print("=== RAG CHATBOT CONFIGURATION ===")
        print(f"Demo Mode: {self.demo_mode}")
        print(f"Database DSN: {self.database.dsn} (Enabled: {self.database.enabled})")
        print(f"vLLM URL: {self.vllm.base_url} | Model: {self.vllm.model_name}")
        print(f"Thinking Mode Default: {self.qwen3.enable_thinking}")
        print(f"Server Host/Port: {self.server.host}:{self.server.port}")
        
config = Config()

# MAIN
if __name__ == "__main__":
    config.print_config()