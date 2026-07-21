"""
FILE: config.py
Giới thiệu:
-File cấu hình cho RAG

Mục đích:
- Tập trung tất cả cấu hình vào 1 class duy nhất
- Tối ưu cho Qwen 3.5-4B với vLLM Text-Only mode
- Hỗ trợ thinking mode cho suy luận phứt tạp

Cách thức vận hành:
- import from config import config
- Truy cập: config.vllm.model_name, config.qwen3.enable_thinking
- Override qua .env file hoặc environment variables:
    + RAG_VLLM_MODEL_NAME=Qwen3.5-4B
    + RAG_QWEN3_ENABLE_THINKING=false

Lưu ý:
- Model: Qwen3.5-4B
- Model Dir: models/Qwen3.5-4B
- GPU Memory: 55% (dành VRAM cho KV cache nhờ bỏ vision encoder)
- max_num_seqs: 25
- Sampling: temp 0.7, top-p 0.8, top-k 20 (non-thinking mode)
- Presnece Penalty: 1.5
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
    enabled: bool = field(default_factory=lambda: os.getenv("POSTGRES_ENABLED", "true").lower() in ("true", "1", "yes"))
    # continue
    @property
    def dsn (self) -> str:
        """_summary_
        """
        return f"postgresql"

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
    # continue
    
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
    # continue
    
# RETRIEVAL CONFIG
@dataclass
class RetrievalConfig:
    """
    Cấu hình cho RAG retrieval pipeline
    """
    # continue

# QDRANT VECTOR DATABASE CONFIG
@dataclass
class QdrantConfig:
    """
    Cấu hình Qdrant vector database
    thay thế FAISS cho vector storage với full CRUD support
    """
    # continue
    
# REDIS CONFIG
@dataclass
class RedisConfig:
    """
    cấu hình cho Redis semantic cache
    """
    # continue
    
# RERANKER CONFIG
@dataclass
class RerankerConfig:
    """_summary_
    """
    # continue
    
# RESILIENCE CONFIG
@dataclass
class ResilienceConfig:
    """
    Cấu hình cho Circuit Breaker và Graceful Degradation
    """
    # continue
    
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
    # continue
    
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
    
    #continue
    
    def print_config(self):
        """ in ra tất cả settings """
        # continue
        
config = Config()

# MAIN
if __name__ == "__main__":
    config.print_config()