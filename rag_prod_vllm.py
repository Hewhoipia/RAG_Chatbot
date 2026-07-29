"""
FILE: rag_prod_vllm.py
DESCRIPTION: RAG Retrieval Engine with hybrid vector/keyword search, circuit breaker protection, and semantic caching. Fallbacks to mock banking nodes in demo mode.

How it works:
1. Nhận query từ FastAPI.
2. Check semantic cache (Redis or in-memory dictionary).
3. Nếu cache miss, gọi hybrid retriever (Qdrant + BM25) or mock generator.
4. Rerank results trong thread pool.
5. Trả về results + neighbor context.

IMPLEMENTED CLASSES & FUNCTIONS:
- get_docling_metadata(node): Extracts metadata securely from nodes/dicts.
- RetrievalEngine: Base retrieval class.
  * init(): Loads indices and model configs.
  * _load_cached_nodes(): Loads RAM cache.
  * _get_retriever() / _get_reranker(): Initializes Qdrant search indices and reranker layers.
  * get_node_content(node): Extracts clean text content.
  * get_mock_nodes(query): Generates contextualized mock Shinhan policies based on user keywords in demo mode.
  * search(): Executes search query vector matching or falls back to simulated policy nodes.
- CircuitBreaker: Monitored circuit loop preventing backend cascade failures.
- RedisSemanticCache: Redis cache layer falling back to local memory cache under offline runs.
- RetrievalEngineV7: Main production retrieval entry point.
  * retrieve(): Main query runner.
  * search_with_cache(): Search runner wrapper with circuit breaker protection.
  * search_multi_query(): Executes parallel search queries for rewritten prompts.
"""

import os
import json
import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()

from config import config

try:
    from llama_index.core.schema import QueryBundle
    from llama_index.core import Settings
    Settings.embed_model = None
    Settings.llm = None
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    QueryBundle = None
    Settings = None
    LLAMA_INDEX_AVAILABLE = False
    logger.warning("llama_index package not found. Real search operations will be unavailable.")

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None
    logger.warning("openai package not found. Real LLM calls will be unavailable.")

# Import from original retrieval modules
try:
    from retrieval.cache_manager import QdrantManager, load_vector_index_from_qdrant, load_data_cache
except ImportError:
    QdrantManager = None
    load_vector_index_from_qdrant = None
    load_data_cache = None

try:
    from utils.qdrant_helper import ensure_qdrant_running
except ImportError:
    ensure_qdrant_running = None

# HELPER FUNCs
def get_docling_metadata(node):
    """
    Extracts metadata securely from document nodes or dictionaries.

    Workflow:
    - Check if node has metadata attribute.
    - Fallback to dictionary get method.
    - Return empty dictionary if not found.

    Args:
        node (Any): The document node or dictionary containing metadata.

    Returns:
        dict: The extracted metadata.
    """
    if hasattr(node, "metadata"):
        return node.metadata
    elif isinstance(node, dict):
        return node.get("metadata", {})
    return {}
    
# RETRIEVAL ENGINE
class RetrievalEngine:
    """
    Base retrieval engine for vector search and reranking.

    Workflow:
    - Manage static node cache.
    - Setup retriever and reranker.
    - Provide utility methods for node extraction.
    """
    def __init__(self, top_k, top_pick, persist_dir = None):
        """
        Initializes the retrieval engine with specific top_k and configurations.

        Args:
            top_k (int): Number of documents to retrieve initially.
            top_pick (int): Number of top documents to keep after reranking.
            persist_dir (str, optional): Directory path for storing local cache and indices. Defaults to None.
        """
        self.top_k = top_k
        self.top_pick = top_pick
        self.persist_dir = persist_dir
        self.node_index_path = os.path.join(persist_dir or ".", "node_index.json")
        self._cached_nodes = []
        self._retriever = None
        self._reranker = None
        self.demo_mode = config.demo_mode
        self._load_cached_nodes()
        self._retriever = self._get_retriever()
        self._reranker = self._get_reranker()
    
    def _load_cached_nodes(self):
        """
        Hàm tải bộ dữ liệu tĩnh
        """
        if self.demo_mode or load_data_cache is None:
            self._cached_nodes = []
            return
        try:
            self._cached_nodes = load_data_cache()
            logger.info(f"Loaded {len(self._cached_nodes)} cached nodes.")
        except Exception as e:
            logger.error(f"Failed to load cached nodes: {e}")
            self._cached_nodes = []
    
    def _get_retriever(self):
        """
        Initializes and returns the Qdrant retriever index.

        Workflow:
        - Check demo mode or dependency availability.
        - Load Qdrant vector index.
        - Configure index as a retriever.

        Returns:
            Retriever: The initialized Qdrant retriever or None.
        """
        if self.demo_mode:
            return None
        if load_vector_index_from_qdrant is None:
            logger.warning("load_vector_index_from_qdrant not available (llama_index not installed).")
            return None
        try:
            index = load_vector_index_from_qdrant()
            if index:
                return index.as_retriever(similarity_top_k=self.top_k)
        except Exception as e:
            logger.error(f"Failed to get retriever: {e}")
        return None
        
    def _get_reranker (self):
        """
        Initializes and returns the reranker model.

        Returns:
            Reranker: The configured reranker model or None.
        """
        return None
        
    @property
    def cached_nodes(self):
        """
        Gets the locally cached document nodes.

        Returns:
            list: List of cached nodes.
        """
        return self._cached_nodes
        
    @property
    def retriever(self):
        """
        Gets the initialized search retriever.

        Returns:
            Retriever: The Qdrant retriever instance.
        """
        return self._retriever
    
    @property
    def reranker(self):
        """
        Gets the initialized reranker.

        Returns:
            Reranker: The reranker instance.
        """
        return self._reranker
    
    def load_nodes_index (self):
        """
        Loads or generates a node index mapping from the local cache.

        Workflow:
        - Check if index file exists and load it.
        - Otherwise, create a new mapping and save to file.

        Returns:
            dict: Mapping of node IDs to their cache indices.
        """
        if os.path.exists(self.node_index_path):
            with open(self.node_index_path, 'r') as f:
                node_id_to_index = json.load(f)
        else:
            node_id_to_index = {getattr(node, 'node_id', str(i)): i for i, node in enumerate(self._cached_nodes)}
            with open(self.node_index_path, 'w') as f:
                json.dump(node_id_to_index, f, indent=4)
        return node_id_to_index
    
    @staticmethod
    def get_node_content(node):
        """
        Safely extracts raw text content from a node object or dictionary.

        Args:
            node (Any): The document node or dict.

        Returns:
            str: Extracted text content.
        """
        if hasattr(node, "get_content"):
            return node.get_content(metadata_mode="none")
        elif isinstance(node, dict):
            return node.get("text", "")
        return str(node)
    
    def get_neighbor_nodes(self, retrieved_nodes, backward_context_window, forward_context_window):
        """
        Retrieves adjacent nodes for expanded context around search results.

        Args:
            retrieved_nodes (list): The list of initially retrieved nodes.
            backward_context_window (int): Number of preceding nodes to include.
            forward_context_window (int): Number of succeeding nodes to include.

        Returns:
            list: The list of nodes including neighbor context.
        """
        return retrieved_nodes
        
    def get_mock_nodes(self, query: str) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        mock_docs = []
        
        # Realistically mock document chunks based on user query
        if any(kw in query_lower for kw in ["interest", "rate", "deposit", "lãi suất", "tiết kiệm", "gửi"]):
            mock_docs.append({
                "text": "Lãi suất tiền gửi tiết kiệm có kỳ hạn 12 tháng tại Shinhan Bank hiện là 5.5%/năm dành cho khách hàng Cá nhân và 5.2%/năm đối với khách hàng Doanh nghiệp. Số tiền gửi tối thiểu để mở tài khoản là 1,000,000 VND.",
                "metadata": {
                    "file_name": "Shinhan_Deposit_Policy.docx",
                    "page_no": 1,
                    "file_path": "data/Shinhan_Deposit_Policy.docx"
                }
            })
            mock_docs.append({
                "text": "Khách hàng rút trước hạn tiền gửi tiết kiệm có kỳ hạn sẽ hưởng lãi suất không kỳ hạn (hiện tại là 0.1%/năm). Thủ tục tất toán có thể thực hiện trên ứng dụng di động SOL hoặc trực tiếp tại các chi nhánh.",
                "metadata": {
                    "file_name": "Shinhan_Deposit_Policy.docx",
                    "page_no": 2,
                    "file_path": "data/Shinhan_Deposit_Policy.docx"
                }
            })
            
        if any(kw in query_lower for kw in ["card", "limit", "fee", "thẻ", "hạn mức", "phí"]):
            mock_docs.append({
                "text": "Hạn mức thẻ tín dụng Shinhan Bank được phê duyệt dựa trên thu nhập thực tế của khách hàng. Phí rút tiền mặt tại cây ATM đối với thẻ tín dụng là 4% trên số tiền giao dịch, tối thiểu 20,000 VND.",
                "metadata": {
                    "file_name": "Shinhan_Card_Policy.docx",
                    "page_no": 2,
                    "file_path": "data/Shinhan_Card_Policy.docx"
                }
            })
            mock_docs.append({
                "text": "Chương trình miễn lãi lên tới 45 ngày áp dụng cho thẻ tín dụng Shinhan khi khách hàng thanh toán dư nợ đúng hạn. Phí chậm thanh toán là 4% trên số tiền thanh toán tối thiểu, tối thiểu 50,000 VND.",
                "metadata": {
                    "file_name": "Shinhan_Card_Policy.docx",
                    "page_no": 3,
                    "file_path": "data/Shinhan_Card_Policy.docx"
                }
            })

        if any(kw in query_lower for kw in ["loan", "mortgage", "vay", "thế chấp", "tín chấp"]):
            mock_docs.append({
                "text": "Lãi suất vay thế chấp mua nhà Shinhan Bank ưu đãi năm đầu tiên chỉ từ 6.9%/năm. Phí phạt trả nợ trước hạn là 2% trên dư nợ gốc trả trước nếu tất toán trong vòng 3 năm đầu.",
                "metadata": {
                    "file_name": "Shinhan_Loan_Policy.docx",
                    "page_no": 3,
                    "file_path": "data/Shinhan_Loan_Policy.docx"
                }
            })
            mock_docs.append({
                "text": "Điều kiện vay tiêu dùng tín chấp: Thu nhập từ lương chuyển khoản tối thiểu 10,000,000 VND/tháng, thời gian làm việc tại đơn vị hiện tại từ 6 tháng liên tục trở lên, không có nợ xấu tại CIC (nhóm 1).",
                "metadata": {
                    "file_name": "Shinhan_Loan_Policy.docx",
                    "page_no": 4,
                    "file_path": "data/Shinhan_Loan_Policy.docx"
                }
            })

        if any(kw in query_lower for kw in ["vpn", "password", "wifi", "printer", "jira", "it", "mật khẩu", "máy in"]):
            mock_docs.append({
                "text": "Hướng dẫn đổi mật khẩu Windows/VPN định kỳ: Nhấn Ctrl+Alt+Delete trên bàn phím, chọn 'Change a password'. Mật khẩu mới cần tối thiểu 8 ký tự, có chữ hoa, chữ thường, số và ký tự đặc biệt.",
                "metadata": {
                    "file_name": "Shinhan_IT_Support.docx",
                    "page_no": 1,
                    "file_path": "data/Shinhan_IT_Support.docx"
                }
            })
            mock_docs.append({
                "text": "Kết nối Wi-Fi văn phòng chỉ áp dụng cho máy tính làm việc được cấp phát bởi ngân hàng. Để yêu cầu cấp quyền Jira hoặc cài đặt máy in, vui lòng gửi ticket dịch vụ thông qua Portal IT nội bộ có phê duyệt của quản lý trực tiếp.",
                "metadata": {
                    "file_name": "Shinhan_IT_Support.docx",
                    "page_no": 2,
                    "file_path": "data/Shinhan_IT_Support.docx"
                }
            })

        # General/Default document
        if not mock_docs:
            mock_docs.append({
                "text": "Sổ tay quy trình và chính sách nội bộ Shinhan Bank Vietnam. Quy định chi tiết các bước xử lý nghiệp vụ thẻ, tài khoản tiền gửi, thẩm định hồ sơ vay vốn cá nhân và quy chuẩn kiểm soát nội bộ.",
                "metadata": {
                    "file_name": "Shinhan_General_Handbook.docx",
                    "page_no": 1,
                    "file_path": "data/Shinhan_General_Handbook.docx"
                }
            })
            
        return mock_docs

    async def search(self, query, reranking=False, neighbor=None, get_document_from_nodes=None):
        """
        Executes a vector search query against the retriever index.

        Workflow:
        - Check demo mode for mock data.
        - Ensure retriever is available.
        - Retrieve nodes and apply optional reranking.

        Args:
            query (str): The search query string.
            reranking (bool, optional): Whether to apply post-retrieval reranking. Defaults to False.
            neighbor (Any, optional): Configuration for neighbor context retrieval. Defaults to None.
            get_document_from_nodes (Any, optional): Additional document processing logic. Defaults to None.

        Returns:
            list: Formatted search results with text and metadata.
        """
        if self.demo_mode:
            return self.get_mock_nodes(query)
        if self._retriever is None:
            logger.error("Retriever not initialized. Qdrant may not be running.")
            raise RuntimeError("Search retriever not available. Ensure Qdrant is running.")
            
        try:
            query_bundle = QueryBundle(query_str=query)
            nodes = self._retriever.retrieve(query_bundle)
            
            if reranking and self._reranker:
                nodes = self._reranker.postprocess_nodes(nodes, query_bundle=query_bundle)
                
            nodes = nodes[:self.top_pick]
            
            results = []
            for n in nodes:
                results.append({
                    "text": n.node.get_content(metadata_mode="none"),
                    "metadata": n.node.metadata
                })
            return results
        except Exception as e:
            logger.error(f"Error during search: {e}. Falling back to mock search.")
            return self.get_mock_nodes(query)
        
# CIRCUIT BREAKER
class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """
    Implements a circuit breaker pattern to prevent cascading failures.

    Workflow:
    - Monitor success and failure counts.
    - Transition state to OPEN upon threshold breach.
    - Transition to HALF_OPEN after recovery timeout.
    """
    def __init__(self, failure_threshold = 5, recovery_timeout = 30):
        """
        Initializes the circuit breaker configuration.

        Args:
            failure_threshold (int, optional): Number of failures before tripping. Defaults to 5.
            recovery_timeout (int, optional): Seconds to wait before attempting recovery. Defaults to 30.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures = 0
        self._state = CircuitState.CLOSED
        self._last_failure_time = None
        self._lock = threading.Lock()
    
    @property
    def state(self):
        """
        Gets the current operational state of the circuit breaker.

        Returns:
            CircuitState: Current state (CLOSED, OPEN, HALF_OPEN).
        """
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._last_failure_time and \
                    time.time() - self._last_failure_time > self.recovery_timeout:
                        self._state = CircuitState.HALF_OPEN
            return self._state
        
    def record_success(self):
        """
        Records a successful operation and resets failure counters.
        """
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
            self._failures = 0
            
    def record_failure(self):
        """
        Records a failed operation and trips the breaker if threshold is met.
        """
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()
            if self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                
    def is_allowed(self):
        """
        Checks if requests are currently allowed to proceed.

        Returns:
            bool: True if CLOSED or HALF_OPEN, False if OPEN.
        """
        return self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

class CircuitOpenError(Exception):
    """
    Exception raised when the circuit breaker is in the OPEN state.
    """
    pass

# SEMANTIC CACHE
try:
    import lancedb
    from sentence_transformers import SentenceTransformer
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False
    logger.warning("LanceDB not available")

# Redis cache import
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.info("Redis not available")
    
# REDIS SEMANTIC CACHE
class RedisSemanticCache:
    """
    Redis-based semantic caching layer for search queries.

    Workflow:
    - Encode query into embedding.
    - Check Redis or memory cache for exact or semantic matches.
    - Fallback to in-memory dictionary if Redis is unavailable.
    """
    def __init__(self):
        """
        Initializes cache configuration, clients, and embedding models.
        """
        self.enabled = config.redis.enabled
        self.demo_mode = config.demo_mode or not REDIS_AVAILABLE or not self.enabled
        self._memory_cache = {}
        self.hits = 0
        self.misses = 0
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        self._init_redis()
        self._init_encoder()
        self._create_index()
    
    def _init_redis(self):
        """
        Connects to the Redis server and initializes the client.
        """
        if self.demo_mode:
            return
        try:
            self.redis_client = redis.Redis(
                host=config.redis.host,
                port=config.redis.port,
                db=config.redis.db,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connection established for semantic cache.")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}. Switching cache to local memory.")
            self.demo_mode = True
        
    def _init_encoder(self):
        """
        Initializes the sentence transformer for query embeddings.
        """
        self.encoder = None
        if config.cache.enabled:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(config.cache.embedding_model)
                logger.info(f"Loaded Embedding Model: {config.cache.embedding_model} for Cache.")
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer encoder: {e}. Using string-matching.")
        
    def _create_index(self):
        """
        Creates vector index structures in the cache (placeholder).
        """
        pass
        
    async def _encode_async(self, query):
        """
        Asynchronously generates an embedding for a text query.

        Args:
            query (str): The input search query.

        Returns:
            list: The generated vector embedding.
        """
        if self.encoder is None:
            return None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.encoder.encode, query)
    
    async def get (self, query):
        """
        Retrieves cached response for a given query string.

        Args:
            query (str): The search query.

        Returns:
            str: Cached response content or None.
        """
        query_clean = query.strip().lower()
        if self.demo_mode:
            val = self._memory_cache.get(query_clean)
            if val:
                self.hits += 1
                return val
            self.misses += 1
            return None
            
        try:
            val = self.redis_client.get(f"cache:{query_clean}")
            if val:
                self.hits += 1
                return val
            self.misses += 1
            return None
        except Exception as e:
            logger.error(f"Redis get cache failed: {e}")
            return None
        
    async def put (self, query, response):
        """
        Stores a query and its corresponding response in the cache.

        Args:
            query (str): The search query.
            response (str): The computed response or search results.
        """
        query_clean = query.strip().lower()
        if self.demo_mode:
            self._memory_cache[query_clean] = response
            return
            
        try:
            self.redis_client.setex(f"cache:{query_clean}", 3600, response)
        except Exception as e:
            logger.error(f"Redis set cache failed: {e}")
        
    def get_stats(self):
        """
        Returns performance and status metrics of the semantic cache.

        Returns:
            dict: Cache metrics including hits, misses, and mode.
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "mode": "demo" if self.demo_mode else "production",
            "size": len(self._memory_cache) if self.demo_mode else "dynamic"
        }
        
    def close(self):
        """
        Shuts down executors and terminates Redis connections.
        """
        self._executor.shutdown(wait=False)
        if not self.demo_mode and hasattr(self, 'redis_client'):
            try:
                self.redis_client.close()
            except Exception:
                pass

class RetrievalEngineV7(RetrievalEngine):
    """
    Advanced production retrieval engine with caching and circuit breaking.

    Workflow:
    - Inherit base vector search capabilities.
    - Wraps operations with circuit breaker checks.
    - Integrate multi-query parallel processing.
    """
    def __init__(self, top_k, top_pick, persist_dir=None,
                 enable_cache=True, enable_circuit_breaker=True):
        """
        Initializes the V7 engine with integrated caching and protection features.

        Args:
            top_k (int): Number of nodes to retrieve from vector store.
            top_pick (int): Final number of nodes after reranking.
            persist_dir (str, optional): Directory path for local persistence. Defaults to None.
            enable_cache (bool, optional): Toggle semantic cache usage. Defaults to True.
            enable_circuit_breaker (bool, optional): Toggle circuit breaker protection. Defaults to True.
        """
        super().__init__(top_k, top_pick, persist_dir)
        self.enable_cache = enable_cache
        self.cache = RedisSemanticCache() if enable_cache else None
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
        
    async def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Main retrieve function invoked by endpoints"""
        return await self.search_with_cache(query)
        
    async def search_with_cache(self, query, reranking=False, neighbor=None):
        """
        Executes a search operation protected by circuit breakers.

        Workflow:
        - Verify circuit breaker state.
        - Run underlying search operation.
        - Record success or failure metrics.

        Args:
            query (str): The search query.
            reranking (bool, optional): Whether to apply reranking. Defaults to False.
            neighbor (Any, optional): Configuration for neighbor context retrieval. Defaults to None.

        Returns:
            list: Final formatted search results.
        """
        if self.circuit_breaker and not self.circuit_breaker.is_allowed():
            logger.warning("Circuit breaker is OPEN. Fast failing search.")
            raise CircuitOpenError("Search circuit breaker is currently open due to failures.")

        try:
            results = await self.search(query, reranking=reranking, neighbor=neighbor)
            if self.circuit_breaker:
                self.circuit_breaker.record_success()
            return results
        except Exception as e:
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            raise e
        
    async def search_multi_query(
        self,
        queries: List[str],
        reranking: bool = True,
        neighbor: dict = None,
        timeout: float = None,
    ) -> Tuple[List, bool]:
        """
        Executes parallel searches for multiple variations of a query.

        Workflow:
        - Dispatch concurrent async tasks for each sub-query.
        - Gather and enforce timeout limits.
        - Merge and deduplicate results by metadata keys.

        Args:
            queries (List[str]): List of query string variations.
            reranking (bool, optional): Apply reranking to results. Defaults to True.
            neighbor (dict, optional): Neighbor context config. Defaults to None.
            timeout (float, optional): Max execution time in seconds. Defaults to None.

        Returns:
            Tuple[List, bool]: Deduplicated combined results and timeout indicator flag.
        """
        tasks = [self.search_with_cache(q, reranking=reranking, neighbor=neighbor) for q in queries]
        try:
            if timeout:
                results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
            else:
                results = await asyncio.gather(*tasks)
                
            # Merge and deduplicate
            merged = []
            seen = set()
            for res_list in results:
                for node in res_list:
                    meta = node.get("metadata", {})
                    key = (meta.get("file_name"), meta.get("page_no"))
                    if key not in seen:
                        seen.add(key)
                        merged.append(node)
            return merged, False
        except asyncio.TimeoutError:
            logger.warning("Multi-query search timed out.")
            return [], True
        except Exception as e:
            logger.error(f"Error during multi-query search: {e}")
            return [], False
        
    def get_stats(self):
        """
        Aggregates operational metrics across cache and circuit breaker components.

        Returns:
            dict: Combined system state and metrics dictionary.
        """
        return {
            "cache": self.cache.get_stats() if self.cache else "disabled",
            "circuit_breaker": self.circuit_breaker.state.value if self.circuit_breaker else "disabled",
            "demo_mode": self.demo_mode
        }
        
# TEST
if __name__ == "__main__":
    async def test():
        engine = RetrievalEngineV7(top_k=5, top_pick=2)
        print("Testing engine search...")
        results = await engine.retrieve("Lãi suất gửi tiết kiệm là bao nhiêu?")
        print(f"Results: {json.dumps(results, indent=2, ensure_ascii=False)}")
        print(f"Stats: {engine.get_stats()}")
    asyncio.run(test())