# Giới thiệu:
# - main interface (Factory + abstract class) cho Query Classification Router
# Mục đích:
# 1. Định nghĩa output chuẩn cho Query Classification Router
# 2. Factory pattern để đổi giữa 3 options (A, B+, C)
# 3. Abstract base class đảm bảo mọi classifier đều có method classify()
# Cách sử dụng:
# from classifier.query_classifier import QueryClassifier, ClassifierMode
# classifier = QueryClassifier.create(mode = ClassifierMode.HYBRID)
# result = classifier.classify("...")
# print (result.label, result.confidence)
# Lưu ý:
# - HYBRID mode cần file classifier_router.pkl (chạy build_router_cache.py trước)
# - LLM_BASED mode cần vLLM server đang chạy tại localhost:8011
# - RULE_BASED mode chạy offline, k cần GPU

import time
import logging
from enum import Enum
from typing import Literal, Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel
logger = logging.getLogger(__name__)

# ENUM: chế độ classifier
class ClassifierMode(str, Enum):
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"
    LLM_BASED = "llm_based"

# PYDANTIC MODEL: kết quả phân loại
class ClassifyResult(BaseModel):
    label: Literal["chit-chat", "simple", "complex", "clarify"]
    confidence: float
    reason: str
    latency_ms: float = 0.0
    path: Literal["rule", "semantic_router", "llm"] = "rule"
    
# ABSTRACT BASE CLASS: BASE CLASSIFIER
class BaseClassifier(ABC):
    """
    Cách thức vận hành:
    - Subclass phải implement _classify_impl()
    - classify() tự động bọc thêm latency measurement
    Lưu ý:
    - Không gọi trực tiếp _classify_impl() từ bên ngoài
    """
    @abstractmethod
    def _classify_impl(self, query:str) -> ClassifyResult:
        pass # khai báo rỗng, chờ subclass ghi đè lên
    
    def classify(self, query: str) -> ClassifyResult:
        start_time = time.perf_counter()
        try:
            result = self._classify_impl(query)
        except Exception as e:
            logger.error(f"Lỗi phân loại query '{query[:50]}...': {e}")
            result = ClassifyResult(
                label="simple",
                confidence=0.3,
                reason=f"Error fallback: {str(e)[:50]}",
                path="rule",
            )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        result.latency_ms = round(elapsed_ms, 2)
        logger.info(
            f"[QueryClassifier] query='{query[:60]}...' "
            f"-> label={result.label}, confidence = {result.confidence:.2f},"
            f"path={result.path}, latency={result.latency_ms:.1f}ms"
        )
        return result
    
# FACTORY: Tạo CLASSIFIER theo MODE
class QueryClassifier:
    @staticmethod
    def create(mode: ClassifierMode = ClassifierMode.HYBRID) -> BaseClassifier:
        logger.info(f"Khởi tạo QueryClassifier mode = {mode.value}...")
        if mode == ClassifierMode.RULE_BASED:
            from .rule_based import RuleBasedClassifier
            return RuleBasedClassifier()
        elif mode == ClassifierMode.HYBRID:
            from .hybrid import HybridClassifier
            return HybridClassifier()
        elif mode == ClassifierMode.LLM_BASED:
            from .llm_based import LLMBasedClassifier
            return LLMBasedClassifier()
        else:
            raise ValueError(f"Mode không hợp lệ: {mode}")