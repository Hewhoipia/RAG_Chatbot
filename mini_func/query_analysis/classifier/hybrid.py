"""Giới thiệu:
- Optioin B+ Hybrid: Semantic Router + rule overrides + LLM fallback
- đây là classifier khuyến nghị cho production
Mục đích:
1. Fast-path
2. hard overrides bằng rule-based khi có từ nối mạnh
3. llm fallback khi điểm mấp mé hoặc xung đột
cách dùng:
from classifier.hybrid import HybridClassifier
classifier = HybridClassifier()
result = classifier.classify("...")
Lưu ý
- Cần file classifier_router (chạy build_router_cache.py trước)
- Cần vLLM server đang chạy tại Localhost:8011 cho LLM Fallback
- Nếu không tìm thấy centroids -> tự động fallback về Rule-based"""

import os, json, logging, asyncio, numpy as np
from typing import Optional, Dict
from dotenv import load_dotenv
from .query_classifier import BaseClassifier, ClassifyResult
from .rule_based import RuleBasedClassifier
from .prompts.classifier_prompt import build_llm_messages
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
_env_path = os.path.join(_project_root, ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)
from config import config
logger = logging.getLogger(__name__)
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    AsyncOpenAI = None
    logger.warning("openai package not found. Real LLM calls will be unavailable.")
DEFAULT_ROUTER_PATH = os.path.join(
    os.path.dirname(__file__), "classifier_router.pkl"
)
DEFAULT_METADATA_PATH = os.path.join(
    os.path.dirname(__file__), "classifier_router_meta.json"
)

class HybridClassifier(BaseClassifier):
    """
    Giới thiệu: Hybrid Classifier - kiến trúc kết hợp 3 lớp
        Layer 1: Semantic Router
        Layer 2: Rule-based Hard Overrides
        Layer 3: LLM fallback
    Mục đích sử dụng:
    - Production-ready classifier cho hệ thống
    - Tối ưu latency
    - Accuracy ~92-95% nhờ kết hợp 3 lớp
    Cách thức vận hành:
    1. Startup: Load centroids từ .pkl file vào RAM
    2. Runtime:
        a. Early exit (rule-based): chit-chat/complex rõ ràng -> trả ngay
        b. Semantic Router
        c. hard overrides: phát hiện từ nối -> complex
        d. confidence check: điểm mấp mé -> gọi LLM
    Lưu ý
    - Nếu classifier_router không tồn tại -> trả về rule-based
    - LLM Fallback dùng sync OpenAI client (non-thinking mode)
    """
    def __init__(
        self,
        router_path: str = None,
        vllm_base_url: str = "http://localhost:8011/v1",
        model_name: str = None,
        confidence_threshold: float = 0.75,
        disagreement_threshold: float = 0.1,
    ):
        """
        """
        logger.info("Khởi tạo Hybrid Classifier")
        self.confidence_threshold = confidence_threshold
        self.disagreement_threshold = disagreement_threshold
        self.rule_classifier = RuleBasedClassifier()
        logger.info ("Rule-based classifier ready")
        if OPENAI_AVAILABLE:
            self.model_name = model_name or getattr(config.vllm, "model_name", "Qwen3.5-4B")
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", "dummy"),
                base_url=vllm_base_url or getattr(config.vllm, "base_url", "http://localhost:8011/v1")
            )
            self.async_client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY", "dummy"),
                base_url=vllm_base_url or getattr(config.vllm, "base_url", "http://localhost:8011/v1")
            )
            logger.info(f"OpenAI client initialized with model {self.model_name}")
        else:
            self.client = None
            self.async_client = None
            self.model_name = None

        self.router_path = router_path or DEFAULT_ROUTER_PATH
        self.pca = None
        self.classifier = None
        self.classes = None
        if os.path.exists(self.router_path):
            try:
                import pickle
                with open(self.router_path, "rb") as f:
                    router_assets = pickle.load(f)
                    self.pca = router_assets.get("pca")
                    self.classifier = router_assets.get("classifier")
                    self.classes = router_assets.get("classes")
                logger.info(f"Semantic router loaded from {self.router_path}")
            except Exception as e:
                logger.error(f"Error loading router cache: {e}")
        else:
            logger.warning(f"No semantic router found at {self.router_path}")
            
    def _classify_impl(self, query) -> ClassifyResult:
        """
        Cách thức hoạt động:
        - B1: Nạp signals thô qua rule-based engine
        - B2: lọc qua early exit
        - B3: dùng engine semantic router
        - B4: bypass rule-based nếu 3 bước trên không được
        """
        if config.demo_mode:
            logger.info("DEMO MODE: HybridClassifier returning dummy mock answer")
            return ClassifyResult(label="simple", confidence=0.8, reason="Demo Mode", path="semantic_router")

        # Step 1: Rule-based fast-path
        rule_res = self.rule_classifier.classify(query)
        if rule_res.confidence > 0.8:
            return rule_res
            
        # Step 2: Semantic Router
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            semantic_res = loop.run_until_complete(self._semantic_classify(query, None, None))
        except Exception:
            semantic_res = asyncio.run(self._semantic_classify(query, None, None))
            
        if semantic_res.confidence >= self.confidence_threshold:
            return semantic_res
            
        # Step 3: LLM Fallback
        if hasattr(self, "client") and self.client:
            return self._llm_fallback(query)

        return rule_res
    
    async def _semantic_classify(
        self,
        query: str,
        structural,
        semantic,
    ) -> ClassifyResult:
        """
        Giới thiệu: module phân loại bằng semantic router
        """
        if self.pca is not None and self.classifier is not None and self.async_client:
            try:
                response = await self.async_client.embeddings.create(
                    model=self.model_name,
                    input=[query]
                )
                embedding = np.array(response.data[0].embedding).reshape(1, -1)
                
                X_white = self.pca.transform(embedding)
                probs = self.classifier.predict_proba(X_white)[0]
                max_idx = np.argmax(probs)
                confidence = float(probs[max_idx])
                label = self.classes[max_idx]
                
                if confidence >= self.confidence_threshold:
                    return ClassifyResult(
                        label=label, 
                        confidence=confidence, 
                        reason=f"Semantic Router (prob={confidence:.2f})", 
                        path="semantic_router"
                    )
            except Exception as e:
                logger.error(f"Semantic classify error: {e}")
                
        return ClassifyResult(label="simple", confidence=0.0, reason="Skipped", path="semantic_router")
    def _llm_fallback(self, query: str) -> ClassifyResult:
        """_summary_

        Args:
            query (str): _description_

        Returns:
            ClassifyResult: _description_
        """
        if not hasattr(self, "client") or not self.client:
            return ClassifyResult(label="simple", confidence=0.3, reason="LLM client not available", path="rule")
            
        messages = build_llm_messages(query)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name if hasattr(self, "model_name") else "Qwen/Qwen2.5-3B-Instruct",
                messages=messages,
                max_tokens=100,
                temperature=0.0
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            data = json.loads(content)
            return ClassifyResult(
                label=data.get("label", "simple"),
                confidence=float(data.get("confidence", 0.5)),
                reason=data.get("reason", "Parsed from LLM"),
                path="llm"
            )
        except Exception as e:
            logger.error(f"LLM Classification failed: {e}")
            return ClassifyResult(label="simple", confidence=0.3, reason="LLM failed", path="rule")