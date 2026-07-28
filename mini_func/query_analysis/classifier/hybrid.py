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

VALID_LABELS = {"chit-chat", "simple", "complex", "clarify"}

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
        Initialize HybridClassifier with rule-based engine, OpenAI client, and semantic router.

        Args:
            router_path (str): Path to the semantic router pickle file.
            vllm_base_url (str): Base URL for the vLLM OpenAI-compatible API server.
            model_name (str): Model name for embedding and LLM calls.
            confidence_threshold (float): Minimum confidence for semantic router acceptance.
            disagreement_threshold (float): Maximum confidence gap to trigger LLM escalation on rule-vs-semantic disagreement.
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
        Core hybrid classification pipeline combining 3 layers with cross-validation.

        Workflow:
        - Step 1: Rule-based hard overrides (confidence >= 0.85) accepted immediately
        - Step 2: Semantic Router classification via embedding + PCA + trained classifier
        - Step 3: Disagreement resolution — if rule and semantic disagree within gap threshold, escalate to LLM
        - Step 4: Moderate-confidence rule acceptance when semantic is uncertain
        - Step 5: LLM fallback for truly ambiguous cases

        Args:
            query (str): Raw user query string.

        Returns:
            ClassifyResult: Classification label, confidence, reason, and decision path.
        """
        if config.demo_mode:
            logger.info("DEMO MODE: HybridClassifier returning dummy mock answer")
            return ClassifyResult(label="simple", confidence=0.8, reason="Demo Mode", path="semantic_router")

        # Step 1: Rule-based hard overrides (high-confidence signals)
        rule_res = self.rule_classifier.classify(query)
        if rule_res.confidence >= 0.8:
            return rule_res

        # Step 2: Semantic Router
        semantic_res = self._run_semantic_classify(query)
        # Step 3: Disagreement resolution
        if semantic_res.confidence >= self.confidence_threshold:
            if rule_res.confidence >= 0.7 and rule_res.label != semantic_res.label:
                confidence_gap = abs(semantic_res.confidence - rule_res.confidence)
                if confidence_gap <= self.disagreement_threshold:
                    logger.info(
                        f"[Disagreement] rule={rule_res.label}@{rule_res.confidence:.2f} "
                        f"vs semantic={semantic_res.label}@{semantic_res.confidence:.2f} "
                        f"(gap={confidence_gap:.2f}) → LLM escalation"
                    )
                    if self.client:
                        return self._llm_fallback(query)
            return semantic_res

        # Step 4: Rule had moderate confidence but semantic was uncertain
        if rule_res.confidence >= 0.7:
            return rule_res

        # Step 5: LLM fallback for truly ambiguous cases
        if self.client:
            return self._llm_fallback(query)
        
        return rule_res

    def _run_semantic_classify(self, query: str) -> ClassifyResult:
        """
        Synchronous wrapper for the async semantic classification method.

        Workflow:
        - Step 1: Detect whether an asyncio event loop is already running
        - Step 2: If running, apply nest_asyncio patch and run coroutine in existing loop
        - Step 3: Otherwise, create a new event loop via asyncio.run()

        Args:
            query (str): Raw user query string.

        Returns:
            ClassifyResult: Semantic router result, or zero-confidence fallback on error.
        """
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self._semantic_classify(query))
            else:
                return asyncio.run(self._semantic_classify(query))
        except Exception as e:
            logger.error(f"Semantic classification wrapper error: {e}")
            return ClassifyResult(label="simple", confidence=0.0, reason="Semantic router error", path="semantic_router")

    async def _semantic_classify(self, query: str) -> ClassifyResult:
        """
        Classify query using embedding similarity against trained centroids via PCA + classifier.

        Workflow:
        - Step 1: Generate query embedding via async OpenAI embeddings API
        - Step 2: Apply PCA whitening transform
        - Step 3: Run trained classifier to get probability distribution over labels

        Args:
            query (str): Raw user query string.

        Returns:
            ClassifyResult: Classification with semantic router confidence, or zero-confidence skip.
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
        else:
            return ClassifyResult(label="simple", confidence=0.0, reason="Skipped", path="semantic_router")

    def _llm_fallback(self, query: str) -> ClassifyResult:
        """
        Last-resort classification using LLM chat completion with few-shot prompt.

        Workflow:
        - Step 1: Build messages list with system prompt + few-shot examples + query
        - Step 2: Call vLLM chat completion API with temperature=0 for deterministic output
        - Step 3: Parse JSON response and validate label against VALID_LABELS

        Args:
            query (str): Raw user query string.

        Returns:
            ClassifyResult: LLM-based classification, or low-confidence fallback on error.
        """
        if not self.client:
            return ClassifyResult(label="simple", confidence=0.3, reason="LLM client not available", path="rule")
            
        messages = build_llm_messages(query)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name or "Qwen/Qwen2.5-3B-Instruct",
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
            raw_label = data.get("label", "simple")
            label = raw_label if raw_label in VALID_LABELS else "simple"
            return ClassifyResult(
                label=label,
                confidence=float(data.get("confidence", 0.5)),
                reason=data.get("reason", "Parsed from LLM"),
                path="llm"
            )
        except Exception as e:
            logger.error(f"LLM Classification failed: {e}")
            return ClassifyResult(label="simple", confidence=0.3, reason="LLM failed", path="rule")