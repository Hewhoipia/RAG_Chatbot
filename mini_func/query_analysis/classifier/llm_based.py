"""
llm_based.py

Giới thiệu:
- Option C: Phân loại câu hỏi hoàn toàn bằng LLM (vLLM).
Mục đích:
1. Đảm bảo độ chính xác cao nhất.
2. Dùng làm reference hoặc khi có GPU mạnh.
Cách sử dụng:
from classifier.llm_based import LLMBasedClassifier
classifier = LLMBasedClassifier()
result = classifier.classify("...")
"""
import os
import json
import logging
import asyncio
from typing import Optional
from .query_classifier import BaseClassifier, ClassifyResult
from .prompts.classifier_prompt import build_llm_messages
from config import config # import app config

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class LLMBasedClassifier(BaseClassifier):
    def __init__(self, vllm_base_url: str = "http://localhost:8011/v1", model_name: str = None):
        self.vllm_base_url = vllm_base_url
        self.model_name = model_name or "Qwen/Qwen2.5-3B-Instruct"
        if OPENAI_AVAILABLE and not config.demo_mode:
            self.client = OpenAI(base_url=self.vllm_base_url, api_key="EMPTY")
        else:
            self.client = None

    def _classify_impl(self, query: str) -> ClassifyResult:
        if config.demo_mode:
            logger.info("DEMO MODE: LLMBasedClassifier returning dummy mock answer")
            return ClassifyResult(label="simple", confidence=0.8, reason="Demo Mode", path="llm")
            
        if not self.client:
            logger.error("OpenAI client not available")
            return ClassifyResult(label="simple", confidence=0.3, reason="OpenAI unavailable", path="rule")

        messages = build_llm_messages(query)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
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
