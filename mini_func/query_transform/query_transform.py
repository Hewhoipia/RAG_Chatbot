# =============================================================================
# FILE: query_transform.py
# DESCRIPTION: Query Transformation Module cho RAG Pipeline with Demo Mode support
# =============================================================================

import logging
import re
from typing import List
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None
from config import config

logger = logging.getLogger(__name__)

def create_async_client() -> AsyncOpenAI:
    """
    Giới thiệu: Khởi tạo luồng mạng Component gọi mạng AsyncOpenAI cho vLLM server local.
    """
    if config.demo_mode:
        logger.info("[QueryTransform] running in DEMO MODE. Returning None client.")
        return None
    return AsyncOpenAI(
        base_url=config.vllm.base_url,
        api_key="EMPTY",
        timeout=30.0,
    )

async def multi_query_rewrite(
    client: AsyncOpenAI,
    query: str,
    n: int = 2,
    model_name: str = None,
) -> List[str]:
    """
    Giới thiệu: Hàm Rewrite diễn đạt tạo N mẫu biến thể (Rewrite variant) tương đương câu hỏi mẹ.
    """
    if config.demo_mode or client is None:
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["lãi suất", "tiết kiệm", "gửi"]):
            variants = [
                "Lãi suất tiền gửi Shinhan Bank hiện tại là bao nhiêu?",
                "Cập nhật bảng lãi suất gửi tiết kiệm mới nhất",
                "Quy định về lãi suất tiền gửi có kỳ hạn của Shinhan Bank"
            ]
        elif any(kw in query_lower for kw in ["thẻ", "card", "hạn mức"]):
            variants = [
                "Hạn mức và phí rút tiền thẻ tín dụng Shinhan?",
                "Thủ tục mở thẻ tín dụng tại ngân hàng Shinhan",
                "Quy chế sử dụng thẻ tín dụng Shinhan Bank mới nhất"
            ]
        elif any(kw in query_lower for kw in ["vay", "loan", "thế chấp"]):
            variants = [
                "Điều kiện và hồ sơ vay mua nhà tại Shinhan Bank?",
                "Quy trình giải ngân khoản vay thế chấp",
                "Lãi suất các gói vay tín chấp của Shinhan"
            ]
        else:
            variants = [
                f"Biến thể câu hỏi: {query}",
                f"Tra cứu thông tin liên quan đến: {query}",
                f"Tìm kiếm tài liệu về: {query}"
            ]
        return variants[:n]

    model = model_name or config.vllm.model_name
    prompt = f"""You are a search query optimizer for a banking knowledge base.

Given the original query below, rewrite it in {n} different way(s) that:
1. Preserve the EXACT intent and use the SAME language as the original query.
2. Use different wording, synonyms, or sentence structures
3. IMPORTANT: DO NOT translate, expand, or alter specific banking acronyms, document codes, product names, or technical terms (e.g., eKYC, B-Score, CIC, POS). Keep them exactly as they are.
4. Strip away conversational fillers (e.g., "cho hỏi", "dạ", "please tell me"). Focus strictly on core banking concepts (e.g., thẩm định, giải ngân, interest rate).

Original query: "{query}"

Return ONLY the rewritten queries, one per line. No numbering, no explanation."""

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = (response.choices[0].message.content or "").strip()
        
        lines = []
        for line in raw.split("\n"):
            line = line.strip()
            line = re.sub(r"^[\d]+[.)]\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)
            line = line.strip()
            if line and line != query:
                lines.append(line)
        
        result = lines[:n]
        logger.info(f"[QueryTransform] Rewrite query: {len(result)} variants")
        return result

    except Exception as e:
        logger.error(f"[QueryTransform] multi_query_rewrite error: {e}")
        return []

async def query_decomposition(
    client: AsyncOpenAI,
    query: str,
    n: int = 2,
    model_name: str = None,
) -> List[str]:
    """
    Giới thiệu: Khối phân phối phân rã cụm nhóm truy vấn phức tạp thành chuỗi câu con chia nhỏ.
    """
    if config.demo_mode or client is None:
        query_lower = query.lower()
        if "lãi suất" in query_lower and "thẻ" in query_lower:
            return ["Lãi suất tiền gửi Shinhan là bao nhiêu?", "Phí và hạn mức thẻ tín dụng Shinhan Bank?"][:n]
        return [f"Khía cạnh 1: {query}", f"Khía cạnh 2: {query}"][:n]

    model = model_name or config.vllm.model_name
    prompt = f"""You are a Query Analyst for a Banking RAG system.
Your task is to break down a complex query below into smaller, fully self-contained sub-queries to improve vector search retrieval.

MANDATORY RULE:
1. Each sub-query should target only ONE intent, action, entity comparison, or specific aspect that can be searched independently.
2. Use the SAME language as the original query.
3. IMPORTANT: DO NOT translate, expand, or alter specific banking acronyms, document codes, product names, or technical terms. Keep them exactly as they are.
4. Strip away conversational fillers. Focus strictly on core banking concepts.
5. Every sub-query MUST be fully understandable on its own. Replace all pronouns with the exact entity names from the original query.

Complex query: "{query}"

Return ONLY the sub-questions, one per line. No numbering, no explanation."""
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = (response.choices[0].message.content or "").strip()
        
        lines = []
        for line in raw.split("\n"):
            line = line.strip()
            line = re.sub(r"^[\d]+[.)]\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)
            line = line.strip()
            if line:
                lines.append(line)
        
        result = lines[:n]
        logger.info(f"[QueryTransform] Decompose query: {len(result)} sub-questions")
        return result

    except Exception as e:
        logger.error(f"[QueryTransform] query_decomposition error: {e}")
        return []

async def step_back_query(
    client: AsyncOpenAI,
    query: str,
    model_name: str = None,
) -> str:
    """
    Giới thiệu: Bóc xuất kỹ thuật tạo lệnh Step-Back Prompting.
    """
    if config.demo_mode or client is None:
        query_lower = query.lower()
        if "lãi suất" in query_lower:
            return "Chính sách lãi suất của Shinhan Bank Vietnam"
        if "thẻ" in query_lower:
            return "Quy định và điều khoản phát hành thẻ tín dụng Shinhan"
        return "Quy chế nghiệp vụ và thủ tục hoạt động ngân hàng"

    model = model_name or config.vllm.model_name
    prompt = f"""You are an expert Query Optimizer for a Banking RAG system.
Your task: Given a ORIGINAL QUERY below, generate ONE broader, more abstract question that captures the underlying concept, policy, or business rule.

MANDATORY RULE:
1. The step-back question MUST focus on retrieving general guidelines, product policies, or banking regulations related to the query.
2. Use the SAME language as the original query.
3. IMPORTANT: DO NOT translate, expand, or alter specific banking acronyms, document codes, product names, or technical terms. Keep them exactly as they are.
4. Strip away conversational fillers.

ORIGINAL QUERY: "{query}"

Step-back question (one line, same language as input, no markdown, no explanation):"""
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        result = (response.choices[0].message.content or "").strip()
        result = result.strip('"').strip("'").strip()
        logger.info(f"[QueryTransform] Step-back: {len(result)} stepback questions")
        return result

    except Exception as e:
        logger.error(f"[QueryTransform] step_back_query error: {e}")
        return ""

async def call_llm(
    client: AsyncOpenAI,
    prompt: str,
    model_name: str = None,
) -> str:
    """
    Giới thiệu: Cổng bọc wrapper ép gọi LLM Generator trực tiếp lùi xa luồng DB Search.
    """
    if config.demo_mode or client is None:
        prompt_lower = prompt.lower()
        if "clarify" in prompt_lower:
            return "Vui lòng cung cấp thêm thông tin chi tiết hoặc từ khóa cụ thể để trợ lý có thể hỗ trợ tra cứu chính xác nhất."
        return "Xin chào! Tôi có thể giúp gì cho bạn về các quy trình, thẻ tín dụng, lãi suất tiết kiệm hoặc nghiệp vụ ngân hàng Shinhan hôm nay?"

    model = model_name or config.vllm.model_name
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return (response.choices[0].message.content or "").strip()

    except Exception as e:
        logger.error(f"[QueryTransform] call_llm error: {e}")
        return "Xin lỗi, tôi đang gặp sự cố kỹ thuật. Vui lòng thử lại sau."
