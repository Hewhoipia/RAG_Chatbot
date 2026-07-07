# =============================================================================
# FILE: query_transform.py
# DESCRIPTION: Query Transformation Module cho RAG Pipeline v7.2
# DATE: 2026-04-19
# VERSION: 1.0
# =============================================================================
#
# GIỚI THIỆU (Introduction):
# --------------------------
# Module cung cấp các hàm async để biến đổi query (rewriting, decomposition,
# step-back prompting) và gọi LLM trực tiếp cho chit-chat/clarify responses.
# Sử dụng AsyncOpenAI client để gọi vLLM server local.
#
# MỤC ĐÍCH (Purpose):
# -------------------
# 1. multi_query_rewrite(): Tạo N biến thể rewriting của query gốc
# 2. query_decomposition(): Phân rã query phức tạp thành sub-questions
# 3. step_back_query(): Tạo câu hỏi khái quát hơn (step-back prompting)
# 4. call_llm(): Gọi LLM trực tiếp cho chit-chat/clarify responses
#
# CÁCH SỬ DỤNG (Usage):
# ----------------------
# from mini_function.query_routing.query_transform import (
#     create_async_client, multi_query_rewrite, query_decomposition,
#     step_back_query, call_llm
# )
#
# client = create_async_client()
# rewrites = await multi_query_rewrite(client, "Lãi suất vay mua nhà?", n=2)
# decomps = await query_decomposition(client, "So sánh lãi suất vay...")
# step_back = await step_back_query(client, "Phí rút tiền ATM nước ngoài?")
#
# LƯU Ý (Notes):
# ---------------
# - Module viết mới thay thế advanced_query.query_transform đã bị mất
# - Sử dụng AsyncOpenAI (không phải sync OpenAI) cho non-blocking I/O
# - Temperature cao (0.7) cho rewriting → đa dạng, thấp (0.3) cho decomp → chính xác
# - Qwen3.5: enable_thinking=False cho tất cả transform calls (không cần reasoning)
# - Nếu LLM parse fail → trả về list rỗng (graceful degradation)
#
# THAM KHẢO:
# ----------
# - Implementation plan: mini_function/implementation_plan_20260419
# - Query routing draft: helpers/query_routing/query_routing.py
# =============================================================================

import logging
import re
from typing import List

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def create_async_client() -> AsyncOpenAI:
    """
    Giới thiệu: Khởi tạo luồng mạng Component gọi mạng AsyncOpenAI cho vLLM server local.
    
    Mục đích sử dụng: Thiết lập bộ Session làm Hub chung giao tiếp (Không liên quan nhúng mạng Embedding Server).
    
    Cách thức hoạt động:
    - Kéo Load object config model base endpoint và tạo Instance khởi động mạng.
    
    Đầu vào - Đầu ra của hàm đó:
    - Đầu vào: Không có.
    - Đầu ra:
        + (AsyncOpenAI): Kết nối mượt qua proxy client.
    
    Lưu ý: Parameter gốc ép `api_key="EMPTY"` tĩnh do vLLM open endpoint offline không kiểm duyệt key mã hóa.
    """
    from config_v7 import config
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
    
    Mục đích sử dụng: Đóng làm Router mở rộng phạm vi Search giúp bắt rộng nhiều Chunk liên quan bằng cách đa dạng hóa biểu đạt cho mảng nhúng Semantic Vector Hybrid.
    
    Cách thức hoạt động:
    - Truyền tham số Nhiệt cao thiết đặt (temperature=0.7) giúp nặn mẫu list Prompt sinh ra N mảng các câu đồng nghĩa cùng ngôn ngữ nhưng đa dạng diễn tả.
    - Phân tích cú pháp Parser: Ép RegEx gạch rập rạch mảng theo ngắt lề (\n) và xóa triệt để ký tự Numbering ngắt đánh số tự do.
    
    Đầu vào - Đầu ra của hàm đó:
    - Đầu vào:
        + `client` (AsyncOpenAI): Cụm Connection API móc vLLM Client.
        + `query` (str): Chuỗi dòng input hỏi.
        + `n` (int): Cân lượng số câu rút.
        + `model_name` (str): Model override config.
    - Đầu ra:
        + (List[str]): Hộp line text array dòng mảng nội dung sạch rập.
        
    Lưu ý: Bắt buộc tắt boolean `enable_thinking=False` để mô hình tạo List mảng nhỏ không sinh rườm rà ra thẻ Think. Quá trình fail mạng trả Array rỗng Bypass chạy tiếp.
    """
    from config_v7 import config
    model = model_name or config.vllm.model_name

    prompt = f"""You are a search query optimizer for a banking knowledge base.

Given the original query below, rewrite it in {n} different way(s) that:
1. Preserve the EXACT intent and use the SAME language as the original query.
2. Use different wording, synonyms, or sentence structures
3. IMPORTANT: DO NOT translate, expand, or alter specific banking acronyms, document codes, product names, or technical terms (e.g., eKYC, B-Score, CIC, POS). Keep them exactly as they are.
4. Strip away conversational fillers (e.g., "cho hỏi", "dạ", "please tell me"). Focus strictly on core banking concepts (e.g., thẩm định, giải ngân, interest rate).
5. Provide a mix of:
   - Keyword-dense versions (short, focused on main entities).
   - Natural phrasing variations (using domain-specific synonyms).

Original query: "{query}"

Return ONLY the rewritten queries, one per line. No numbering, no explanation."""

    """
    PROMPT CHANGE:
    - add RULE:
        3, 4 (xử lí ngôn ngữ, không dịch các thuật ngữ phứt tạp, bỏ các câu xã giao)
        5 (đưa ra thêm tuỳ chọn để rewrite)
    
    Advanced prompt recommendation:
    1. Output nên trả về JSON hoặc JSON array như:
    ["variation 1", "variation 2", "variation 3"]
    2. Nên có thêm Few-shot example (nếu cần)
    """

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = (response.choices[0].message.content or "").strip()
        
        # Parse: mỗi dòng là 1 rewrite, loại bỏ dòng trống và numbering
        lines = []
        for line in raw.split("\n"):
            line = line.strip()
            # Bỏ numbering patterns: "1.", "1)", "- ", "* "
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
    Giới thiệu: Khối phân phối phân rã cụm nhóm truy vấn phức tạp (Macro target) thành chuỗi câu con chia nhỏ (Micro list sub-questions).
    
    Mục đích sử dụng: Tập trung từng phần text tách tháo gỡ khía cạnh khơi nhỏ cho câu Compare Multi-topic (So sánh). Giúp việc lọt lọc hit Vector Retriever RAG trọn vẹn hơn.
    
    Cách thức hoạt động:
    - Đưa string Template ép nhiệt độ kìm thấp (0.3) cho LLM để đảm bảo bóc tách không chém gió sáng tạo sai đề yêu cầu. Xỏ Parse RegEx mảng qua Filter ngắt đoạn và đánh vảy gạch đếm đầu dòng.
    
    Đầu vào - Đầu ra của hàm đó:
    - Đầu vào: Dải cấu trúc bộ Parameter Call tương tự hàm Rewriting (`client`, `query`, `n`).
    - Đầu ra:
        + (List[str]): Nhóm List array nhỏ các dòng câu mớm RAG bẻ gãy.
        
    Lưu ý: Parameter Decomposition macro nên gọi gút chạy lúc gánh Label mang 'complex'. Max array return = N mốc Items lặp giới hạn.
    """
    from config_v7 import config
    model = model_name or config.vllm.model_name

    prompt = f"""You are a Query Analyst for a Banking RAG system.
Your task is to break down a complex query below into smaller, fully self-contained sub-queries to improve vector search retrieval.

MANDATORY RULE:
1. Each sub-query should target only ONE intent, action, entity comparison, or specific aspect that can be searched independently.
2. Use the SAME language as the original query.
3. IMPORTANT: DO NOT translate, expand, or alter specific banking acronyms, document codes, product names, or technical terms (e.g., eKYC, B-Score, CIC, POS). Keep them exactly as they are.
4. Strip away conversational fillers (e.g., "cho hỏi", "dạ", "please tell me"). Focus strictly on core banking concepts (e.g., thẩm định, giải ngân, interest rate).
5. Every sub-query MUST be fully understandable on its own. Replace all pronouns (e.g., "it", "they", "thẻ đó", "chúng") with the exact entity names from the original query.
6. Generate ONLY as many sub-queries as necessary based on the logical divisions in the original text. Do not invent extra queries.

Complex query: "{query}"

Return ONLY the sub-questions, one per line. No numbering, no explanation."""

    """
    PROMPT CHANGE:
    - Add RULE:
        1 (sub-query chỉ tập trung vào 1 ý duy nhất)
        2, 3, 4 (xử lí ngôn ngữ và các câu hỏi xã giao)
        5 giả sử query "hạn mức của Signature và Platinum, lãi suất của chúng là gì" -> llm dễ tách thành "hạn mức của Signature", "hạn mức của Platinum", "lãi suất của chúng" -> bỏ dô vector search k tìm ra "chúng" là gì
        6. xác định số lượng sub-query dựa trên query gốc chứ không set cứng n = 2
        
    Advanced prompt recommendation:
    1. Output nên trả về JSON hoặc JSON array như:
    ["sub 1", "sub 2", "sub 3"]
    2. Nên có thêm FEW-SHOT EXAMPLES:
Original: "Cho mình hỏi lãi suất của thẻ Shinhan Platinum và điều kiện mở thẻ đó là gì?"
Output: ["Lãi suất của thẻ Shinhan Platinum là bao nhiêu?", "Điều kiện mở thẻ Shinhan Platinum là gì?"]
    """
    
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
    Giới thiệu: Bóc xuất kỹ thuật tạo lệnh Step-Back Prompting (Truy vấn trừu tượng quy quát).
    
    Mục đích sử dụng: Dập nguyên mẫu của tổ chức Google DeepMind giúp bắt Background broad context RAG chung để mở rộng Range hit Retrieval thay vì focus điểm ảnh quá hẹp.
    
    Cách thức hoạt động:
    - Nạp nguyên Query user input, rọi qua Template LLM sinh câu Big Picture 1 câu một tấc. Luồng chặn thinking tắt, khóa nhiệt Temperature tĩnh 0.3 kìm chế. Loại tỉa dọn rác dính String Quotes wrap text.
    
    Đầu vào - Đầu ra của hàm đó:
    - Đầu vào: Nhóm tham số base vLLM Call (`client`, `query`).
    - Đầu ra:
        + (str): Form Object text duy nhất 1 chuỗi String trả lời.
        
    Lưu ý: Output block return là cụm text trơn chuỗi không array, nếu gặp tình thế Crash connection vLLM rớt mạng trả Null rỗng vượt vòng Bypass nhẹ hều.
    """
    from config_v7 import config
    model = model_name or config.vllm.model_name

    prompt = f"""You are an expert Query Optimizer for a Banking RAG system.
Your task: Given a ORIGINAL QUERY below, generate ONE broader, more abstract question that captures the underlying concept, policy, or business rule.

MANDATORY RULE:
1. The step-back question MUST focus on retrieving general guidelines, product policies, or banking regulations related to the query.
2. Use the SAME language as the original query.
3. IMPORTANT: DO NOT translate, expand, or alter specific banking acronyms, document codes, product names, or technical terms (e.g., eKYC, B-Score, CIC, POS). Keep them exactly as they are.
4. Strip away conversational fillers (e.g., "cho hỏi", "dạ", "please tell me"). Focus strictly on core banking concepts (e.g., thẩm định, giải ngân, interest rate).
5. Do NOT make it too broad. Keep the core banking domain/product in the query (e.g., if the user asks about a specific credit card's late fee, ask about credit card late fee policies in general, NOT "what is a fee?").

ORIGINAL QUERY: "{query}"

Step-back question (one line, same language as input, no markdown, no explanation):"""

    """
    PROMT CHANGE:
    - add RULE:
        1 (câu hỏi step-back nên tập trung vào khía cạnh nghiệp vụ của query gốc như guidelines, policies, ...)
        2, 3, 4 (xác định ngôn ngữ, không dịch một vài thuật ngữ phức tạp, bỏ qua các câu từ xã giao mà tập trung vào ý chính)
        5 (không step-back quá rộng, chỉ dừng ở mức core banking domain)
    
    Advanced prompt recommendation:
    1. Output nên trả về JSON như:
{
  "step_back_query": "<string: the generalized question>"
}
    2. Nên thêm vào FEW-SHOT EXAMPLES:
Original: "Thẻ Shinhan Visa Platinum của mình quẹt lố hạn mức 2 triệu thì phí phạt tính thế nào?"
{"step_back_query": "Quy định và cách tính phí phạt vượt hạn mức đối với thẻ tín dụng là gì?"}

Original: "Tôi muốn rút hồ sơ vay mua nhà dự án Vinhomes trước hạn thì thủ tục ra sao?"
{"step_back_query": "Quy trình, điều kiện và thủ tục tất toán trước hạn đối với sản phẩm khoản vay thế chấp bất động sản?"}

Original: "How to reset my Shinhan SOL mobile banking password if I locked it?"
{"step_back_query": "What is the general procedure to recover or reset an account for the mobile banking application?"}
    """
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        result = (response.choices[0].message.content or "").strip()
        # Loại bỏ quotes nếu LLM wrap trong quotes
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
    Giới thiệu: Cổng bọc wrapper ép gọi LLM Generator trực tiếp lùi xa luồng DB Search vòng vèo rườm rà.
    
    Mục đích sử dụng: Tuyến gọi Call nhanh chớp nháy (Direct Endpoint) cho nhóm Label Chat tán dóc (Chit-chat) hoặc Vòi thông tin (Clarify). Tạo câu kết ngắn xả về Frontend mượt mà.
    
    Cách thức hoạt động: Cấu hình block API Open AI chốt max short token (500 chữ giới hạn), set Temp bay 0.7 và rạch rạch Force ngắt cờ suy nghĩ (Thinking = False). Return answer mộc tựa String.
    
    Đầu vào - Đầu ra của hàm đó:
    - Đầu vào: Nắm Call bộ Parameter Base. Truyền lệnh text qua Prompt đúc tĩnh.
    - Đầu ra:
        + (str): Chuỗi đoạn text kết xuất thô ngắn.
        
    Lưu ý: Dịch vụ Exception chèn sẵn đệm câu xin lỗi bản xứ dập tiếng Việt (Graceful Degradation Error) dự phòng rớt node server Async.
    """
    from config_v7 import config
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
