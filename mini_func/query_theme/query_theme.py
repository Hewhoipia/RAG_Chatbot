# =============================================================================
# FILE: query_theme.py
# DESCRIPTION: Xác định banking categories từ câu query của người dùng
# DATE: 2026-05-11
# =============================================================================

import asyncio
import json
import re
from typing import List

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# 14 Banking Categories
# ---------------------------------------------------------------------------
CATEGORIES = [
    "Bancassurance",
    "Capital Market",
    "Card",
    "Customer Service",
    "Deposit",
    "Digital Banking",
    "Finance & Accounting",
    "HR & GA",
    "Internal Control",
    "IT",
    "Loan",
    "Organization & Management",
    "Remittance",
    "Trade Finance",
]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_CATEGORY_LIST = "\n".join(f"- {c}" for c in CATEGORIES)  # pre-compute trước

_SYSTEM_PROMPT = f"""\
You are a banking domain classifier. Given a user query, identify ALL relevant categories from the list below.

CATEGORIES: {_CATEGORY_LIST}

MANDATORY RULE:
1. A query can belong to 1, multiple, or 0 categories.
2. If the query spans multiple domains (e.g., "Paying a loan via credit card"), include ALL relevant categories.
3. The query may be in multiple languages. Apply the same classification logic.
4. Return ONLY a valid JSON array of matched category strings, e.g. ["Card", "Loan"]
5. Use the EXACT category names from the CATEGORIES list above
6. Return [] if no category matches
7. Do NOT add explanation or extra text

Output STRICTLY a JSON array with this schema (no markdown formatting, no explanation outside the JSON):
[<array of strings: exact category names or empty>]
"""

_USER_TEMPLATE = 'Query: "{query}"\n\nMatched categories (JSON array only):'

'''
PROMPT CHANGE:
- thêm RULE:
    1, 2 (giúp llm phân loại multiple domains trong 1 query tốt hơn)
    3 (một query có thể nhiều ngôn ngữ, áp dụng cách phân loại như nhau)
- thêm format cho output

Advanced Prompt Recommendation:
1. Đề xuất output có thêm phần reasoning, sau đó test xem độ chính xác có được cải thiện không, format JSON như sau:
{
  "reasoning": "<string: brief 1-2 sentence analysis>",
  "categories": [<array of strings: exact category names or empty>]
}
Lưu ý: cần viết lại back-end chỉ lấy phần JSON array của "categories"

2. Nên bổ sung thêm FEW-SHOT EXAMPLES:
[
    {
        "user" : "Lãi suất thẻ tín dụng và thủ tục vay thế chấp?",
        "classifier": "{"reasoning": "Query asks about credit card interest (Card) and mortgage procedures (Loan).", "categories": ["Card", "Loan"]}"
    },

    {
        "user": "SOL App từ chối cho người dùng đăng ký Internet Banking.",
        "classifier": "{"reasoning": "Query is about a technical issue logging into the mobile application.", "categories": ["Digital Banking", "IT"]}"
    },

    {
        "user": "Tổng thống Hàn Quốc là ai?",
        "classifier": "{"reasoning": "Query is out of domain and unrelated to any banking categories.", "categories": []}"
    },
]
'''

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
async def query_theme(query: str, model_name: str = "Qwen3.5-4B") -> List[str]:
    """Trả về list các banking categories liên quan đến câu query."""
    client = AsyncOpenAI(base_url="http://localhost:8011/v1", api_key="EMPTY", timeout=30.0)

    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _USER_TEMPLATE.format(query=query)},
        ],
        max_tokens=100,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    # Lấy nội dung LLM trả về, có thể là None
    raw = (response.choices[0].message.content or "").strip()

    # Parse JSON array từ response
    try:
        # Cắt lấy phần [...] trong text LLM trả về, bỏ qua text thừa xung quanh
        # raw = 'Sure! Here are: ["Card", "Loan"]'  →  match = '["Card", "Loan"]'
        # raw = '["Card", "Loan"]'                  →  match = '["Card", "Loan"]'
        match = re.search(r"\[.*?\]", raw, re.DOTALL)

        # match là Match object (nếu tìm thấy), hoặc None (nếu không tìm thấy)
        # match = <re.Match object; span=(17, 33), match='["Card", "Loan"]'>
        #   └─ .group() → lấy đúng chuỗi đã khớp: '["Card", "Loan"]'
        # match = None → bỏ qua .group(), dùng [] thay thế (tránh AttributeError)
        # json.loads('["Card", "Loan"]') → Python list: ["Card", "Loan"]
        result = json.loads(match.group()) if match else []

        # Tạo dict lookup {tên_thường: tên_chuẩn} để so sánh case-insensitive
        # CATEGORIES = ["Card", "Loan", ...]
        # valid = {"card": "Card", "loan": "Loan", ...}
        valid = {c.lower(): c for c in CATEGORIES}

        # Lọc chỉ giữ item hợp lệ + trả về đúng tên chuẩn trong CATEGORIES
        # result = ["card", "Loans", "Banking"]
        # "card"    → "card"    in valid → True  → lấy valid["card"]    = "Card"  
        # "Loans"   → "loans"   in valid → False → bỏ qua              
        # "Banking" → "banking" in valid → False → bỏ qua           
        # → ["Card"]
        return [valid[r.lower()] for r in result if r.lower() in valid]

    except Exception:
        return []


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
async def test():
    cases = [
        "What is the different of interest rate between credit card and loan?",
        "How can I open installment deposit account?",
        "What is the AML process?",
        "Tôi muốn chuyển tiền quốc tế, phí là bao nhiêu?",
        "Quy trình tuyển dụng nhân viên mới như thế nào?",
        "Xin chào, hôm nay trời đẹp nhỉ?",  # ngoài domain → []
    ]

    print("=" * 60)
    for q in cases:
        result = await query_theme(q)
        print(f"In:  {q}")
        print(f"Out: {result}")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(test())
