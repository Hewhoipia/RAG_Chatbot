# 14 mảng nghiệp vụ bank
# mỗi key là tên mảng, value là danh sách từ khoá đặc trưng (lowercase)
# cấu trúc: [tiếng việt] -> [Tiếng Anh/thuật ngữ quốc tế]
BANKING_DOMAINS = {
    "bancassurance": [
        # tiếng việt
        "bảo hiểm",
        # continue
        
        # tiếng anh
        "bancassurance",
        # continue
        
    ],
    "capital_market": [
        # continue
    ],
    "card": [
        # continue
    ],
    "customer_service": [
        # continue
    ],
    "deposit":[
        # continue
    ],
    "digital_marketing":[
        # continue
    ],
    "finance_accounting": [
        # continue
    ],
    "hr_general_affair":[
        # continue
    ],
    "internal_control":[
        # continue
    ],
    "it": [
        # continue
    ],
    "loan": [
        # continue
    ],
    "organization_management": [
        # continue
    ],
    "remittance": [
        # continue
    ],
    "trade_finance":[
        # continue
    ],
    "legal_property":[
        # continue
    ],
}

BANKING_ACTIONS = [
    "tra cứu", "kiểm tra",
    # continue
]

ALL_BANKING_TERMS = set()
for domain_terms in BANKING_DOMAINS.values():
    ALL_BANKING_TERMS.update(domain_terms)
ALL_BANKING_TERMS.update(BANKING_ACTIONS)

# Word-boundary matching - giải quyết lỗi false-positive cho từ ngắn
# vấn đề: substring matching đơn giản ('kw in query') gây false-poitive:
#   - "ai" (IT) match trong "tổng thống Mỹ là AI?"
#   - "nim" (finance) match trong "minimum balance"
import re
SHORT_KW_THRESHOLD = 4
_short_kw_patterns = {}
for kw in ALL_BANKING_TERMS:
    if len(kw) <= SHORT_KW_THRESHOLD:
        _short_kw_patterns[kw] = re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE)
def _keyword_in_query(kw: str, query_lower:str) -> bool:
    """_summary_

    Args:
        kw (str): _description_
        query_lower (str): _description_

    Returns:
        bool: _description_
    """
    if len(kw) <= SHORT_KW_THRESHOLD:
        pattern = _short_kw_patterns.get(kw)
        if pattern:
            return bool(pattern.search(query_lower))
        return False
    else:
        return kw in query_lower
    
def count_domain_matches(query_lower: str) -> dict:
    """_summary_

    Args:
        query_lower (str): _description_

    Returns:
        dict: _description_
    """
    matched = {}
    for domain, kws in BANKING_DOMAINS.items():
        for kw in kws:
            if _keyword_in_query(kw, query_lower):
                matched[domain] = matched.get(domain, 0) + 1
    return matched