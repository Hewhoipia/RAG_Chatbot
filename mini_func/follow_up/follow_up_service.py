"""
Description: generate smart follow-up questions
"""
from typing import List

async def generate_smart_follow_ups(query: str, answer: str) -> List[str]:
    # Returns some static smart follow-up suggestions based on terms
    query_lower = query.lower()
    if "lãi suất" in query_lower or "deposit" in query_lower or "tiết kiệm" in query_lower or "gửi" in query_lower:
        return [
            "Lãi suất kỳ hạn 6 tháng là bao nhiêu?",
            "Thủ tục rút tiền trước hạn như thế nào?",
            "Có thể mở tài khoản tiết kiệm online không?"
        ]
    elif "thẻ" in query_lower or "card" in query_lower or "hạn mức" in query_lower:
        return [
            "Làm thế nào để tăng hạn mức thẻ tín dụng?",
            "Phí thường niên các loại thẻ là bao nhiêu?",
            "Quy trình báo mất thẻ khẩn cấp?"
        ]
    elif "vay" in query_lower or "loan" in query_lower or "thế chấp" in query_lower:
        return [
            "Vay mua xe cần những thủ tục gì?",
            "Cách tính lãi suất vay trả góp?",
            "Thời hạn vay tối đa là bao lâu?"
        ]
    return [
        "Xem hướng dẫn thủ tục mở thẻ tín dụng",
        "Tra cứu lãi suất tiền gửi tiết kiệm mới nhất",
        "Liên hệ hỗ trợ kỹ thuật IT"
    ]
