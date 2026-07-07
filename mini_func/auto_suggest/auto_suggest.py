import asyncio
from typing import List
from mini_func.auto_suggest.search_algorithm import WordSuffixTrie
import logging
import re
logger = logging.getLogger(__name__)

# 1. DỮ LIỆU ĐƯỢC LOAD SẴN VÀO RAM KHI SERVER KHỞI ĐỘNG
# Trong thực tế, bạn có thể load list này từ Database khi chạy app
FAQ_DATABASE = [
    "Cách đổi mật khẩu VPN",
    "Lỗi màn hình xanh 0x0000",
    "Không kết nối được wifi công ty",
    "Cài đặt máy in tầng 3",
    "Xin cấp quyền Jira",
    "Quên mật khẩu email Outlook",
    "Cấu hình Cisco AnyConnect",
    "Lỗi VPN the client agent was unable",
    "Màn hình bị nháy liên tục",
    "ATM là gì",
    "mở thẻ tín dụng cần xem xét gì",
    "kết nối mạng cục bộ như thế nào",
    "quy trình đăng ký ekyc cho KH cá nhân",
    "thay đổi thông tin người đại diện pháp luật của DN",
    "check CIF trên màn hình aither như thế nào",
    "các sản phẩm dành cho hộ kinh doanh",
    "các sản phẩm POS cho HKD và DN",
    "quy định về POST thông tin lên mxh",
    ""
]

def create_method_client() -> WordSuffixTrie:
    trie = WordSuffixTrie()
    # Chèn dữ liệu
    for idx, faq in enumerate(FAQ_DATABASE):
        trie.insert(faq, idx)
    return trie

# HÀM XỬ LÝ LOGIC TÌM KIẾM ĐỘC LẬP (Tối ưu tốc độ)
async def get_autocomplete_suggestions(query: str, limit: int = 5) -> List[str]:
    """
    Hàm tìm kiếm các câu hỏi chứa keyword.
    Sử dụng List Comprehension của Python để có tốc độ duyệt mảng C-level nhanh nhất.
    """
    try:
        trie = create_method_client()
        if not query.strip():
            return []
        
        # Tìm tất cả các câu có chứa cụm từ người dùng đang gõ
        results = trie.search(query, limit)
        
        # Trả về tối đa 1-5 câu
        return results
    except Exception as e:
        logger.error (f"[AutoSuggest] auto_suggest error: {e}")
        return []

async def run_test():
    search1 = await get_autocomplete_suggestions("mật")
    print(search1)
    search2 = await get_autocomplete_suggestions("lỗi")
    print(search2)

if __name__ == "__main__":
    # Instead of just calling run_test(), use asyncio.run()
    asyncio.run(run_test())