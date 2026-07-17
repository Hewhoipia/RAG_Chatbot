import asyncio
from typing import List
from mini_func.auto_suggest.search_algorithm import AutoSuggestEngine
from mini_func.auto_suggest import faq_database
import logging
import re
import os
logger = logging.getLogger(__name__)

# DỮ LIỆU ĐƯỢC LOAD SẴN VÀO RAM KHI SERVER KHỞI ĐỘNG
# Trong thực tế, bạn có thể load list này từ Database khi chạy app
#FAQ_DATABASE = faq_database.FAQ_DATABASE
try:
    # Read the file and format it into a list of tuples
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'FAQ_DATABASE.txt')
    with open(file_path, 'r', encoding='utf-8') as file:
        # line.strip() removes the '\n' and any extra whitespace at the ends
        FAQ_DATABASE = [(line.strip(),) for line in file if line.strip()]
    engine = AutoSuggestEngine(FAQ_DATABASE)
except Exception as e:
    #logger.error (f"[AutoSuggest] engine cannot start")
    print ("[AutoSuggest] cannot start engine:", e)

# HÀM XỬ LÝ LOGIC TÌM KIẾM ĐỘC LẬP (Tối ưu tốc độ)
async def get_autocomplete_suggestions(query: str, limit: int = 5) -> List[str]:
    """
    Hàm xử lý logic tìm kiếm các câu hỏi chứa keyword.
    Sử dụng List Comprehension của Python để có tốc độ duyệt mảng C-level nhanh nhất.
    """
    try:
        if not query.strip():
            return []
        
        # Tìm tất cả các câu có chứa cụm từ người dùng đang gõ
        results = engine.search(query, limit)
        
        # Trả về tối đa 1-5 câu
        return results
    except Exception as e:
        #logger.error (f"[AutoSuggest] auto_suggest error: {e}")
        print ("[AutoSuggest] error at:", e)
        return []

async def run_test():
    search1 = await get_autocomplete_suggestions("mật")
    print(search1)
    search2 = await get_autocomplete_suggestions("lỗi")
    print(search2)

if __name__ == "__main__":
    # Instead of just calling run_test(), use asyncio.run()
    asyncio.run(run_test())