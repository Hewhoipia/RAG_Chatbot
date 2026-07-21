import asyncio
import logging
import os
from typing import List
from mini_func.auto_suggest.search_algorithm import AutoSuggestEngine

logger = logging.getLogger(__name__)

# Dữ liệu được load sẵn vào ram khi server khởi động
# Trong thực tế hoàn toàn có thể load list FAQ từ Database khi chạy
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'FAQ_DATABASE.txt')
    with open(file_path, 'r', encoding='utf-8') as file:
        FAQ_DATABASE = [(line.strip(),) for line in file if line.strip()]
    # tạo engine
    engine = AutoSuggestEngine(FAQ_DATABASE)
except Exception as e:
    logger.error(f"[AutoSuggest] cannot start engine:", e)

# Hàm xử lý logic tìm kiếm
async def get_autocomplete_suggestions(query: str, limit: int = 5) -> List[str]:
    """
    Hàm xử lý logic tìm kiếm các câu hỏi chứa keyword hoặc chứa ký tự liên quan
    Đầu vào: query: từ front-end -> fastapi -> here
            limit: số lượng kết quả hiển thị trong menu dropdown, mặc định 5
    Đầu ra: kết quả search với số lượng 'limit'
    """
    try:
        if not query.strip():
            return []
        # gọi hàm tìm kiếm
        results = engine.search(query, limit)
        return results
    except Exception as e:
        logger.error (f"[AutoSuggest] auto_suggest error: {e}")
        return []

async def run_test():
    search1 = await get_autocomplete_suggestions("e mai l")
    print(f"Kết quả search: {search1}")
    search2 = await get_autocomplete_suggestions("a tm")
    print(f"Kết quả search: {search2}")
if __name__ == "__main__":
    asyncio.run(run_test())