# Giới thiệu:
#- Tier 3 Signals - Dự đoán Retrieval
#- Phân tích xem câu hỏi cần bao nhiêu nguồn tài liệu để trả lời
# và liệu câu hỏi có đủ rõ ràng để xử lý hay không

# Mục đích:
#1. Signal 3.1: Dự đoán số nguồn cần thiết (Based on entity/domain count)
#2. Signal 3.2: Mức độ tự tin của câu hỏi (confidence assessment)
#3. Signal 3.3: Phát hiện câu mơ hồ cấu trúc (structure ambiguity -> câu quá ngắn)
#4. Signal 3.4: Phát hiện câu mơ hồ ngữ nghĩa (semantic ambiguity -> từ khoá quá rộng)

# Cách sử dụng:
# from classifier.signals.retrieval_predict import RetrievalSignals
# signals = RetrievalSignals(query, domain_count =1, word_count = 5)
# print (signals.is_ambiguous) # True/False
# print (signals.ambiguity_reason) # str(lý do bị gán clarify)
# print (signals.predicted_sources) # int

# Lưu ý:
# - is_ambiguous = True -> trả về label "clarify" thay vì cố phân loại sai
# - predicted_sources >=2 -> tín hiệu "complex"

import re
from dataclasses import dataclass, field

BROAD_TERMS = {
    "lãi suất": [
        "vay", "cho vay", # continue
    ],
    "interest rate": [
        "loan", # continue
    ],
    # continue
}

# CONTEXT QUALIFIERS - nhận diện câu đã có context cụ thể
# Nếu câu chứa các pattern này, có nghĩa là người dùng đã chỉ định đối tượng cụ thể (loại A, nhóm 1, mã tài liệu, ...)
# -> không nên gán clarify
CONTEXT_QUALIFIERS = re.compile(
    r'(?:'
    r'loại\s*[a-z0-9&]|' # "loại A", "loại B", "loại 1"
    r'nhóm\s*\d|'       # "nhóm 1", "nhóm 2"
    # "cấp 1", "cấp 2"
    # "hạng A", "hạng B"
    # "ai trả", "ai chịu" (intent rõ)
    # "cho ai", "dành cho"
    # "mã ABC123"
    # "số 123"
    r')',
    re.IGNORECASE
)

@dataclass
class RetrievalSignals:
    """_summary_
    """
    query: str
    domain_count: int = 0
    word_count: int = 0
    
    # signal 3.1
    predicted_sources: int = field(init=False, default=1)
    
    #signal 3.3 + 3.4
    is_ambiguous: bool = field(init=False, default=False)
    
    # lý do bị gán ambiguous
    ambiguity_reason: str = field(init=False, default="")
    
    def __post_init__(self):
        """
        Giới thiệu: hàm chạy ngầm được kích hoạt tự động sau khi khởi tạo obj RetrievalSignals
        Mục đích: đánh giá nhanh số lượng nguồn cần truy vấn và rà soát các câu hỏi vô nghĩa
        Cách thức hđ:
        - Tính lượng document dự kiến dựa trên "domain_count
        - gọi hàm nhánh `_detect_ambiguity` phân loại câu cụt
        - gọi hàm nhánh `_detect_semantic_ambiguity` phân loại câu rộng
        
        Đầu vào & đầu ra:
        - Đầu vào: lấy dữ liệu domain_count
        - Đầu ra:
            + (None) cấp lại giá trị cho biến `predicted_sources` và `is_ambiguous`
        Lưu ý: chỉ số `predicted_sources` mang tính tham khảo dự đoán, k thay thế thuật toán top-k của Qdrant DB
        """
        # continue
        
    def _detect_ambiguity(self):
        """_summary_
        """
        # continue
        
    def _detect_semantic_ambiguity(self):
        """_summary_
        """
        # continue