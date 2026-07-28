"""Giới thiệu:
- Tier 3 Signals - Dự đoán Retrieval
- Phân tích xem câu hỏi cần bao nhiêu nguồn tài liệu để trả lời
và liệu câu hỏi có đủ rõ ràng để xử lý hay không

Mục đích:
1. Signal 3.1: Dự đoán số nguồn cần thiết (Based on entity/domain count)
2. Signal 3.2: Mức độ tự tin của câu hỏi (confidence assessment)
3. Signal 3.3: Phát hiện câu mơ hồ cấu trúc (structure ambiguity -> câu quá ngắn)
4. Signal 3.4: Phát hiện câu mơ hồ ngữ nghĩa (semantic ambiguity -> từ khoá quá rộng)

Cách sử dụng:
from classifier.signals.retrieval_predict import RetrievalSignals
signals = RetrievalSignals(query, domain_count =1, word_count = 5)
print (signals.is_ambiguous) # True/False
print (signals.ambiguity_reason) # str(lý do bị gán clarify)
print (signals.predicted_sources) # int

Lưu ý:
- is_ambiguous = True -> trả về label "clarify" thay vì cố phân loại sai
- predicted_sources >=2 -> tín hiệu "complex"
"""

import re
from dataclasses import dataclass, field

BROAD_TERMS = {
    "lãi suất": [
        "vay", "cho vay", "tiết kiệm", "tiền gửi"
    ],
    "interest rate": [
        "loan", "deposit", "saving"
    ],
    "phí": [
        "chuyển tiền", "thường niên", "rút tiền", "duy trì"
    ],
    "thẻ": [
        "tín dụng", "ghi nợ", "visa", "mastercard"
    ],
    "bảo hiểm": [
        "khoản vay", "nhân thọ", "sức khoẻ"
    ]
}

# CONTEXT QUALIFIERS - nhận diện câu đã có context cụ thể
# Nếu câu chứa các pattern này, có nghĩa là người dùng đã chỉ định đối tượng cụ thể (loại A, nhóm 1, mã tài liệu, ...)
# -> không nên gán clarify
CONTEXT_QUALIFIERS = re.compile(
    r'(?:'
    r'loại\s*[a-z0-9&]|' # "loại A", "loại B", "loại 1"
    r'nhóm\s*\d|'       # "nhóm 1", "nhóm 2"
    r'cấp\s*\d|'        # "cấp 1", "cấp 2"
    r'hạng\s*[a-z]|' # "hạng A", "hạng B"
    r'ai\s+trả|ai\s+chịu|' # "ai trả", "ai chịu" (intent rõ)
    r'cho\s+ai|dành\s+cho|' # "cho ai", "dành cho"
    r'mã\s*\w+|' # "mã ABC123"
    r'số\s*\d+' # "số 123"
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
        # 1. Dự đoán số lượng nguồn (domain_count)
        self.predicted_sources = max(1, self.domain_count)
        
        # 2. Câu quá ngắn (cụt lủn)
        self._detect_ambiguity()
        
        # 3. Câu quá rộng (semantic ambiguity)
        if not self.is_ambiguous:
            self._detect_semantic_ambiguity()
            
    def _detect_ambiguity(self):
        """Phát hiện cấu trúc câu quá ngắn (<= 2 từ)."""
        if self.word_count <= 2:
            self.is_ambiguous = True
            self.ambiguity_reason = "Query too short ({self.word_count} word(s)) (structure ambiguity)"
            
        # câu 2-3 từ -> kiểm tra WH-word và dấu hỏi, bỏ qua domain_count
        if self.word_count <= 3:
            query_lower = self.query.lower()
            has_clear_question = any (
                pattern in query_lower
                for pattern in [
                    "là gì", "bao nhiêu", "ở đâu", "khi nào", "what is", "how much",
                    "where", "when", "how to", "why", "who"
                ]
            ) or "?" in self.query
            if not has_clear_question:
                self.is_ambiguous = True
                self.ambiguity_reason = "Query too short ({self.word_count} word(s)) (structure ambiguity)"
            
    def _detect_semantic_ambiguity(self):
        """Phát hiện câu chứa từ khoá chung chung mà không có specifier."""
        if self.domain_count == 0:
            return
        
        query_lower = self.query.lower().strip()
        if CONTEXT_QUALIFIERS.search(query_lower):
            return
            
        for broad, specifiers in BROAD_TERMS.items():
            if broad in query_lower:
                if not any(spec in query_lower for spec in specifiers):
                    self.is_ambiguous = True
                    self.ambiguity_reason = f"Broad keyword '{broad}' without specific context (semantic ambiguity)"
                    break
        return