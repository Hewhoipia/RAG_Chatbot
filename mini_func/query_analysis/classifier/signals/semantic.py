# Giới thiệu:
#- Tier 2 Signals - phân tích ngữ nghĩa (semantic Analysis)
#- Kết hợp Intent taxonomy, domain coverage, và temporal signals

# Mục đích:
#1. Signal 2.1: phân loại Intent Type (Factoid, definition, comparison, ...)
#2. Signal 2.1+: phát hiện disguised intent (bề ngoài simple bản chất complex)
#3. Signal 2.2: đếm domain coverage (single domain -> simple, multi -> complex)
#4. Signal 2.3: temporal signals (thay đổi theo thời gian -> complex)

# Cách sử dụng:
# from classifier.signals.semantic import SemanticSignals
# signals = SemanticSignals(query)
# print (signals.intent_type) # "factoid" | ...
# print (signals.domain_count) # int

# Lưu ý:
# - Sử dụng domain_banking.py để đếm domain
# - Intent Taxonomy dựa trên Query analysis.txt (8 loại intent)

from dataclasses import dataclass, field
from ..keywords.domain_banking import count_domain_matches
from .structural import(
    COMPARISON_CONNECTIVES,
    CAUSAL_CONNECTIVES,
    CONDITIONAL_CONNECTIVES,
    PROCEDURAL_PATTERNS,
    AGGREGATION_PATTERNS,
    RECOMMENDATION_PATTERNS,
    TEMPORAL_PATTERNS,
    YES_NO_PATTERNS,
    WH_WORDS_SIMPLE,
    DISGUISED_INTENT,
)

@dataclass
class SemanticSignals:
    """_summary_
    """
    query: str
    query_lower: str = field(init=False)
    
    intent_type: str = field(init=False, default="unknown")
    is_complex_intent: bool = field(init=False, default=False)
    is_simple_intent: bool = field(init=False, default=False)

    has_disguised_intent: bool = field(init=False, default=False)

    domain_count: int = field(init=False, default=0)
    matched_domains: dict = field(init=False, default_factory=dict)
    has_banking_terms: bool = field(init=False, default=False)

    has_temporal: bool = field(init=False, default=False)

    def __post_init__(self):
        """
        Giới thiệu: chạy ngầm được kích hoạt tự động ngay sau tạo obj SemanticSignals
        Mục đích: xử lý và tính toán 4 mảng nhận diện tín hiệu ngữ nghĩa (Intent type, Disguised Intent, Domain Coverage, Temporal)
        Cách thức:
        - Chuẩn hoá chuỗi Inbound thành chữ thường
        - Gọi _detect_intent_type phân loại ý đồ
        - Khảo sát biến ngầm (disguised intent)
        - Gọi thư viện rà quét từ khoá nghiệp vụ đếm số lượng Domain ngân hàng liên quan tới
        - Chặn bắt tín hiệu đổi thời gian
        Đầu vào - đầu ra:
        - Đầu vào: dùng attributes của obj
        - Đầu ra: None
        Lưu ý: Domain coverage là chỉ số đánh giá độ rộng thông tin. càng nhiều thì câu càng complex.
        """
        self.query_lower = self.query.lower().strip()
        self._detect_intent_type()
        self.has_disguised_intent = bool(DISGUISED_INTENT.search(self.query_lower))
        self.matched_domains = count_domain_matches(self.query_lower)
        self.domain_count = len(self.matched_domains)
        self.has_banking_terms = self.domain_count > 0
        self.has_temporal = bool(TEMPORAL_PATTERNS.search(self.query_lower))

    def _detect_intent_type(self):
        """_summary_
        """
        q = self.query_lower
        
        # kiểm tra complex intents trước
        if COMPARISON_CONNECTIVES.search(q):
            self.intent_type = "comparison"
            self.is_complex_intent = True
            return
        if CAUSAL_CONNECTIVES.search(q):
            self.intent_type = "causal"
            self.is_complex_intent = True
            return
        if CONDITIONAL_CONNECTIVES.search(q):
            self.intent_type = "multi_hop"
            self.is_complex_intent = True
            return
        if PROCEDURAL_PATTERNS.search(q):
            self.intent_type = "procedure"
            self.is_complex_intent = True
            return
        if AGGREGATION_PATTERNS.search(q):
            self.intent_type = "aggregation"
            self.is_complex_intent = True
            return
        if RECOMMENDATION_PATTERNS.search(q):
            self.intent_type = "recommendation"
            self.is_complex_intent = True
            return
        if WH_WORDS_SIMPLE.search(q):
            self.intent_type = "factoid"
            self.is_simple_intent = True
            return
        if YES_NO_PATTERNS.search(q):
            self.intent_type = "yes_no"
            self.is_simple_intent = True
            return
        self.intent_type = "unknown"