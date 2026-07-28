"""Giới thiệu
- Tier 1 Signals - phân tích cấu trúc câu (structural analysis)
- Bóc tách các dấu hiệu hình thái rõ ràng từ câu hỏi mà ko cần hiểu ngữ nghĩa sâu
Mục đích:
1. Signal 1.1: phân tích cấu trúc cảm thán, WH-word, chào hỏi
2. Signal 1.2: đếm mật độ thông tin entity counting
3. Signal 1.3: độ dài câu (tie-breaker, yếu nhất)
4. Signal 1.4: từ nối logic (so sánh, nhân quả, điều kiện - mạnh nhất)
Cách sử dụng:
from classifier.signals.structural import StructuralSignals
signals = StructuralSignals(query)
print(signals.has_strong_connectives) # true/false
print(signals.entity_count) # int
Lưu ý:
- tất cả patterns đã được compile sẵn lúc import để tối ưu tốc độ
- Signal 1.4 (connective words) là tín hiệu cực mạnh, dùng cho hard overrides
- signal 1.3 chỉ dùng làm tie-breaker khi các signal khác k thể quyết định"""

import re
from dataclasses import dataclass, field
from typing import List

# Compiled regex patterns

# Signal 1.1
# từ cảm thán
EMOTION_MARKERS = re.compile(
    r'(?:nhé|nha|ạ|ơi|hả|á|haha|hihi|hehe|trời ơi|hay quá|tuyệt|wow|ok|okie)',
    re.IGNORECASE
)
# chào hỏi
GREETING_PATTERNS = re.compile(
    r'(?:^chào|^xin chào|^hello|^hi|^hey|bạn khoẻ|cảm ơn|thank|bye|tạm biệt)',
    re.IGNORECASE
)
# đại từ nhân xưng
PERSONAL_PRONOUNS = re.compile(
    r'(?:bạn ơi|ad ơi|mình ơi|shop ơi|anh ơi|chị ơi|em ơi)',
    re.IGNORECASE
)
# WH-words
WH_WORDS_SIMPLE = re.compile(
    r'(?:là gì|bao nhiêu|bao lâu|ở đâu|khi nào|mấy giờ|what is|how much|where)',
    re.IGNORECASE
)

# Signal 1.4: từ nối mạnh
# so sánh
COMPARISON_CONNECTIVES = re.compile(
    r'(?:so sánh|khác nhau|so với|hơn|hay là|nên chọn|khác biệt|compare|versus|vs|different|difference|differ'
    r'|giữa\s+\S+\s+và\s+|loại\s+\S+\s+(?:và|hoặc|&)\s+)',
    re.IGNORECASE
)
# Nhân quả
CAUSAL_CONNECTIVES = re.compile(
    r'(?:tại sao|nguyên nhân|lý do|vì sao|do đâu|bởi vì|cho nên|dẫn đến|ảnh hưởng)',
    re.IGNORECASE
)
#điều kiện/ giả thời/ bắt buộc
CONDITIONAL_CONNECTIVES = re.compile(
    r'(?:nếu|trong trường hợp|giả sử|khi và chỉ khi|bắt buộc|điều kiện|yêu cầu)',
    re.IGNORECASE
)
# quy trình/thủ tục/quy định
PROCEDURAL_PATTERNS = re.compile(
    r'(?:hướng dẫn|cách|thủ tục|quy định|quy trình|các bước|làm sao|thế nào|step)',
    re.IGNORECASE
)
# tổng hợp / liệt kê
AGGREGATION_PATTERNS = re.compile(
    r'(?:tổng hợp|liệt kê|những|các|danh sách|list|tất cả)',
    re.IGNORECASE
)
# disguised intent: "cách/quy định ... là gì" -> complex (bề ngoài simple nhưng là procedure/policy)
DISGUISED_INTENT = re.compile(
    r'(?:quy định.*là gì|cách.*là gì|làm thế nào.*là gì)',
    re.IGNORECASE
)
#temporal -> complex
# bỏ "hiện tại" chỉ giữ các pattern thể hiện SỰ THAY ĐỔI thật sự theo thời gian
TEMPORAL_PATTERNS = re.compile(
    r'(?:từ.*đến|thay đổi|cũ.*mới|trước.*sau|\d{4}.*\d{4}|gần đây)',
    re.IGNORECASE
)
# đề xuất -> complex
RECOMMENDATION_PATTERNS = re.compile(
    r'(?:đề xuất|gợi ý|nên|tư vấn|recommend|suggest|advise)',
    re.IGNORECASE
)
# yes/no -> simple
YES_NO_PATTERNS = re.compile(
    r'(?:có.*không|có được.*không|được.*không|phải.*không|hay không|có cần.*không)',
    re.IGNORECASE
)

# signal 1.5: calculation intent
CALCULATION_PATTERNS = re.compile(
    r'(?:tính|công thức|tính toán|tỷ lệ|lãi suất|hạn mức|cách tính)',
    re.IGNORECASE
)

# IMPLICIT INTENT PATTERNS - phát hiện intent k cần dấu câu

# signal 1.6: imperative / yêu cầu hành động
# user chat dạng mệnh lệnh: "cho tôi xem...", "kiểm tra...", "check..."
# "mở"/"xem" chỉ được bắt ở đây (kèm context), không ở ACTION_VERB
IMPERATIVE_PATTERNS = re.compile(
    r'(?:cho tôi xem|kiểm tra|check|giúp tôi|hãy|vui lòng)',
    re.IGNORECASE
)

# signal 1.7: action verbs - động từ hành động banking-specific
# khi có action verb -> user đang yêu cầu hành động cụ thể, không cần dấu câu ?
# Lưu ý: "mở", "xem" bị loại vì quá rộng
ACTION_VERB_PATTERNS = re.compile(
    r'(?:làm|thực hiện|cập nhật|đăng ký|hủy|thay đổi)',
    re.IGNORECASE
)

# Signal 1.8; implicit lookup triggers
# cụm từ mang ý tra cứu ngầm - khi đi kèm domain term + đủ context = có intent
# vd "phí chuyên tiền swift" -> phí (trigger) + swift(domain) + 4 từ -> intent
IMPLICIT_LOOKUP_TRIGGERS = re.compile(
    r'(?:thông tin|chi tiết|tài liệu|chính sách|hồ sơ)',
    re.IGNORECASE
)

@dataclass
class StructuralSignals:
    """
    Tier 1 signal extractor analyzing query structure via regex without deep semantic understanding.

    Extracts greeting/emotion markers, WH-words, question marks, word count,
    strong connective words, calculation intent, and implicit intent patterns.
    """
    query: str
    query_lower: str = field(init=False)

    # 1.1
    emotion_count:int = field(init=False, default=0)
    has_greeting:bool = field(init=False, default=False)
    has_personal_pronoun: bool = field(init=False, default=False)
    has_wh_simple:bool = field(init=False, default=False)
    has_yes_no:bool = field(init=False, default=False)

    # 1.2
    word_count: int = field(init=False,default=0)
    has_question_mark:bool = field(init=False, default=False)
    has_exclamation: bool = field(init=False, default=False)
    question_mark_count: int = field(init=False, default=0)

    # signal 1.4
    has_comparison: bool = field(init=False, default=False)
    has_causal: bool = field(init=False, default=False)
    has_conditional: bool = field(init=False, default=False)
    has_procedural: bool = field(init=False, default=False)
    has_aggregation: bool = field(init=False, default=False)
    has_disguised_intent: bool = field(init=False, default=False)
    has_temporal: bool = field(init=False, default=False)
    has_recommendation: bool = field(init=False, default=False)

    has_calculation : bool = field(init=False, default=False)

    has_premise_setup: bool = field(init=False, default=False)
    
    has_strong_connectives: bool = field(init=False, default=False)

    # signal 1.6
    has_imperative: bool = field(init=False, default=False)
    has_action_verb: bool = field(init=False, default=False)
    has_lookup_trigger: bool = field(init=False, default=False)

    def __post_init__(self):
        """
        Initialize all structural signals by running compiled regex patterns on the input query.

        Workflow:
        - Step 1: Normalize query and compute basic metrics (word count, punctuation)
        - Step 2: Detect emotion, greeting, WH-word, yes/no patterns
        - Step 3: Detect connective patterns (comparison, causal, conditional, aggregation)
        - Step 4: Detect implicit intent (imperative, action verb, lookup trigger)
        """
        self.query_lower = self.query.lower()
        self.word_count = len(self.query.split())
        self.has_question_mark = "?" in self.query
        self.has_exclamation = "!" in self.query
        self.question_mark_count = self.query.count("?")

        self.emotion_count = len(EMOTION_MARKERS.findall(self.query_lower))
        self.has_greeting = bool(GREETING_PATTERNS.search(self.query_lower))
        self.has_personal_pronoun = bool(PERSONAL_PRONOUNS.search(self.query_lower))
        self.has_wh_simple = bool(WH_WORDS_SIMPLE.search(self.query_lower))
        self.has_yes_no = bool(YES_NO_PATTERNS.search(self.query_lower))

        self.has_comparison = bool(COMPARISON_CONNECTIVES.search(self.query_lower))
        self.has_causal = bool(CAUSAL_CONNECTIVES.search(self.query_lower))
        self.has_conditional = bool(CONDITIONAL_CONNECTIVES.search(self.query_lower))
        self.has_procedural = bool(PROCEDURAL_PATTERNS.search(self.query_lower))
        self.has_aggregation = bool(AGGREGATION_PATTERNS.search(self.query_lower))
        self.has_disguised_intent = bool(DISGUISED_INTENT.search(self.query_lower))
        self.has_temporal = bool(TEMPORAL_PATTERNS.search(self.query_lower))
        self.has_recommendation = bool(RECOMMENDATION_PATTERNS.search(self.query_lower))
        self.has_calculation = bool(CALCULATION_PATTERNS.search(self.query_lower))
        
        self._detect_premise_setup()
        
        self.has_strong_connectives = any([
            self.has_comparison,
            self.has_causal,
            self.has_conditional,
            self.has_aggregation,
        ])
        self.has_imperative = bool(IMPERATIVE_PATTERNS.search(self.query_lower))
        self.has_action_verb = bool(ACTION_VERB_PATTERNS.search(self.query_lower))
        self.has_lookup_trigger = bool(IMPLICIT_LOOKUP_TRIGGERS.search(self.query_lower))
    
    def _detect_premise_setup(self):
        """Detect long premise setup (>8 words before first comma) indicating complex context framing."""
        if "," in self.query_lower:
            premise = self.query_lower.split(",")[0].strip()
            premise_word_count = len(premise.split())
            if premise_word_count > 8:
                self.has_premise_setup = True
                
    def is_likely_chitchat(self) -> bool:
        """Quick heuristic for casual/social queries based on emotion markers and short greetings.

        Returns:
            bool: True if query likely contains only social markers with no substantive question.
        """
        if self.emotion_count >= 2 and not self.has_question_mark:
            return True
        if self.has_greeting and self.word_count <= 6:
            return True
        return False
    
    def is_likely_complex(self) -> bool:
        """Quick heuristic for complex queries based on strong connective word presence.

        Returns:
            bool: True if query contains comparison, causal, conditional, or aggregation connectives.
        """
        return self.has_strong_connectives
    
    @property
    def has_query_intent(self) -> bool:
        """Determine if the query expresses a clear question intent via explicit or implicit signals.

        Returns:
            bool: True if explicit intent (WH-word, ?, yes/no), structural connectives,
                  imperative, action verb, or lookup trigger with sufficient context is detected.
        """
        # tầng 1: explicit intent (dấu câu + từ hỏi)
        if any([
            self.has_wh_simple,
            self.has_question_mark,
            self.has_yes_no,
        ]):
            return True
        # tầng 2: Structure connectives
        if any([
            self.has_procedural,
            self.has_comparison,
            self.has_causal,
            self.has_recommendation,
            self.has_calculation,
        ]):
            return True
        # tầng 3: implicit intent
        if self.has_imperative:
            return True
        if self.has_action_verb:
            return True
        if self.has_lookup_trigger and self.word_count >= 6:
            return True
        return False