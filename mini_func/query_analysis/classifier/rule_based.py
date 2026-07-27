# Giói thiệu:
# Option A phân loại câu hỏi bằng Regex và Keyword matching, không cần GPU hay model AI
# Mục đích:
# 1. Fast-path classifier: xử lý ~40% queries trong <5ms
# 2. Cung cấp "Hard Override" signals cho Hybrid classifier (Option B+)
# 3. Dùng standalone khi không có GPU/vLLM (offline testing)
# Cách dùng:
# from classifier.rule_based import RuleBasedClassifier
# classifier = RuleBasedClassifier()
# result = classifier.classify("...")

import logging
from .query_classifier import BaseClassifier, ClassifyResult
from .signals.structural import StructuralSignals
from .signals.semantic import SemanticSignals
from .signals.retrieval_predict import RetrievalSignals
logger = logging.getLogger(__name__)
DEBUG_SIGNALS = False

class RuleBasedClassifier (BaseClassifier):
    """_summary_

    Args:
        BaseClassifier (_type_): _description_
    """
    def _classify_impl(self, query):
        """_summary_

        Args:
            query (_type_): _description_

        Returns:
            _type_: _description_
        """
        structural = StructuralSignals(query)
        semantic = SemanticSignals(query)
        retrieval = RetrievalSignals(
            query=query,
            domain_count = semantic.domain_count,
            word_count = structural.word_count,
        )

        # debug mode: in tất cả signals để trace lỗi
        # logger.debug(f"Structural: {structural.__dict__}")
        # logger.debug(f"Semantic: {semantic.__dict__}")
        
        # MENTAL FLOWCHART - EARLY EXIT
        # Thứ tự thực thi:
        # 1. chit-chat guard (với banking intent override)
        # 1b. off-domain question guard (có ? nhưng 0 banking)
        # 1c. r&r exception rule
        # 1d. bare banking noun -> clarify
        # 2. complex hard override
        # 2b. tiered ambiguous check
        # 3+. Full scoring
        
        # B1: kiểm tra chi-chat
        # Banking intent Priority override:
        # - nếu có greeting/emotion NHƯNG cũng có banking term + intent -> ko chit-chat
        # VD: "chào bạn lãi suất vay hiện tại bao nhiêu" -> simple
        # VD: "hello check giúp swift fee" -> simple
        if (structural.has_greeting or structural.emotion_count > 0) and not (semantic.has_banking_terms and structural.has_query_intent()):
            return ClassifyResult(
                label="chit-chat",
                confidence=0.9,
                reason="Chit-chat detected with no banking intent",
                path="rule"
            )
        
        # B1b: off-domain question with ? but - banking terms
        # vd: "hôm nay ăn gì?" -> có ? 0 liên quan banking -> chit-chat
        if structural.has_question_mark and not semantic.has_banking_terms:
            return ClassifyResult(
                label="chit-chat",
                confidence=0.8,
                reason="Question with no banking terms (off-domain)",
                path="rule"
            )
        
        # B1c: E&R Exception Rule
        # các yêu cầu liên quan vai trò, trách nhiệm, R&R của phòng ban là các tìm kiếm phứt tạp
        # ép nhẵn thẳng sang complex thay vì gắn clarify
        RR_KEYWORDS = {"r&r", "r & r", "vai trò", "trách nhiệm", "role", "responsibility", "responsibilities",}
        if any(kw in structural.query_lower for kw in RR_KEYWORDS):
            return ClassifyResult(
                label="complex",
                confidence=0.9,
                reason="R&R keyword detected",
                path="rule"
            )
        
        # B1d: Bare banking noun check
        # có banking terms NHƯNG không có intent -> clarify
        # phải chạy SAU chit-chat để guard để tránh: "hello bảo hiểm khoản vay" bị gán clarify thay vì chit-chat
        # ví dụ:
        # - "swift" -> clarify (no intent)
        # "bảo hiểm khoản vay" -> clarify
        # "lãy suất vay mua nhà" -> clarify
        # NHƯNG:
        # "tra cứu phí swift" -> has_action_verb -> has_intent = True
        # "phí chuyển tiền swift" -> lookup_trigger + 4 từ -> has_intent = True
        if semantic.has_banking_terms and not structural.has_query_intent() and structural.word_count <= 8:
            return ClassifyResult(
                label="clarify",
                confidence=0.85,
                reason="Bare banking noun (no intent/action detected)",
                path="rule"
            )
        
        # B2: kiểm tra complex (hard override)
        # từ nối mạnh (so sánh, nhân quả, ...) -> complex
        if structural.has_strong_connectives:
            return ClassifyResult(
                label="complex",
                confidence=0.9,
                reason="Strong connective word detected (so sánh, nhân quả, vv)",
                path="rule"
            )
        
        # B2b: Tiered ambiguous check
        # phân tầng thay vì gán clarify đồng loạt
        # signal 3.3: câu quá ngắn
        # signal 3.4: câu chứa từ khoá bank quá rộng
        if structural.word_count <= 2 and semantic.has_banking_terms:
            return ClassifyResult(
                label="clarify",
                confidence=0.8,
                reason="Query too short (<3 words) with banking terms",
                path="rule"
            )
        
        # multi-domain -> complex
        if semantic.domain_count >= 2:
            return ClassifyResult(
                label="complex",
                confidence=0.85,
                reason=f"Multi-domain ({semantic.domain_count} domains detected)",
                path="rule",
            )
        
        # Disguised intent (cách tính lãi suất là gì?)
        if semantic.has_disguised_intent:
            return ClassifyResult(
                label="complex",
                confidence=0.85,
                reason="Disguised intent: procedure inside definition",
                path="rule",
            )
        
        # procedural intent rõ ràng
        if structural.has_procedural:
            return ClassifyResult(
                label="complex",
                confidence=0.85,
                reason=f"Procedural intent: {semantic.intent_type}",
                path="rule",
            )
            
        # Temporal (thay đổi theo thời gian)
        if semantic.has_temporal and semantic.has_banking_terms:
            return ClassifyResult(
                label="complex",
                confidence=0.80,
                reason="Temporal change + banking domain",
                path="rule",
            )
            
        # recommendation
        if structural.has_recommendation and semantic.has_banking_terms:
            return ClassifyResult(
                label="complex",
                confidence=0.80,
                reason="recommendation intent",
                path="rule",
            )
            
        # cal intent: câu có số tiền + yêu cầu tính toán -> complex
        if structural.has_calculation:
            return ClassifyResult(
                label="complex",
                confidence=0.85,
                reason="Calculation intent: numeric + output required",
                path="rule",
            )
            
        # nhiều sub-question (>= 2 dấu hỏi)
        if structural.question_mark_count >= 2:
            return ClassifyResult(
                label="complex",
                confidence=0.85,
                reason=f"Multiple sub-questions ({structural.question_mark_count} question marks)",
                path="rule",
            )
            
        # Premise setup dài + banking terms -> complex
        if structural.has_premise_setup and semantic.has_banking_terms:
            return ClassifyResult(
                label="complex",
                confidence=0.75,
                reason=f"Premise setup (>8 words before comma) + banking domain",
                path="rule",
            )
            
        # FULL SCORING (cho các case còn lại)
        
        # technical  jaron density: nếu >= 2 thuật ngữ kỹ thuật
        # nội bộ -> complex dù có WH-word đơn giản
        INTERNAL_JARGON = [
            "rcvr", "tsbd", "ltv", "dscr", "casa", "npl", "cof",
            "equipment term loan", "lor", "los", "nii", "nim",
            "tài sản bảo đảm", "tỷ lệ cho vay",
        ]
        
        jargon_count = sum(
            1 for j in INTERNAL_JARGON
            if j in structural.query_lower
        )
        if jargon_count >= 2:
            return ClassifyResult(
                label="complex",
                confidence=0.8,
                reason=f"High jargon density ({jargon_count} internal terms)",
                path="rule",
            )
            
        # Nếu có WH-word đơn giản + 1 domain -> simple
        # giảm confidence
        # (có thể là disguised intent chưa được bật)
        if structural.has_wh_simple and semantic.domain_count == 1:
            return ClassifyResult(
                label="simple",
                confidence=0.75,
                reason=f"WH-word + single domain ({semantic.intent_type})",
                path="rule",
            )
            
        # Yes/No question + banking domain -> simple
        # chỉ áp dụng cho câu ngắn (<= 12 từ). câu yes/no dài thường phứt tạp
        # hơn bề ngoài (ví dụ: "bank có cho vay đối tượng ..." = 17 từ)
        # giảm confidence
        if structural.has_yes_no and semantic.has_banking_terms and structural.word_count <= 12:
            return ClassifyResult(
                label="simple",
                confidence=0.7,
                reason=f"Yes/no question (<=12 words), {semantic.domain_count} domain(s)",
                path="rule",
            )
            
        # câu có baking terms nhưng k có từ nối -> phân loại theo độ dài
        if semantic.has_baking_terms and not structural.has_strong_connectives:
            if structural.word_count > 15:
                return ClassifyResult(
                    label="complex",
                    confidence=0.6,
                    reason=f"Long query ({structural.word_count} words),"
                            f"banking terms, no connectives",
                    path="rule",
                )
                
            confidence = 0.6 if structural.word_count <= 10 else 0.5
            return ClassifyResult(
                label="simple",
                confidence=confidence,
                reason=f"Banking terms present, no connectives, {structural.word_count} words",
                path="rule",
            )
            
        # không có banking terms - phân biệt rõ hơn
        if not semantic.has_banking_terms:
            if structural.has_greeting or structural.emotion_count >= 2:
                return ClassifyResult(
                    label="chit-chat",
                    confidence=0.8,
                    reason="Chit-chat: social markers, no banking terms",
                    path="rule",
                )
            elif structural.word_count <= 3:
                return ClassifyResult(
                    label="clarify",
                    confidence=0.7,
                    reason="Very short, no banking terms -> need clarification",
                    path="rule",
                )
            else:
                return ClassifyResult(
                    label="chit-chat",
                    confidence=0.5,
                    reason="Off-domain: no banking terms detected, uncertain",
                    path="rule",
                )
                
        # fallback cuối cùng
        return ClassifyResult(
            label="simple",
            confidence=0.5,
            reason="Default fallback, no strong signals",
            path="rule",
        )
        
    def get_signals(self, query: str) -> dict:
        """_summary_

        Args:
            query (str): _description_

        Returns:
            dict: _description_
        """
        structural = StructuralSignals(query)
        semantic = SemanticSignals(query)
        retrieval = RetrievalSignals(
            query=query,
            domain_count=semantic.domain_count,
            word_count=structural.word_count,
        )
        return {
            "structural": structural,
            "semantic": semantic,
            "retrieval": retrieval,
        }