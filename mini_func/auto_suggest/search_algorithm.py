# Thuật toán tím kiếm áp dụng cho tập FAQ nhỏ
# Cân nhắc methodology.md nếu cần scale-up

from typing import List, Set, Union, Tuple
import re

# Bảng ánh xạ ký tự không dấu sang có dấu (regex)
CHAR_MAP = {
    'a': '[aàáạảãâầấậẩẫăằắặẳẵ]',
    'e': '[eèéẹẻẽêềếệểễ]',
    'i': '[iìíịỉĩ]',
    'o': '[oòóọỏõôồốộổỗơờớợởỡ]',
    'u': '[uùúụủũưừứựửữ]',
    'y': '[yỳýỵỷỹ]',
    'd': '[dđ]'
}

def remove_vietnamese_accents(s: str) -> str:
    s = s.lower()
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[đ]', 'd', s)
    return s

def _match_single_token(text_lower: str, token_lower: str) -> bool:
    """Hàm phụ trợ: Dò tìm 1 token duy nhất trong chuỗi gốc"""
    text_idx = 0
    query_idx = 0
    while text_idx < len(text_lower) and query_idx < len(token_lower):
        char_t = text_lower[text_idx]
        char_q = token_lower[query_idx]
        is_match = (char_t == char_q) or (remove_vietnamese_accents(char_t) == char_q)
        if is_match:
            query_idx += 1
        else:
            query_idx = 0 
            char_q0 = token_lower[0]
            if (char_t == char_q0) or (remove_vietnamese_accents(char_t) == char_q0):
                query_idx = 1  
        text_idx += 1
    return query_idx == len(token_lower)

def fuzzy_sequence_match(original_text: str, query: str) -> bool:
    """
    Kiểm tra xem chuỗi query có khớp mờ với original_text hay không
    - Gõ không dấu: bắt mọi từ có dấu và không dấu
    - Gõ có dấu: bắt khớp chính xác từ có dấu
    Cho phép đảo vị trí từ (vd: "mới vay" và "vay mới")
    """
    query_tokens = query.split()
    if not query_tokens:
        return False
    text_lower = original_text.lower()
    # Kiểm tra từng token
    for token in query_tokens:
        token_lower = token.lower()
        if not _match_single_token(text_lower, token_lower): return False      
    return True

def fuzzy_highlight(original_text: str, query: str) -> dict:
    """Highlight bỏ qua khoảng trắng nhưng giữ nguyên dấu và định dạng gốc"""
    query_tokens = query.split()
    if not query_tokens:
        return {"raw_text": original_text}
    patterns = []
    # Xây dựng regex cho từ
    for token in query_tokens:
        token_no_spaces = token.lower()
        regex_parts = []
        for char in token_no_spaces:
            unaccented_char = remove_vietnamese_accents(char)
            if char == unaccented_char and char in CHAR_MAP:
                regex_parts.append(CHAR_MAP[char])
            else:
                regex_parts.append(re.escape(char))
        patterns.append(r''.join(regex_parts))
    # nối tất cả các chuỗi regex
    combined_pattern_str = '|'.join(f"({p})" for p in patterns)
    try:
        pattern = re.compile(combined_pattern_str, re.IGNORECASE)
        # dùng lambda để bọc span chính xác
        highlighted_text = pattern.sub(
            lambda m: f"<span class='autocomplete-item-highlight'>{m.group(0)}</span>", 
            original_text
        )
        return {
            "raw_text": original_text,
            "highlighted_html": highlighted_text
        }
    except Exception:
        return {"raw_text": original_text}

INTENT_STOPWORDS = [
    "hướng dẫn", "cách", "làm sao để", "làm thế nào", "làm sao", "cách để",
    "như thế nào", "ở đâu", "khi nào", "tại sao", "vì sao", "thế nào", "như nào",
    "quy trình", "thủ tục", "các bước", "quy định", "chính sách", 
    "phương pháp", "điều kiện", "chi tiết", "cụ thể", "thông tin", 
    "tài liệu", "biểu mẫu", "form mẫu",
    "cho tôi hỏi", "cho mình hỏi", "xin hỏi", "hỏi về", "tôi muốn", "bạn", "bạn ơi",
    "mình muốn", "tôi cần", "mình cần", "cần hỗ trợ", "hỗ trợ giúp", 
    "giúp tôi", "tư vấn giúp", "xin cấp", "yêu cầu", "xin", "cho xin",
    "xử lý", "khắc phục", "giải quyết", "sửa lỗi", "cài đặt", 
    "thiết lập", "cấu hình", "kiểm tra", "tra cứu",
    "về việc", "liên quan đến", "về vấn đề", "trường hợp",
    "trong trường hợp", "khi bị", "nếu bị", "do bị", "bị"
]

class AutoSuggestEngine:
    """ Core engine of auto suggestion """
    def __init__(self, database: List[Union[str, Tuple[str]]]):
        self.database = self._preprocess_database(database)
        self.intent_stopwords = INTENT_STOPWORDS
        
    def _capitalize_first_letter(self, text: str) -> str:
        text = str(text).strip()
        if not text:
            return text
        return text[0].upper() + text[1:]

    def _preprocess_database(self, raw_db: List[Union[str, Tuple[str]]]) -> List[Union[str, Tuple[str]]]:
        """Tiền xử lý toàn bộ database trước khi lưu vào RAM"""
        processed_db = []
        for item in raw_db:
            if isinstance(item, tuple):
                clean_text = self._capitalize_first_letter(item[0])
                processed_db.append((clean_text,))
            else:
                clean_text = self._capitalize_first_letter(item)
                processed_db.append(clean_text)
        return processed_db
    
    def _normalize_query(self, query: str) -> str:
        """Hàm hỗ trợ: bỏ các từ xã giao, gây nhiễu để lấy core intent"""
        normalized = query.lower()
        for word in self.intent_stopwords:
            normalized = re.sub(r'\b' + word + r'\b', ' ', normalized)
        return ' '.join(normalized.split())
    
    def _calculate_score(self, original_text: str, query: str) -> int:
        """Hàm chấm điểm độ ưu tiên của từ khóa"""
        text_unaccented = remove_vietnamese_accents(original_text).lower()
        query_unaccented = remove_vietnamese_accents(query).strip().lower()
        
        # rank 1: khớp nguyên tử (word boundary)
        # vd: gõ "atm" -> khớp đúng chữ "ATM" đứng độc lập trong "ATM là gì"
        if re.search(r'\b' + re.escape(query_unaccented) + r'\b', text_unaccented):
            return 100
            
        # rank 2: khớp chuỗi liên tục bên trong 1 từ khác
        # vd: gõ "atm" -> khớp chữ "batman" (nếu có)
        if query_unaccented in text_unaccented:
            return 80
            
        query_tokens = query_unaccented.split()
        
        # rank 3: đảo vị trí nhưng khớp các từ độc lập
        # VD: Gõ "mới vay" -> tìm chữ "mới" và chữ "vay" tách rời trong câu
        all_tokens_as_words = all(re.search(r'\b' + re.escape(t) + r'\b', text_unaccented) for t in query_tokens)
        if all_tokens_as_words:
            return 70
            
        # rank 4: khớp xuyên khoảng trắng
        # vd: gõ "atm" -> khớp "đặt máy" (dat may -> datmay)
        return 50 
    
    def search(self, query: str, limit: int = 5) -> List[str]:
        try:
            # tiền xử lý
            cleaned_query = self._normalize_query(query)
            if not cleaned_query:
                cleaned_query = query
            # Khởi tạo mảng lưu kết quả kèm điểm số
            scored_results = []
            for faq in self.database:
                faq = faq[0] if isinstance(faq, tuple) else faq
                faq = str(faq)
                if fuzzy_sequence_match(faq, cleaned_query):
                    # Nếu khớp, tính điểm của câu này
                    score = self._calculate_score(faq, cleaned_query)
                    scored_results.append((score, faq))
                    
            # Sắp xếp mảng theo điểm số (giảm dần) 
            # bằng điểm thì giữ nguyên
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # áp limit và highlight
            final_results = []
            for score, faq in scored_results[:limit]:
                highlighted = fuzzy_highlight(faq, query)
                final_results.append(highlighted)
            return final_results
        except Exception as e:
            print (f"[AutoSuggestEngine] error at search:", e)
            return []