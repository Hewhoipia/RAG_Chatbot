# Đây chỉ là phần demo auto suggest bằng thuật toán áp dụng cho tập FAQ nhỏ ~1000 câu
# Cân nhắc đề xuất trong methodology.md để tăng khả năng scale-up của hệ thống nếu cần
from typing import List, Tuple, Union, Set
import re

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

def fuzzy_sequence_match(original_text: str, query: str) -> bool:
    """
    Kiểm tra xem chuỗi query có khớp mờ với original_text hay không.
    - Gõ không dấu: Bắt mọi từ có dấu.
    - Gõ có dấu: Yêu cầu khớp chính xác dấu.
    """
    # CHÚ Ý: Lột MỌI khoảng trắng, đưa về chữ thường, nhưng GIỮ NGUYÊN DẤU
    text_no_spaces = original_text.replace(" ", "").lower()
    query_no_spaces = query.replace(" ", "").lower()
    
    if not query_no_spaces:
        return False

    text_idx = 0
    query_idx = 0
    
    while text_idx < len(text_no_spaces) and query_idx < len(query_no_spaces):
        char_t = text_no_spaces[text_idx]
        char_q = query_no_spaces[query_idx]
        
        # LOGIC CỐT LÕI MỚI:
        # Điều kiện 1 (char_t == char_q): Khớp chính xác (Cùng có dấu hoặc cùng không dấu).
        # Điều kiện 2: Ký tự gốc lột dấu đi thì giống với ký tự người dùng gõ.
        is_match = (char_t == char_q) or (remove_vietnamese_accents(char_t) == char_q)
        
        if is_match:
            query_idx += 1
        else:
            query_idx = 0 
            # Bắt đầu lại nếu ký tự hiện tại khớp với ký tự đầu của query
            char_q0 = query_no_spaces[0]
            if (char_t == char_q0) or (remove_vietnamese_accents(char_t) == char_q0):
                query_idx = 1
                
        text_idx += 1
        
    return query_idx == len(query_no_spaces)

# Bản đồ ánh xạ ký tự không dấu sang bộ ký tự có dấu (Regex)
CHAR_MAP = {
    'a': '[aàáạảãâầấậẩẫăằắặẳẵ]',
    'e': '[eèéẹẻẽêềếệểễ]',
    'i': '[iìíịỉĩ]',
    'o': '[oòóọỏõôồốộổỗơờớợởỡ]',
    'u': '[uùúụủũưừứựửữ]',
    'y': '[yỳýỵỷỹ]',
    'd': '[dđ]'
}

def fuzzy_highlight(original_text: str, query: str) -> str:
    """Highlight bỏ qua khoảng trắng nhưng giữ nguyên dấu và định dạng gốc"""
    # Vẫn giữ nguyên dấu của query để phân tích
    query_no_spaces = query.replace(" ", "").lower()
    if not query_no_spaces:
        return original_text
        
    regex_parts = []
    
    for char in query_no_spaces:
        unaccented_char = remove_vietnamese_accents(char)
        
        # NẾU GÕ KHÔNG DẤU (char == unaccented_char) -> Bung Regex để highlight tất cả biến thể
        if char == unaccented_char and char in CHAR_MAP:
            regex_parts.append(CHAR_MAP[char])
            
        # NẾU GÕ CÓ DẤU (Ví dụ 'ẻ', 'đ') -> Khóa Regex, bắt buộc highlight chính xác ký tự đó
        else:
            regex_parts.append(re.escape(char))
            
    # Nối các ký tự lại, cho phép có khoảng trắng xen giữa
    pattern_str = r'\s*'.join(regex_parts)
    
    try:
        pattern = re.compile(f"({pattern_str})", re.IGNORECASE)
        highlighted_text = pattern.sub(r"<span class='autocomplete-item-highlight'>\1</span>", original_text)
        return {
            "raw_text": original_text,
            "highlighted_html": highlighted_text
        }
    except Exception:
        return {
            "raw_text": original_text
        }

class AutoSuggestEngine:
    def __init__(self, database: List[Union[str, Tuple[str]]]):
        self.database = database
        
    def _calculate_score(self, original_text: str, query: str) -> int:
        """Hàm chấm điểm độ ưu tiên của từ khóa"""
        text_unaccented = remove_vietnamese_accents(original_text).lower()
        query_unaccented = remove_vietnamese_accents(query).lower()
        
        # MỨC 1 (100 ĐIỂM): Khớp nguyên từ (Word Boundary)
        # VD: Gõ "atm" -> Khớp đúng chữ "ATM" đứng độc lập trong "ATM là gì"
        if re.search(r'\b' + re.escape(query_unaccented) + r'\b', text_unaccented):
            return 100
            
        # MỨC 2 (80 ĐIỂM): Khớp chuỗi liên tục (Nằm bên trong một từ khác)
        # VD: Gõ "atm" -> Khớp chữ "Batman" (nếu có)
        if query_unaccented in text_unaccented:
            return 80
            
        # MỨC 3 (50 ĐIỂM): Khớp xuyên khoảng trắng (Fuzzy match)
        # VD: Gõ "atm" -> Khớp "đặt máy" (dat may -> datmay)
        return 50    
    
    def search(self, query: str, limit: int = 5) -> List[str]:
        try:
            # Khởi tạo mảng lưu kết quả kèm điểm số
            scored_results = []
            for faq in self.database:
                faq = faq[0] if isinstance(faq, tuple) else faq
                faq = str(faq)
                if fuzzy_sequence_match(faq, query):
                    # Nếu khớp, tính điểm của câu này
                    score = self._calculate_score(faq, query)
                    scored_results.append((score, faq))
                    
            # BƯỚC QUAN TRỌNG: Sắp xếp mảng theo điểm số (giảm dần) 
            # Cùng điểm thì giữ nguyên thứ tự gốc
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # Cuối cùng mới cắt lấy limit và bọc thẻ Highlight
            final_results = []
            for score, faq in scored_results[:limit]:
                highlighted = fuzzy_highlight(faq, query)
                final_results.append(highlighted)
                
            return final_results
        except Exception as e:
            print (f"[AutoSuggestEngine] error at search:", e)
            return []