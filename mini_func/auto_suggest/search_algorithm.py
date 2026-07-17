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
    Kiểm tra xem TẤT CẢ các từ trong query có xuất hiện trong original_text không.
    Cho phép đảo vị trí từ (vd: "mới vay" = "vay mới")
    """
    query_tokens = query.split() # Tách từ khóa thành mảng ["mới", "vay"]
    if not query_tokens:
        return False

    text_lower = original_text.lower()

    # Kiểm tra từng token, nếu có 1 token không tồn tại -> Loại ngay câu này
    for token in query_tokens:
        token_lower = token.lower()
        if not _match_single_token(text_lower, token_lower):
            return False
            
    return True

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

def fuzzy_highlight(original_text: str, query: str) -> dict:
    """Highlight nhiều từ khóa rời rạc, không phân biệt thứ tự"""
    query_tokens = query.split()
    if not query_tokens:
        return {"raw_text": original_text}
        
    patterns = []
    
    # Xây dựng regex cho TỪNG từ khóa
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
            
    # Nối tất cả các chuỗi regex bằng toán tử OR (|)
    # VD: (v\s*a\s*y) | (m\s*ớ\s*i)
    combined_pattern_str = '|'.join(f"({p})" for p in patterns)
    
    try:
        pattern = re.compile(combined_pattern_str, re.IGNORECASE)
        # Sử dụng lambda để bọc span chính xác cho group vừa match được
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
    # Nhóm hỏi đáp, tìm kiếm giải pháp (How-to / What / Why)
    "hướng dẫn", "cách", "làm sao để", "làm thế nào", "làm sao", "cách để",
    "như thế nào", "ở đâu", "khi nào", "tại sao", "vì sao", "thế nào", "như nào",
    
    # Nhóm thủ tục, văn bản (Process / Documentation)
    "quy trình", "thủ tục", "các bước", "quy định", "chính sách", 
    "phương pháp", "điều kiện", "chi tiết", "cụ thể", "thông tin", 
    "tài liệu", "biểu mẫu", "form mẫu",

    # Nhóm từ xưng hô, lịch sự, yêu cầu (Request / Politeness)
    "cho tôi hỏi", "cho mình hỏi", "xin hỏi", "hỏi về", "tôi muốn", "bạn", "bạn ơi",
    "mình muốn", "tôi cần", "mình cần", "cần hỗ trợ", "hỗ trợ giúp", 
    "giúp tôi", "tư vấn giúp", "xin cấp", "yêu cầu", "xin", "cho xin",

    # Nhóm động từ mang tính chất giải quyết sự cố (Action Fillers)
    "xử lý", "khắc phục", "giải quyết", "sửa lỗi", "cài đặt", 
    "thiết lập", "cấu hình", "kiểm tra", "tra cứu",

    # Nhóm từ nối, từ đệm chỉ hoàn cảnh (Context Wrappers)
    "về việc", "liên quan đến", "về vấn đề", "trường hợp",
    "trong trường hợp", "khi bị", "nếu bị", "do bị", "bị"
]

class AutoSuggestEngine:
    def __init__(self, database: List[Union[str, Tuple[str]]]):
        # Nạp dữ liệu qua một hàm tiền xử lý thay vì gán trực tiếp
        self.database = self._preprocess_database(database)
        
        # Danh sách các từ nhiễu không mang giá trị tìm kiếm cốt lõi
        self.intent_stopwords = INTENT_STOPWORDS
        
    def _capitalize_first_letter(self, text: str) -> str:
        """Viết hoa chữ cái đầu, giữ nguyên định dạng của phần còn lại"""
        text = str(text).strip()
        if not text:
            return text
        # Lấy ký tự đầu viết hoa + toàn bộ chuỗi từ ký tự thứ 2 trở đi
        return text[0].upper() + text[1:]

    def _preprocess_database(self, raw_db: List[Union[str, Tuple[str]]]) -> List[Union[str, Tuple[str]]]:
        """Tiền xử lý toàn bộ database trước khi lưu vào RAM"""
        processed_db = []
        for item in raw_db:
            if isinstance(item, tuple):
                # Bóc tuple ra, xử lý text, rồi đóng gói lại thành tuple
                clean_text = self._capitalize_first_letter(item[0])
                processed_db.append((clean_text,))
            else:
                # Nếu là string bình thường
                clean_text = self._capitalize_first_letter(item)
                processed_db.append(clean_text)
                
        return processed_db
    
    def _normalize_query(self, query: str) -> str:
        """Hàm lột bỏ các từ nhiễu để trích xuất core intent"""
        normalized = query.lower()
        # Thay thế các từ stopword bằng khoảng trắng
        for word in self.intent_stopwords:
            # Dùng regex để xóa nguyên cụm từ (tránh xóa nhầm chữ cái bên trong từ khác)
            normalized = re.sub(r'\b' + word + r'\b', ' ', normalized)
        
        # Xóa các khoảng trắng thừa sau khi lột từ
        return ' '.join(normalized.split())
    
    def _calculate_score(self, original_text: str, query: str) -> int:
        """Hàm chấm điểm độ ưu tiên của từ khóa"""
        text_unaccented = remove_vietnamese_accents(original_text).lower()
        query_unaccented = remove_vietnamese_accents(query).strip().lower()
        
        # MỨC 1 (100 ĐIỂM): Khớp nguyên cụm dính liền (Word Boundary)
        # VD: Gõ "vay mới" -> Bắt đúng cụm "vay mới"
        if re.search(r'\b' + re.escape(query_unaccented) + r'\b', text_unaccented):
            return 100
            
        # MỨC 2 (80 ĐIỂM): Khớp chuỗi liên tục (Đúng thứ tự, nằm trong từ khác)
        if query_unaccented in text_unaccented:
            return 80
            
        query_tokens = query_unaccented.split()
        
        # MỨC 3 (70 ĐIỂM): Đảo vị trí nhưng khớp nguyên các từ độc lập
        # VD: Gõ "mới vay" -> Tìm thấy chữ "mới" và chữ "vay" tách rời trong câu
        all_tokens_as_words = all(re.search(r'\b' + re.escape(t) + r'\b', text_unaccented) for t in query_tokens)
        if all_tokens_as_words:
            return 70
            
        # MỨC 4 (50 ĐIỂM): Khớp xuyên khoảng trắng hoặc đảo lộn vị trí từ
        return 50 
    
    def search(self, query: str, limit: int = 5) -> List[str]:
        try:
            # Tiền xử lý (Pre-process) câu gõ của người dùng
            # "hướng dẫn tính lãi suất" -> sẽ biến thành "tính lãi suất"
            cleaned_query = self._normalize_query(query)
            # Nếu lột sạch không còn chữ nào (VD user gõ mỗi chữ "cách"), thì dùng lại query gốc
            if not cleaned_query:
                cleaned_query = query
            # Khởi tạo mảng lưu kết quả kèm điểm số
            scored_results = []
            for faq in self.database:
                faq = faq[0] if isinstance(faq, tuple) else faq
                faq = str(faq)
                # Đưa cleaned_query vào thuật toán dò tìm
                if fuzzy_sequence_match(faq, cleaned_query):
                    # Nếu khớp, tính điểm của câu này
                    score = self._calculate_score(faq, cleaned_query)
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