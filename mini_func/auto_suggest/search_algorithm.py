# Đây chỉ là phần demo auto suggest bằng thuật toán Word-Level Suffix Trie, áp dụng cho tập FAQ nhỏ.
# Cân nhắc đề xuất trong methodology.md để tăng khả năng scale-up của hệ thống
from typing import List, Set
import re

def remove_vietnamese_accents(s: str) -> str:
    """Hàm loại bỏ dấu tiếng Việt, chuyển về chữ cái la-tinh cơ bản"""
    s = s.lower()
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[đ]', 'd', s)
    return s

def highlight_text(original_text: str, query: str) -> str:
    """Hàm bọc thẻ HTML cho các từ khóa trùng khớp (có hoặc không dấu)"""
    unaccented_query_words = remove_vietnamese_accents(query).split()
    
    # Chia câu gốc thành mảng các từ, bảo toàn khoảng trắng và dấu câu nếu cần
    # (Để đơn giản, ở đây cắt bằng split() và nối bằng dấu cách)
    words = original_text.split()
    highlighted_words = []
    
    for word in words:
        unaccented_word = remove_vietnamese_accents(word)
        match_found = False
        
        # Kiểm tra xem từ hiện tại có bắt đầu bằng bất kỳ từ khóa nào người dùng gõ không
        for q_word in unaccented_query_words:
            if unaccented_word.startswith(q_word):
                # Nếu khớp, bọc thẻ span quanh từ gốc có dấu
                highlighted_words.append(f"<span class='highlight'>{word}</span>")
                match_found = True
                break
                
        if not match_found:
            highlighted_words.append(word)
            
    return " ".join(highlighted_words)

class TrieNode:
    def __init__(self):
        self.children = {}
        # Thay vì chỉ lưu True/False, lưu tập hợp ID của các câu gốc để chống trùng lặp
        self.original_texts = set()

class WordSuffixTrie:
    def __init__(self):
        self.root = TrieNode()
        self.faq_database = [] # Nơi lưu trữ văn bản gốc

    def insert(self, text: str, text_id: int):
        # Chuẩn hóa câu gốc thành câu không dấu
        unaccented_text = remove_vietnamese_accents(text)
        
        # Cắt các từ dựa trên câu KHÔNG DẤU
        words = unaccented_text.split()
        
        for i in range(len(words)):
            suffix_str = " ".join(words[i:])
            # Truyền cả suffix KHÔNG DẤU và text gốc CÓ DẤU vào cây
            self._insert_suffix(suffix_str, text, text_id)
            
    def _insert_suffix(self, suffix: str, original_text: str, text_id: int):
        node = self.root
        for char in suffix:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            
        # Ở Node cuối của hậu tố này, ghi nhận ID câu gốc
        node.original_texts.add((text_id, original_text))

    def _dfs(self, node: TrieNode, limit: int, results: Set[tuple]):
        """Duyệt sâu để lấy kết quả"""
        if len(results) >= limit:
            return
        
        # Nếu node này kết thúc một hậu tố hợp lệ, thêm câu gốc vào kết quả
        if node.original_texts:
            for item in node.original_texts:
                results.add(item)
                
        for char, child_node in node.children.items():
            self._dfs(child_node, limit, results)

    def search(self, query: str, limit: int = 5) -> List[str]:
        node = self.root
        
        # Chuẩn hóa cả chuỗi gõ của người dùng thành không dấu
        unaccented_query = remove_vietnamese_accents(query.strip())
        
        for char in unaccented_query:
            if char not in node.children:
                return []
            node = node.children[char]
            
        results = set()
        self._dfs(node, limit, results)
        
        final_results = []
        for item in list(results)[:limit]:
            original_text = item[1]
            highlighted = highlight_text(original_text, query)
            final_results.append(highlighted)
            
        return final_results