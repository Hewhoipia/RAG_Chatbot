# =============================================================================
# FILE: text_generate_prod_v7_vllm_client.py
# DESCRIPTION: LLM Client for vLLM Server v7 - Qwen3.5-4B
# DATE: 2024-12-24
# VERSION: 7.1-Client (20+ Concurrent Users, Non-Thinking Mode)
# =============================================================================
#
# GIỚI THIỆU (Introduction):
# --------------------------
# LLM Client cho vLLM Server v7, tối ưu cho Qwen3.5-4B.
# Qwen3.5 mặc định sẽ think — sử dụng enable_thinking=False qua
# chat_template_kwargs để tắt thinking.
#
# LƯU Ý QUAN TRỌNG VỀ QWEN3.5:
# - Qwen3.5 KHÔNG hỗ trợ /think và /nothink soft switch
# - Để tắt thinking, dùng: chat_template_kwargs={"enable_thinking": False}
# - Sampling non-thinking: temp=0.7, top_p=0.8, presence_penalty=1.5
#
# MỤC ĐÍCH (Purpose):
# -------------------
# 1. Kết nối đến vLLM server qua OpenAI-compatible API
# 2. Generate responses với Qwen3.5 model
# 3. Hỗ trợ streaming cho perceived latency thấp
# 4. Toggle Thinking/Non-Thinking mode per-request via API params
#
# CÁCH THỨC VẬN HÀNH (How it works):
# ----------------------------------
# 1. vLLM server chạy trên port 8011 với Qwen3.5 model
# 2. LLMDeploymentV7VLLMClient gửi HTTP requests đến vLLM
# 3. ChatEngineV7VLLMClient format prompts với Qwen3.5 chat template
# 4. Thinking mode toggle qua chat_template_kwargs (không dùng soft switch)
#
# LƯU Ý (Notes):
# --------------
# - Stop tokens: im_end (giống Qwen3/Qwen2.5)
# - Sampling: temp=0.7, top_p=0.8, presence_penalty=1.5 (non-thinking)
# - KHÔNG dùng greedy decoding (temperature=0) - gây repetition loops
# - 100% LOCAL - không sử dụng OpenAI cloud
#
# THAY ĐỔI SO VỚI v7.0:
# -------------------
# - Model: Qwen3-30B-A3B-GPTQ-Int4 → Qwen3.5-4B
# - REMOVED: /no_think soft switch (Qwen3.5 không hỗ trợ)
# - CHANGED: Thinking control chỉ qua chat_template_kwargs
# - REMOVED: generate_chat_async, generate_batch_async (không sử dụng)
#
# THAM KHẢO:
# ----------
# - https://huggingface.co/Qwen/Qwen3.5-4B
#
# =============================================================================


import asyncio
import logging
import time
import re
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass, field
import threading
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

# OpenAI-compatible client library (calls local vLLM server, NOT OpenAI cloud)
from openai import AsyncOpenAI

from config_v7 import config

logger = logging.getLogger(__name__)


# =============================================================================
# QWEN3 STOP TOKENS (giống Qwen2.5)
# =============================================================================
# Qwen3 vẫn sử dụng im_end stop token
QWEN_STOP_TOKEN_1 = "<" + "|im_end|" + ">"
QWEN_STOP_TOKEN_2 = "<" + "|endoftext|" + ">"
QWEN_STOP_TOKENS = [QWEN_STOP_TOKEN_1, QWEN_STOP_TOKEN_2]


# =============================================================================
# THINKING MODE PARSER (NEW in v7)
# =============================================================================
def parse_thinking_response(response: str) -> Tuple[str, Optional[str]]:
    """
    Giới thiệu: Bộ phân tích cú pháp (Parser) chuyên dụng bóc tách nội dung suy luận (Thinking mode) khỏi câu trả lời chung.
    
    Mục đích sử dụng: Xử lý các mô hình có cấu trúc chuỗi tư duy xuất ra nằm trong cặp tag `<think>...</think>` như Qwen, DeepSeek nhằm rút xuất câu trả lời cuối (final answer) cho end-user và phân loại chuỗi tư duy riêng cho UI rendering hoặc logs.
    
    Cách thức hoạt động:
    - B1: Thiết lập pattern Regex khớp với đoạn giới hạn tag XML `<think>`.
    - B2: Yêu cầu Regex regex từ đoạn text nhận được.
    - B3: Trả riêng nội dung Thinking. Dùng hàm sub xóa bỏ đoạn thinking trong văn bản chính để bóc lại phần text answer. Nếu cụm suy nghĩ rỗng thì trả thẳng cụm gốc kèm Null cho text think.
    
    Đầu vào - Đầu ra của hàm đó:
    - Đầu vào:
        + `response` (str): Toàn bộ đoạn văn bản generate trả ra nguyên vẹn từ AI.
    - Đầu ra:
        + (Tuple[str, Optional[str]]): Trả tuple 2 biến cục bộ (Câu trả lời tĩnh, Chuỗi lý do suy luận).
        
    Lưu ý: Thuộc tính re.DOTALL được đặt cờ để parse xuyên break lines (Xuống dòng văn bản).
    """
    # Pattern to match <think>...</think> block
    think_pattern = r'<think>(.*?)</think>'
    
    # Find thinking content
    match = re.search(think_pattern, response, re.DOTALL)
    
    if match:
        thinking_content = match.group(1).strip()
        # Remove thinking block from response
        answer = re.sub(think_pattern, '', response, flags=re.DOTALL).strip()
        return answer, thinking_content
    else:
        return response.strip(), None


# =============================================================================
# THREAD-SAFE METRICS
# =============================================================================
@dataclass
class ClientMetrics:
    """
    ==========================================================================
    GIỚI THIỆU: Thread-safe production metrics for client mode.
    MỤC ĐÍCH: Theo dõi performance của LLM client
    CÁCH THỨC VẬN HÀNH:
    - record_request(): Ghi nhận mỗi request thành công
    - record_error(): Ghi nhận lỗi
    - get_stats(): Lấy thống kê tổng hợp
    LƯU Ý: Sử dụng Lock để đảm bảo thread-safety
    ==========================================================================
    """
    total_requests: int = 0
    total_tokens_generated: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0
    thinking_requests: int = 0  # NEW: Track thinking mode requests
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def record_request(self, tokens: int, latency_ms: float, thinking: bool = False):
        """
        Giới thiệu: Ghi nhận telemetry thành công của một luồng Request tạo sinh (Generate process) từ máy chủ nội tại.
        
        Mục đích sử dụng: Phục vụ thu gom số liệu vận hành hiệu năng số lượng (Metrics) cho bảng phân tích tổng hợp.
        
        Cách thức hoạt động:
        - Sử dụng Lock để block vòng lặp thread, sau đó cộng dồn các chỉ số sinh lượt query, độ phân giải token và độ trễ vào biến lưu trữ chung. Đạt thêm điểm logging thinking nếu cờ boolean parameter bằng True.
        
        Đầu vào - Đầu ra của hàm đó:
        - Đầu vào:
            + `tokens` (int): Chỉ số lượng token đã sinh ra.
            + `latency_ms` (float): Số mili giây hoàn thành trọn bộ Request.
            + `thinking` (bool): Có xài cờ suy nghĩ (Thinking process) hay không.
        - Đầu ra:
            + (None)
            
        Lưu ý: Không thay đổi giá trị thuộc tính của class trực tiếp mà chưa Acquire Lock vì sự cố lỗi Race Condition trong Asynchronous process.
        """
        with self._lock:
            self.total_requests += 1
            self.total_tokens_generated += tokens
            self.total_latency_ms += latency_ms
            if thinking:
                self.thinking_requests += 1
    
    def record_error(self):
        """
        Giới thiệu: Đếm số lượng sự cố xảy xa đối với LLM vLLM Client.
        
        Mục đích sử dụng: Đo đếm sự suy thoái theo dòng thời gian nếu có vấn đề network HTTP.
        
        Cách thức hoạt động:
        - Đóng thread Lock và tăng biến số theo dõi lượng lỗi truy cập.
        
        Đầu vào - Đầu ra của hàm đó:
        - Đầu vào: Không có tham số.
        - Đầu ra:
            + (None)
            
        Lưu ý: Function hướng dọi cho việc kiểm duyệt tracker cục bộ máy ảo Python API.
        """
        with self._lock:
            self.errors += 1
    
    def get_stats(self) -> Dict:
        """
        Giới thiệu: API trích xuất thống kê từ đầu chu kì của quá trình thực thi LLM Engine (bao gồm số query, vòng quay trả tokens và độ phân giải trễ).
        
        Mục đích sử dụng: Hỗ trợ Dashboard kiểm toán nội bộ thông qua đường dẫn endpoint route log kiểm tra Health Check hay Metrics hệ thống.
        
        Cách thức hoạt động:
        - Khóa lấy Lock thread reading, tính trung bình Avg latency sấp sỉ và bao gói format JSON đóng dạng Dict nhả về.
        
        Đầu vào - Đầu ra của hàm đó:
        - Đầu vào: Không có tham số.
        - Đầu ra:
            + (Dict): Nhóm bộ dữ liệu theo dõi sức khỏe kết xuất.
            
        Lưu ý: Các thống kê của class ClientMetrics được reset xóa mới nếu quá trình máy chủ Fast API restart (Do dữ liệu trữ trên RAM Memory của tiến trình web).
        """
        with self._lock:
            avg_latency = self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0
            return {
                "total_requests": self.total_requests,
                "total_tokens_generated": self.total_tokens_generated,
                "avg_latency_ms": round(avg_latency, 2),
                "errors": self.errors,
                "thinking_requests": self.thinking_requests,
            }


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================
class LLMError(Exception):
    """Base exception cho LLM errors."""
    pass


class LLMGenerationError(LLMError):
    """Lỗi khi generate response."""
    pass


class LLMConnectionError(LLMError):
    """Lỗi khi không thể kết nối đến vLLM server."""
    pass


# =============================================================================
# LLM CLIENT V7 - vLLM CHAT COMPLETIONS CLIENT FOR QWEN3.5-4B
# =============================================================================
class LLMDeploymentV7VLLMClient:
    """
    ==========================================================================
    GIỚI THIỆU:
    LLM Client gọi vLLM server qua OpenAI-compatible API.
    Tối ưu cho Qwen3.5-4B.
    
    MỤC ĐÍCH SỬ DỤNG:
    - Kết nối đến vLLM server
    - Generate response từ prompt (completions API)
    - Hỗ trợ streaming
    - Toggle Thinking/Non-Thinking mode qua chat_template_kwargs
    
    CÁCH THỨC VẬN HÀNH:
    1. vLLM server chạy riêng biệt trên port 8011
    2. Client gửi HTTP request đến server
    3. Server trả về response (có thể có <think> block)
    4. Client parse và trả về answer
    
    LƯU Ý:
    - "OpenAI" chỉ là API format, chạy 100% LOCAL
    - Qwen3.5: thinking toggle qua chat_template_kwargs, KHÔNG dùng /nothink
    ==========================================================================
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8011/v1",
        model_name: str = None,
        timeout: float = 180.0,
    ):
        """
        Giới thiệu: Khởi tạo kết nối Client đến AI Model qua chuẩn giao tiếp mạng phi truyền thống vLLM.
        
        Mục đích sử dụng: Xây dựng cổng trung chuyển (Gateway) thực thi điều phối lưu lượng mạng từ hệ thống Core Backend tới cụm server ảo vLLM.
        
        Cách thức hoạt động:
        - Đọc thông số Base URL, tên Model và cấu hình trễ (Timeout) lưu vào RAM.
        - Bắt luồng khởi tạo module `AsyncOpenAI` kết nối qua thư viện HTTP. Lên nòng kích hoạt biến đếm `ClientMetrics` theo dõi tốc độ kết nối, số lỗi trả về và lượng Token Query.
        
        Đầu vào - Đầu ra của hàm đó:
        - Đầu vào:
            + `base_url` (str): Domain gốc trỏ tới IP vLLM (Ví dụ: `http://localhost:8011/v1`).
            + `model_name` (str): Mã tên mã vLLM Model phục vụ load API (mặc định lấy config).
            + `timeout` (float): Quãng thời gian chịu tải trễ (thường thiết lập 180.0 giây cho các hàm sinh thinking dài).
        - Đầu ra: Không có (Khởi tạo Object).
        
        Lưu ý: API Key của thư viện AsyncOpenAI bị set tĩnh về giá trị `EMPTY` do chạy cục bộ hệ nội bộ on-premise, không kết nối mây Cloud ngoài.
        """
        logger.info(f"Khởi tạo LLM Client V7.1 (Qwen3.5-4B)...")
        logger.info(f"   Server URL: {base_url}")
        
        self.base_url = base_url
        self.model_name = model_name or config.vllm.model_name or "Qwen3.5-4B"
        self.timeout = timeout
        self.metrics = ClientMetrics()
        
        # Initialize OpenAI-compatible async client
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="EMPTY",
            timeout=timeout,
        )
        
        logger.info(f"LLM Client V7.1 sẵn sàng!")
        logger.info(f"   Model: {self.model_name}")
        logger.info(f"   Timeout: {timeout}s")
        logger.info(f"   Default Thinking Mode: {config.qwen3.enable_thinking}")
        logger.info(f"   Stop tokens: Qwen format (im_end)")
    
    async def generate_async(
        self, 
        messages: List[Dict],
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        enable_thinking: bool = None,
    ) -> str:
        """
        Giới thiệu: Hàm bất đồng bộ kết xuất chuỗi phản hồi văn bản (Generate response) chạy ngầm (Non-blocking) qua giao thức mạng vLLM.
        
        Mục đích sử dụng: Giao tiếp trực tiếp với LLM qua định dạng Chat Completions API. Gọi trực tiếp khối xử lý Model thay vì code chặn tiến trình luồng (blocking).
        
        Cách thức hoạt động:
        - B1: Kiểm tra override của cờ boolean thinking (mode suy luận). Lôi cấu hình config.
        - B2: Soạn list params Sampling (Nhiệt độ, Top K/P).
        - B3: Submit HTTP request qua AsyncOpenAI. Đính kèm argument `chat_template_kwargs` đặc thù của kiến trúc Qwen vLLM để ép buộc bật tắt thinking.
        - B4: Model tách tag `<think>` vô `reasoning_content` (tự động phân rã). Cắt rỗng khối đó và trả phần answer ở `message.content`. Phóng telemetry ghi log vào Lock.
        
        Đầu vào - Đầu ra của hàm đó:
        - Đầu vào:
            + `messages` (List[Dict]): Dàn tin nhắn JSON hội thoại context.
            + `max_tokens` (int): Cắt ngắn dài text.
            + `enable_thinking` (bool): Gạt chuyển đổi mode sâu tư duy.
            + (Tham số Sampling còn lại): Tuỳ chỉnh nhiệt model tạo sinh.
        - Đầu ra:
            + (str): Một khối lượng chuỗi chữ Text ròng sạch nhất đại diện cho câu trả lời cuối (đã ẩn đoạn suy nghĩ).
            
        Lưu ý: Luôn bắt buộc nã model với hàm `chat.completions.create` (ChatAPI) chứ không xài Raw `completions`, nếu không hệ thống sẽ lỗi Ignore bỏ qua lệnh cắt token thinking.
        """
        start_time = time.time()
        
        # Determine thinking mode
        use_thinking = enable_thinking if enable_thinking is not None else config.qwen3.enable_thinking
        
        # Get sampling params based on mode
        sampling = config.qwen3.get_sampling_params(thinking=use_thinking)
        
        # Use provided values or defaults from sampling config
        actual_temp = temperature if temperature is not None else sampling["temperature"]
        actual_top_p = top_p if top_p is not None else sampling["top_p"]
        actual_top_k = top_k if top_k is not None else sampling["top_k"]
        actual_max_tokens = max_tokens if max_tokens is not None else sampling["max_tokens"]
        presence_penalty = sampling["presence_penalty"]
        
        try:
            # ============================================================
            # QUAN TRỌNG: Dùng chat.completions.create()
            # (KHÔNG PHẢI completions.create())
            #
            # Chỉ Chat Completions API mới hỗ trợ chat_template_kwargs
            # để toggle thinking mode. Với raw Completions API,
            # enable_thinking bị ignore → model LUÔN think.
            #
            # Ref: https://huggingface.co/Qwen/Qwen3.5-4B#instruct-or-non-thinking-mode
            # ============================================================
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=actual_max_tokens,
                temperature=actual_temp,
                top_p=actual_top_p,
                extra_body={
                    "top_k": actual_top_k,
                    # Qwen3.5: toggle thinking via chat_template_kwargs
                    # False → trả lời thẳng, KHÔNG generate <think>
                    # True → think trước, vLLM tách vào reasoning_content
                    "chat_template_kwargs": {"enable_thinking": use_thinking},
                },
                frequency_penalty=0.3,
                presence_penalty=presence_penalty,
                # KHÔNG cần stop tokens - Chat API tự xử lý
            )
            
            # ============================================================
            # RESPONSE FORMAT (Chat API + vLLM --reasoning-parser qwen3):
            # - choices[0].message.content = câu trả lời sạch
            # - choices[0].message.reasoning_content = thinking (hoặc None)
            # → KHÔNG có <think>...</think> tags trong content
            # ============================================================
            generated_text = (response.choices[0].message.content or "").strip()
            
            # Log thinking content nếu có (chỉ cho debug)
            reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
            if reasoning:
                logger.debug(f"Reasoning content ({len(reasoning)} chars): {reasoning[:100]}...")
            
            tokens_generated = response.usage.completion_tokens if response.usage else len(generated_text.split())
            
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record_request(tokens_generated, latency_ms, thinking=use_thinking)
            
            return generated_text
            
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"LLM Client error: {e}")
            raise LLMGenerationError(f"Generation failed: {e}")
    
    async def generate_stream(
        self, 
        messages: List[Dict],
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        enable_thinking: bool = None,
    ) -> AsyncGenerator[Dict[str, str], None]:
        """
        Giới thiệu: Hàm bất đồng bộ trả sinh dòng mạng chữ (Streaming text) nối nhau dạng cục thời gian thực (Real-time).
        
        Mục đích sử dụng: Thả kết quả nháy gõ chữ (Yield tokens) về Front-end HTTP giúp giảm triệt để độ delay chờ đợi (Perceived Latency) phản ứng đầu bảng.
        
        Cách thức hoạt động:
        - B1: Setup các Sampling config và flag giống lệnh `generate_async`.
        - B2: Bật trigger `stream=True`, mở kết nối event luồng ngầm tới API OpenAI-compatible của vLLM chờ Event nổ xuống.
        - B3: Hứng khối chunk của stream bất đồng bộ. Check thuộc tính delta parse dict `reasoning_content` (chữ cho reasoning) và `content` (chữ trả lời).
        - B4: Nhả liên tiến độ cữ (yield) một Dict JSON đóng tag kiểu 'type' báo cáo dạng thức nội dung cho frontend vẽ giao diện (là thinking hay answer block).
        
        Đầu vào - Đầu ra của hàm đó:
        - Đầu vào: Giống phương thức async tĩnh toàn chuỗi. Đầu vào mấu chốt là dải `messages` JSON hội thoại lịch sử.
        - Đầu ra:
            + (AsyncGenerator): Kiểu chạy lặp async Generator cho ra liên tục Yielding object JSON `{type, content}` cho hệ thống SSE Router nhận diện.
            
        Lưu ý: Nếu cài thông số `enable_thinking=True`, vLLM reasoning parser phía server tự động đẩy Yield trước bộ chữ thinking, sau đó nhả null field đoạn này, rồi bắt đầu nhả bộ chữ answer text content nối tiếp.
        """
        # Determine thinking mode
        use_thinking = enable_thinking if enable_thinking is not None else config.qwen3.enable_thinking
        
        # Get sampling params based on mode
        sampling = config.qwen3.get_sampling_params(thinking=use_thinking)
        
        actual_temp = temperature if temperature is not None else sampling["temperature"]
        actual_top_p = top_p if top_p is not None else sampling["top_p"]
        actual_top_k = top_k if top_k is not None else sampling["top_k"]
        actual_max_tokens = max_tokens if max_tokens is not None else sampling["max_tokens"]
        presence_penalty = sampling["presence_penalty"]
        
        try:
            # Chat Completions API với stream=True
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=actual_max_tokens,
                temperature=actual_temp,
                top_p=actual_top_p,
                stream=True,
                extra_body={
                    "top_k": actual_top_k,
                    "chat_template_kwargs": {"enable_thinking": use_thinking},
                },
                frequency_penalty=0.3,
                presence_penalty=presence_penalty,
            )
            
            # vLLM reasoning parser tự tách sẵn:
            # - delta.reasoning_content → thinking tokens
            # - delta.content → answer tokens
            async for chunk in stream:
                delta = chunk.choices[0].delta
                
                reasoning_delta = getattr(delta, 'reasoning_content', None)
                content_delta = delta.content
                
                if reasoning_delta:
                    yield {"type": "thinking", "content": reasoning_delta}
                if content_delta:
                    yield {"type": "answer", "content": content_delta}
                    
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"LLM Client streaming error: {e}")
            raise LLMGenerationError(f"Streaming failed: {e}")
    
    def get_metrics(self) -> Dict:
        """
        Giới thiệu: Bóc xuất báo cáo đo đạc Telemetry hoạt động của Class LLM Client V7.
        
        Mục đích sử dụng: Kết nối với biến metrics phục vụ Router kiểm soát.
        
        Cách thức hoạt động: Call thẳng phương thức lấy JSON từ class Component `ClientMetrics` ngầm.
        
        Đầu vào - Đầu ra của hàm đó:
        - Đầu vào: Không có.
        - Đầu ra: (Dict) Kết xuất JSON độ trễ và lượt requests theo chu kỳ reset máy.
        
        Lưu ý: Hàm là alias wrapper nội hạt gọi ngắn gọn cho `self.metrics.get_stats`.
        """
        return self.metrics.get_stats()
    
    async def health_check(self) -> Dict:
        """
        Giới thiệu: Chức năng theo dõi sức khoẻ tiến trình máy chủ API vLLM phục vụ Model LLM độc lập nằm ngoài bộ Client này.
        
        Mục đích sử dụng: Gọi hàm Kiểm tra tính sẵn sàng Server trả Request và phát hiện lỗi rẽ nhánh kẹt Server, sụp process mạng port 8011.
        
        Cách thức hoạt động:
        - Trỏ gọi thư viện `client.models.list` ngầm để ping mạng lấy trạng thái. Nếu mượt thì format chèn biến Dict nhả ra status. Bật except ngắt nều time out lỗi connection.
        
        Đầu vào - Đầu ra của hàm đó:
        - Đầu vào: Không có.
        - Đầu ra: (Dict) Đóng gói info status array.
        
        Lưu ý: Function này được gọi mồi độc lập từ core file `mybot_prod` endpoint GET /health mà không ảnh hưởng luồng Chat Generate chung.
        """
        try:
            models = await self.client.models.list()
            return {
                "status": "healthy",
                "server": self.base_url,
                "model": self.model_name,
                "available_models": [m.id for m in models.data] if models.data else [],
                "metrics": self.get_metrics(),
                "thinking_mode_default": config.qwen3.enable_thinking,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "server": self.base_url,
                "error": str(e),
            }


# =============================================================================
# CHAT ENGINE V7 CLIENT MODE - QWEN3 TEMPLATE
# =============================================================================
class ChatEngineV7VLLMClient:
    """
    ==========================================================================
    GIỚI THIỆU:
    Chat engine format prompts và gọi vLLM server (CLIENT MODE).
    Sử dụng Qwen3.5 chat template.
    
    MỤC ĐÍCH SỬ DỤNG:
    - Format query + context thành prompt
    - Gọi LLM client để generate response
    - Xử lý sources để hiển thị
    
    LƯU Ý:
    - Qwen3.5 KHÔNG hỗ trợ /think và /nothink soft switch
    - Thinking mode control hoàn toàn qua chat_template_kwargs trên API
    - Template được tối ưu cho bilingual (EN/VI)
    ==========================================================================
    """
    
    # PROMPT TEMPLATE - OPTIMIZED for bilingual RAG
    PROMPT_TEMPLATE = """
    You are an assistant for Shinhan Bank's customer. 
    Answer the QUESTION based on the CONTEXT from documents containing bilingual (English and Vietnamese) policy information.

    CONTEXT (reference only - do not copy language from here):
    {context}

    QUESTION: {question}

    CRITICAL LANGUAGE RULE:
    - Detect the language of the QUESTION above.
    - Your ANSWER MUST be 100% in the exact same language as the QUESTION (e.g., if QUESTION is in Korean, answer in Korean; if English, answer in English; if Vietnamese, answer in Vietnamese).
    - NEVER mix languages. NEVER follow the language of the CONTEXT.

    OFF-TOPIC HANDLING:
    - If the QUESTION is unrelated to banking policies, procedures, or the CONTEXT above (e.g., personal questions, jokes, casual chat like "How are you?"), respond politely declining to answer.
    - IMPORTANT: You must decline in the EXACT SAME LANGUAGE as the user's QUESTION. 
      (Example for English: "I'm sorry, but I can only help with questions related to Shinhan Bank policies and procedures. Please ask a relevant question.")
    - Do NOT make up answers for off-topic questions.

    ANSWER (strictly same language as QUESTION, not CONTEXT):
    """.strip()
    
    def __init__(
        self, 
        llm_client: LLMDeploymentV7VLLMClient = None,
        document_dir: str = None
    ):
        """
        Giới thiệu: Hàm dựng khuôn kiến trúc Chatbot sinh phản hồi (Chat Engine Module).
        
        Mục đích sử dụng: Lắp ráp các tham số phụ thuộc cơ bản (Dependencies) như Module liên kết Model API và thư mục đường dẫn nhằm chuẩn bị tài nguyên đổ cho thuật toán truy hồi.
        
        Cách thức hoạt động: Nhận Parameter con trỏ của `LLMDeploymentV7VLLMClient` và cấp phát URL ổ đĩa cứng chứa kho tài liệu PDF tham chiếu.
        
        Đầu vào - Đầu ra của hàm đó:
        - Đầu vào:
            + `llm_client` (LLMDeploymentV7VLLMClient): Bọc bộ công cụ API HTTP Model.
            + `document_dir` (str): Dẫn xuất vị trí Folder vật lý chứa data (Lấy tự động từ config system nếu để rỗng).
        - Đầu ra: Không có (Khởi tạo Object).
        
        Lưu ý: Tại V7.1, chức năng nạp Tokenizer đã bị gỡ hoàn toàn do bộ gõ Template Chat Qwen 3.5 đã được phần backend vLLM tự động hóa tích hợp vào API Chat Completions. Ta chỉ cần nhòi mảng cấu trúc JSON là tự máy xử lý mượt mà.
        """
        self.llm_client = llm_client
        self.document_dir = document_dir or config.retrieval.data_dir
        # KHÔNG cần tokenizer - vLLM tự áp dụng Qwen3.5 chat template
        # khi sử dụng chat.completions.create() API
        logger.info("ChatEngine V7 initialized (Chat Completions API mode)")
    
    def create_messages(
        self, 
        query: str, 
        retrieved_nodes: List[Dict],
    ) -> List[Dict]:
        """
        Giới thiệu: Dựng khuôn prompt mảng Messages.
        
        Mục đích sử dụng: Gộp nhồi text truy vấn và context đính kèm thành list payload Array cho `chat.completions.create()`.
        
        Cách thức hoạt động:
        - B1: Vòng lặp For quét build ghép string nội dung `context` từ toàn thể mảng tĩnh `retrieved_nodes` thô.
        - B2: Định dạng (Format) `PROMPT_TEMPLATE` thành chuỗi User chuẩn bị đẩy cho AI.
        - B3: Đóng gói lại theo Role dict array báo hiệu tin nhắn của người dùng.
        
        Đầu vào - Đầu ra của hàm đó:
        - Đầu vào:
            + `query` (str): Parameter văn bản chính hỏi AI.
            + `retrieved_nodes` (List[Dict]): Parameter biến dữ liệu node thô của RAG.
        - Đầu ra:
            + (List[Dict]): Formatted Payload.
            
        Lưu ý: Không can thiệp Tokenizer manual nữa vì thư viện vLLM tự động áp khuôn Qwen3.5 chat template lên mảng JSON này.
        """
        # Build context with metadata
        context = ""
        for doc in retrieved_nodes:
            context += f"metadata: {doc['metadata']}\ncontent:{doc['text']}\n\n"
        
        # Format prompt using template
        prompt_content = self.PROMPT_TEMPLATE.format(question=query, context=context)
        
        # Trả về messages array - vLLM tự áp dụng Qwen3.5 chat template
        return [{"role": "user", "content": prompt_content}]
    
    def _process_sources(self, retrieved_nodes: List[Dict]) -> List[Dict]:
        """
        Giới thiệu: Hàm lọc xào nấu Extract nội dung hiển thị Sources.
        
        Mục đích sử dụng: Lấy bằng chứng Citation của file có dính đoạn text trả lời để render bảng hiển thị trên Web Front-End (Dành riêng cho cross-platform).
        
        Cách thức hoạt động:
        - Lặp qua các Document nodes.
        - Chạy luồng dò số trang qua các tham số linh động (`doc_items`, `page_numbers`, `page_idx`).
        - Loại bỏ ký hiệu Path trỏ của ổ cứng tĩnh Windows (VD: `\\\\?\\`), bóc đoạn Link file tĩnh phục vụ cấu trúc URL của trình duyệt nội bộ.
        - Loại bỏ Document Duplicate khóa trùng nhau xài Set `seen_sources` đánh dấu ID key.
        
        Đầu vào - Đầu ra của hàm đó:
        - Đầu vào: (List[Dict]) Dataset tĩnh vector chứa mảng tham số Meta.
        - Đầu ra: (List[Dict]) Trọn bộ Dictionary trả phần thân thô thành dạng Object format chuẩn URL (`filename`, `page_no`, `relative_path`).
        
        Lưu ý: Path rễ mục Web View Font-end luôn phải ép gạch xuyệt ngược sang Forward Slashes `/`.
        """
        sources = []
        seen_sources = set()
        
        for node_info in retrieved_nodes:
            meta = node_info.get('metadata', {})
            file_name = meta.get('file_name', None)
            
            # Fallback chain cho page number (hỗ trợ cả v2.10 và v2.11 metadata format)
            page_no = meta.get('page_no')
            if page_no is None and 'doc_items' in meta:
                try:
                    page_no = meta['doc_items'][0]['prov'][0]['page_no']
                except (KeyError, IndexError, TypeError):
                    pass
            if page_no is None and meta.get('page_numbers'):
                page_no = meta['page_numbers'][0]
            if page_no is None and meta.get('page_idx') is not None:
                page_no = meta.get('page_idx') + 1
            
            full_path = meta.get('file_path', '')
            
            # Cross-platform: Clean Windows long path prefix if present
            if full_path.startswith('\\\\?\\'):
                clean_path = full_path[4:]
            else:
                clean_path = full_path
            
            # Normalize path for current OS
            clean_path = os.path.normpath(clean_path)
            
            # Calculate relative path
            try:
                relative_path = os.path.relpath(clean_path, self.document_dir)
                # Always use forward slashes for URL compatibility
                relative_path_for_url = relative_path.replace('\\', '/')
            except ValueError:
                # Fallback if paths are on different drives (Windows) or invalid
                relative_path_for_url = file_name if file_name else ''
            
            display_page_no = page_no if page_no is not None else 1
            source_key = f"{file_name}-page-{page_no}"
            
            if source_key not in seen_sources:
                file_ext = os.path.splitext(file_name)[1].lower() if file_name else ''
                
                source_dict = {
                    "filename": file_name,
                    "page_no": display_page_no,
                    "relative_path": relative_path_for_url
                }
                
                if file_ext == '.docx':
                    source_dict["text"] = node_info.get('text', '')
                
                sources.append(source_dict)
                seen_sources.add(source_key)
        
        return sources
    
    async def generate_response(
        self, 
        query: str, 
        retrieved_nodes: List[Dict],
        enable_thinking: bool = None,
    ) -> Dict:
        """
        Giới thiệu: Luồng Endpoint chính gọi tiến trình tạo sinh trả lời từ ngữ cảnh đệm.
        
        Mục đích sử dụng: Trở thành Component cốt lõi gọi hàm tổng gom Text tạo chuỗi nội dung đồng nhất cấu hình kết quả sinh cho Route Non-streaming.
        
        Cách thức hoạt động:
        - B1: Pass Array qua khay đựng mảng `create_messages`.
        - B2: Triệu gọi Async Wait `llm_client.generate_async` để lấy được String Text Output đằng sau vLLM Server.
        - B3: Cắt gọt mảng format thông tin trích dẫn qua `_process_sources`.
        - B4: Trả gộp JSON block bao gồm cả answer mượt và mảng sources.
        
        Đầu vào - Đầu ra của hàm đó:
        - Đầu vào: 
            + `query` (str): Văn gốc hỏi ngầm định.
            + `retrieved_nodes` (List[Dict]): Dữ liệu document RAG.
            + `enable_thinking` (bool): Tham số kích hoạt logic tư duy.
        - Đầu ra:
            + (Dict): Nhóm đáp án theo form `{"answer": str, "sources": List[Dict]}`.
            
        Lưu ý: Nếu biến `enable_thinking` được Set Override tác động lên API Client thì ứng dụng vLLM sẽ bỏ qua Default Fallback từ config server gốc.
        """
        # Tạo messages array (KHÔNG cần format với tokenizer)
        messages = self.create_messages(query, retrieved_nodes)
        
        # Gọi Chat Completions API - thinking toggle qua extra_body
        response = await self.llm_client.generate_async(
            messages, 
            enable_thinking=enable_thinking,
        )
        sources = self._process_sources(retrieved_nodes)
        
        return {
            "answer": response,
            "sources": sources
        }
    
    async def generate_response_stream(
        self, 
        query: str, 
        retrieved_nodes: List[Dict],
        enable_thinking: bool = None,
    ) -> AsyncGenerator[Dict[str, str], None]:
        """
        Giới thiệu: Hàm truyền tải mạng kết nối dạng Stream văn bản liên tục (SSE Generator).
        
        Mục đích sử dụng: Bypass chuyển kênh (Delegate) tiến trình nháy chữ Async của Model Client ra luồng Frontend SSE Event để cho Client ngốn.
        
        Cách thức hoạt động:
        - Chạy vòng lặp `async for` đón bắt sự kiện Event chunk theo thời gian ngầm định Generator từ LLM Client.
        - Xả ra luồng Output Text yield thẳng vào phễu Response.
        
        Đầu vào - Đầu ra của hàm đó:
        - Đầu vào: Tham số giống hàm `generate_response`. Vẫn cần array Document context.
        - Đầu ra:
            + (AsyncGenerator): Block Yielding liên tục object JSON `{type, content}` phục giao thức.
            
        Lưu ý: Lượng SSE Chunks văng ra từ đây đã được vLLM tự động phân mảnh theo Type tư duy Thinking hoặc text Trả lời answer.
        """
        messages = self.create_messages(query, retrieved_nodes)
        async for event in self.llm_client.generate_stream(
            messages, 
            enable_thinking=enable_thinking,
        ):
            yield event
    
    # =========================================================================
    # v7.2 NEW: PROMPT TEMPLATES CHO CHIT-CHAT VÀ CLARIFY
    # =========================================================================
    CHITCHAT_PROMPT = """\
The user just sent a casual greeting or an out-of-domain message: '{query}'
Respond politely and warmly, and smoothly steer the conversation back to Shinhan Bank's products and services (e.g., credit cards, loans, bank accounts, etc.).
Keep the response brief, natural, and conversational.
IMPORTANT: You MUST respond in the exact same language that the user used in their message (e.g., if they write in English, answer in English; if Vietnamese, answer in Vietnamese; if Korean, answer in Korean, etc.)."""

    CLARIFY_PROMPT = """\
The user just sent a query that is too short or lacks context: '{query}'
Respond politely, asking them to provide more details or specific keywords so that you can accurately search for the banking information they need.
Keep the response brief and natural.
IMPORTANT: You MUST respond in the exact same language that the user used in their query (e.g., if they write in English, answer in English; if Vietnamese, answer in Vietnamese; if Korean, answer in Korean, etc.)."""

    async def generate_direct_response(
        self,
        query: str,
        response_type: str = "chit-chat",
    ) -> Dict:
        """
        Giới thiệu: Giao thức hàm chạy đường tắt (Shortcut bypass) đâm thẳng phản hồi văn bản tạo sinh.
        
        Mục đích sử dụng: Bỏ qua bước Retrieval rườm rà và chậm chạp. Dành riêng cho phục vụ nhánh giao tiếp vãng lai (nhỏ lẻ) như Xã giao `chit-chat` hoặc Làm rõ truy vấn `clarify`. Chống suy giảm trải nghiệm tốc độ của Chatbot đối với câu tán gẫu.
        
        Cách thức hoạt động:
        - Chọn Prompt Template đóng sẵn từ Constant (Tag Enumeration của `CHITCHAT_PROMPT` và `CLARIFY_PROMPT`).
        - Format chuỗi Prompt tĩnh kết hợp Input và xả qua Hàm Chat tạo sinh ép luồng phi suy luận (Non-thinking). Trả về JSON chèn tag rỗng.
        
        Đầu vào - Đầu ra của hàm đó:
        - Đầu vào:
            + `query` (str): Kịch bản truy xuất User gõ.
            + `response_type` (str): Nhãn Tag (chitchat/clarify).
        - Đầu ra:
            + (Dict): Box JSON mô phỏng cấu trúc RAG Answer nhưng chèn array sources thành rỗng và kèm param meta query_type.
            
        Lưu ý: Trò Shortcut Direct này giảm mạnh bạo yếu tố Latency Delay vLLM từ trung bình >2s xuống chỉ ở ngưỡng ~50ms siêu tốc độ.
        """
        if response_type == "clarify":
            prompt = self.CLARIFY_PROMPT.format(query=query)
        else:
            prompt = self.CHITCHAT_PROMPT.format(query=query)
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm_client.generate_async(
            messages, enable_thinking=False
        )
        
        return {
            "answer": response,
            "sources": [],
            "query_type": response_type,
        }
    
    async def generate_direct_response_stream(
        self,
        query: str,
        response_type: str = "chit-chat",
    ) -> AsyncGenerator[Dict[str, str], None]:
        """
        Giới thiệu: Bộ cài hàm Stream văn bản nhấp nháy đi đường tắt (Shortcut Generator) trực tiếp.
        
        Mục đích sử dụng: Nhả ngay chuỗi nhấp nháy tạo sự mượt mà UX (Cho SSE Event Chunk) với trường hợp nhánh logic xã giao lãng nhách hay kêu gọi diễn giải không cần phải Search DB.
        
        Cách thức hoạt động:
        - Áp Pattern tĩnh đệm bằng User query, gọi qua hàm Sink Stream Generate vùi logic Bypass tắt cờ Thinking Force gạt thành False.
        - Chờ Array Async For nhả yield đều đặn.
        
        Đầu vào - Đầu ra của hàm đó:
        - Đầu vào: Tham số giống hàm tĩnh cơ bản `query` và `response_type`.
        - Đầu ra:
            + (AsyncGenerator): Truyền dẫn SSE Stream Data text.
            
        Lưu ý: SSE Stream chunk Generator nhả qua route này không cung cấp tham số 'thinking' field để tiết kiệm vòng chạy mô hình LLM.
        """
        if response_type == "clarify":
            prompt = self.CLARIFY_PROMPT.format(query=query)
        else:
            prompt = self.CHITCHAT_PROMPT.format(query=query)
        
        messages = [{"role": "user", "content": prompt}]
        async for event in self.llm_client.generate_stream(
            messages, enable_thinking=False
        ):
            yield event
            

# =============================================================================
if __name__ == "__main__":
    async def test():
        print("Testing LLM Client V7.1 (Qwen3.5-4B)...")
        print("Make sure vLLM server is running on port 8011!")
        print()
        
        try:
            client = LLMDeploymentV7VLLMClient(
                base_url="http://localhost:8011/v1",
                model_name="Qwen3.5-4B",
            )
            
            health = await client.health_check()
            print(f"Health: {health}")
            
            if health["status"] == "healthy":
                # Test non-thinking mode (Qwen3.5)
                print("\n--- Non-Thinking Mode (Chat Completions API) ---")
                messages = [{"role": "user", "content": "What is 2+2?"}]
                response = await client.generate_async(
                    messages,
                    enable_thinking=False,
                )
                print(f"Response: {response}")
                
                print(f"\nMetrics: {client.get_metrics()}")
                print("Client test passed!")
            else:
                print("vLLM server not reachable. Start it first with:")
                print("   ./vllm_server_llm_v7.sh")
            
        except Exception as e:
            print(f"Test failed: {e}")
    
    asyncio.run(test())
