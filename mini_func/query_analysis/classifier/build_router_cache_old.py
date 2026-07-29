# =============================================================================
# FILE: build_router_cache.py
# =============================================================================
#
# GIỚI THIỆU (Introduction):
# --------------------------
# Script offline chạy 1 lần để tạo file classifier_centroids.npy.
# Đọc câu mẫu từ prototypes_vi.py → Embedding bằng BGE-M3 → Tính
# trung bình (Mean Pooling) → Lưu 4 Centroid vectors ra file .npy.
#
# MỤC ĐÍCH (Purpose):
# -------------------
# 1. Pre-compute embedding vectors cho Semantic Router
# 2. Lưu cache ra file .npy để runtime load trong <1ms
# 3. Không cần re-embedding mỗi khi server khởi động lại
#
# CÁCH SỬ DỤNG (Usage):
# ----------------------
# cd d:\Code\Chatbot\Production\tools\query_analysis
# python -m classifier.build_router_cache
#
# Hoặc:
# python classifier/build_router_cache.py
#
# LƯU Ý (Notes):
# ---------------
# - Chỉ cần chạy lại khi THÊM/SỬA câu mẫu trong prototypes_vi.py
# - File output: classifier/classifier_centroids.npy (4 vectors, mỗi vector 1024 dims)
# - File metadata: classifier/classifier_centroids_meta.json (thứ tự labels)
# - Cần BGE-M3 model đã tải sẵn tại đường dẫn EMBEDDING_DIR
# =============================================================================

import os
import sys

# Cho phép chạy script trực tiếp từ bất kỳ thư mục nào mà không bị lỗi ModuleNotFound
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()

# Import config
from config import config

async def build_centroids():
    """
    Giới thiệu: Tập lệnh tiền xử lý nén đo đạc 4 Centroid vectors chuẩn làm mốc tọa độ không gian mẫu.
    
    Mục đích sử dụng: Tạo file Cache định hình trước Vector nhúng của lưới màng Semantic Router giúp kéo Latency mốc Runtime xuống <1ms siêu tốc.
    
    Cách thức hoạt động:
    - B1: Trích xuất câu văn mẫu của 4 Nhóm nhãn (Chit-chat, Simple, Complex, Clarify) từ prototypes_vi.py.
    - B2: Gọi vLLM API Embedding để biến các câu thành Vector.
    - B3: Mean pooling trung bình cộng từng nhóm để tạo 1 Centroid cho mỗi nhóm.
    - B4: Xếp chồng Array và ghi đĩa file `.npy` kèm Tracking Info `meta.json`.
    
    Đầu vào - Đầu ra của hàm đó:
    - Đầu vào: Lấy dữ liệu cấu hình từ file constants/env.
    - Đầu ra: Lưu file .npy.
    """
    logger.info(f"=== BUILD ROUTER CACHE ===")

    # --- Bước 2: Load câu mẫu từ prototypes_vi.py ---
    try:
        # Package mode: khi import từ mini_function.query_analysis.classifier
        from .keywords.prototypes_vi import (
            CHITCHAT_SENTENCES,
            SIMPLE_SENTENCES,
            COMPLEX_SENTENCES,
            CLARIFY_SENTENCES,
        )
    except ImportError:
        # Standalone mode: khi chạy python -m classifier.build_router_cache
        from classifier.keywords.prototypes_vi import (
            CHITCHAT_SENTENCES,
            SIMPLE_SENTENCES,
            COMPLEX_SENTENCES,
            CLARIFY_SENTENCES,
        )

    groups = {
        "chit-chat": CHITCHAT_SENTENCES,
        "simple": SIMPLE_SENTENCES,
        "complex": COMPLEX_SENTENCES,
        "clarify": CLARIFY_SENTENCES,
    }

    # Hiển thị thông tin
    for group_name, sentences in groups.items():
        logger.info(f"   {group_name}: {len(sentences)} câu mẫu")

    # --- Bước 3: Khởi tạo AsyncOpenAI client kết nối vLLM ---
    from openai import AsyncOpenAI
    
    emb_url = config.vllm.base_url_embedding if config else "http://localhost:8012/v1"
    emb_model = config.vllm.model_name_embedding if config else "Qwen3-Embedding-4B"
    emb_key = config.vllm.api_key_embedding if config else "EMPTY"
    
    logger.info(f"Khởi tạo AsyncOpenAI client kết nối vLLM ({emb_url})...")
    vllm_client = AsyncOpenAI(
        base_url=emb_url,
        api_key=emb_key
    )
    
    # Lấy tên model thực tế đang chạy trên vLLM
    model_name = emb_model

    try:
        models = await vllm_client.models.list()
        model_name = models.data[0].id
        logger.info(f"   ✓ Đã kết nối vLLM. Sử dụng model: {model_name}")
    except Exception as e:
        logger.warning(f"   ⚠ Lỗi lấy danh sách model từ vLLM: {e}. Fallback dùng: {model_name}")

    # --- Bước 4: Encode và tính Centroid cho mỗi nhóm ---
    centroids = []
    labels = []

    for group_name, sentences in groups.items():
        logger.info(f"Encoding {group_name} ({len(sentences)} sentences) qua vLLM...")

        # Encode tất cả câu trong nhóm bằng vLLM API
        try:
            response = await vllm_client.embeddings.create(
                model=model_name,
                input=sentences
            )
            embeddings = np.array([item.embedding for item in response.data])
        except Exception as e:
            logger.error(f"   ❌ Lỗi khi gọi vLLM API cho nhóm {group_name}: {e}")
            raise e

        # Mean Pooling: Tính trung bình tất cả vectors → 1 Centroid
        centroid = np.mean(embeddings, axis=0)
        centroids.append(centroid)
        labels.append(group_name)

        logger.info(f"   ✓ {group_name} centroid: shape={centroid.shape}, norm={np.linalg.norm(centroid):.4f}")

    # --- Bước 5: Stack và lưu ra file ---
    centroids_array = np.stack(centroids)  # Shape: (4, embedding_dim)
    logger.info(f"Final centroids array shape: {centroids_array.shape}")

    # Đường dẫn output
    output_dir = os.path.dirname(__file__)
    npy_path = os.path.join(output_dir, "classifier_centroids.npy")
    meta_path = os.path.join(output_dir, "classifier_centroids_meta.json")

    # Lưu vectors
    np.save(npy_path, centroids_array)
    logger.info(f"   ✓ Saved centroids: {npy_path}")

    # Lưu metadata (thứ tự labels để mapping đúng index)
    metadata = {
        "labels": labels,
        "embedding_model": model_name,
        "num_sentences": {name: len(sents) for name, sents in groups.items()},
        "embedding_dim": int(centroids_array.shape[1]),
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"   ✓ Saved metadata: {meta_path}")

    logger.info(f"=== BUILD COMPLETE ===")
    logger.info(f"Bạn có thể sử dụng HybridClassifier ngay bây giờ!")

    return centroids_array


# =============================================================================
# ENTRY POINT (Chạy trực tiếp: python -m classifier.build_router_cache)
# =============================================================================
if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Tạo cache centroids cho Semantic Router sử dụng vLLM API")
    args = parser.parse_args()

    asyncio.run(build_centroids())
