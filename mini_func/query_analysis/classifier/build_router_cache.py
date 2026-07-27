# =============================================================================
# FILE: build_router_cache.py
# =============================================================================
#
# GIỚI THIỆU (Introduction):
# --------------------------
# Script offline chạy 1 lần để huấn luyện và tạo file classifier_router.pkl.
# Đọc câu mẫu từ prototypes_vi.py → Embedding bằng BGE-M3 (qua vLLM) →
# Áp dụng PCA-Whitening khử nhiễu Anisotropy → Huấn luyện Logistic Regression Head
# → Lưu toàn bộ pipeline ra file .pkl để phục vụ định tuyến siêu tốc.
#
# MỤC ĐÍCH (Purpose):
# -------------------
# 1. Giải quyết triệt để sự chồng lấn ngữ nghĩa của các nhóm câu mẫu.
# 2. Tạo không gian vector đẳng hướng (Isotropic) thông qua PCA-Whitening.
# 3. Lưu cache router ra file .pkl để runtime load trong <1ms mà không cần huấn luyện lại.
#
# CÁCH SỬ DỤNG (Usage):
# ----------------------
# cd d:\Code\Chatbot\Production\tools\query_analysis
# python -m classifier.build_router_cache

# LƯU Ý (Notes):
# ---------------
# - Chỉ cần chạy lại khi THÊM/SỬA câu mẫu trong prototypes_vi.py
# - File output: classifier/classifier_router.pkl
# - File metadata: classifier/classifier_router_meta.json (thứ tự labels)
# - Cần model đã tải sẵn tại đường dẫn EMBEDDING_DIR
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
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()

# Thử import scikit-learn
try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
except ImportError:
    logger.error("❌ Thiếu thư viện scikit-learn.")
    raise

# Import config_v7
from config_v7 import config

async def build_centroids():
    """
    Giới thiệu: Tập lệnh tiền xử lý huấn luyện mô hình phân loại tuyến tính trên không gian vector đã làm trắng (Whitened).
    
    Mục đích sử dụng: Tạo file Cache chứa tham số PCA và Logistic Regression giúp định tuyến chính xác với Latency <1ms.
    """
    logger.info(f"=== BUILD ROUTER CLASSIFIER CACHE ===")

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

    # --- Bước 4: Encode toàn bộ dữ liệu mẫu ---
    all_embeddings = []
    all_labels = []

    for group_name, sentences in groups.items():
        logger.info(f"Encoding {group_name} ({len(sentences)} sentences) qua vLLM...")

        try:
            response = await vllm_client.embeddings.create(
                model=model_name,
                input=sentences
            )
            embeddings = np.array([item.embedding for item in response.data])
            all_embeddings.append(embeddings)
            # Gán nhãn dạng chuỗi (string) trực tiếp để classifier lưu trữ tự nhiên
            all_labels.extend([group_name] * len(sentences))
            logger.info(f"   ✓ Đã nhận diện và encode thành công {len(embeddings)} vectors.")
        except Exception as e:
            logger.error(f"   ❌ Lỗi khi gọi vLLM API cho nhóm {group_name}: {e}")
            raise e

    # Gộp toàn bộ vector nhúng và nhãn thành 2 ma trận lớn để huấn luyện
    X = np.vstack(all_embeddings)  # Shape: (Tổng số câu mẫu, Kích thước vector 1024)
    y = np.array(all_labels)       # Shape: (Tổng số câu mẫu,)
    
    logger.info(f"Tổng số mẫu nạp vào huấn luyện: {X.shape[0]} mẫu, số chiều vector: {X.shape[1]}")

    # --- Bước 5: Thực hiện PCA-Whitening để bẻ gãy ranh giới chồng lấn (Anisotropy) ---
    logger.info("Đang tính toán ma trận xoay PCA-Whitening...")
    # Số lượng component tối đa dựa trên số mẫu hiện có để tránh phân rã vô hạn
    n_components = min(X.shape[0] - 1, X.shape[1]) 
    
    # whiten=True sẽ tự động thực hiện phép chuẩn hóa: x_white = x_rot / sqrt(lambda)
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_white = pca.fit_transform(X)
    logger.info(f"   ✓ Không gian vector đã được chuẩn hóa về dạng Đẳng hướng. Shape mới: {X_white.shape}")

    # --- Bước 6: Huấn luyện bộ phân loại tối ưu Linear Probe (Logistic Regression) ---
    logger.info("Đang huấn luyện bộ phân lớp tuyến tính Logistic Regression...")
    # Sử dụng multinomial cho bài toán phân loại đa lớp
    clf = LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial', random_state=42)
    clf.fit(X_white, y)
    logger.info(f"   ✓ Huấn luyện classifier hoàn tất. Lớp phân loại: {clf.classes_}")

    # --- Bước 7: Lưu các đối tượng ra file Cache ---
    output_dir = os.path.dirname(__file__)
    pkl_path = os.path.join(output_dir, "classifier_router.pkl")
    meta_path = os.path.join(output_dir, "classifier_router_meta.json")

    # Đóng gói cả PCA và Logistic Regression vào 1 tệp pickle duy nhất
    router_assets = {
        "pca": pca,
        "classifier": clf,
        "classes": clf.classes_.tolist()
    }
    
    with open(pkl_path, "wb") as f:
        pickle.dump(router_assets, f)
    logger.info(f"   ✓ Đã lưu file mô hình Router: {pkl_path}")

    # Lưu metadata để tiện theo dõi/giám sát hệ thống
    metadata = {
        "labels": clf.classes_.tolist(),
        "embedding_model": model_name,
        "num_sentences": {name: len(sents) for name, sents in groups.items()},
        "embedding_dim": int(X.shape[1]),
        "pca_components": int(n_components),
        "total_samples": int(X.shape[0]),
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"   ✓ Đã lưu metadata theo dõi: {meta_path}")

    logger.info(f"=== BUILD COMPLETE ===")
    logger.info(f"Hệ thống định tuyến thông minh đã sẵn sàng hoạt động!")

    return router_assets


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Tạo cache Router thông minh bằng PCA-Whitening và Logistic Regression")
    args = parser.parse_args()

    asyncio.run(build_centroids())