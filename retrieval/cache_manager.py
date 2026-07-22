"""
Description: Qdrant index loader and static cache loader for RAG
"""

import logging
from config import config

logger = logging.getLogger(__name__)

try:
    import qdrant_client
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant_client not installed.")

class QdrantManager:
    """
    Manages Qdrant Client connection and collection status
    """
    def __init__(self):
        self.client = None
        self.host = config.qdrant.host
        self.port = config.qdrant.port
        self.collection_name = config.qdrant.collection_name
        self.demo_mode = config.demo_mode or not QDRANT_AVAILABLE
        
    def get_client(self):
        """Returns active Qdrant Client"""
        if self.demo_mode:
            return None
        if not self.client:
            try:
                self.client = qdrant_client.QdrantClient(host=self.host, port=self.port)
            except Exception as e:
                logger.error(f"Qdrant Client initialization failed: {e}")
        return self.client

def load_vector_index_from_qdrant():
    """
    Loads LlamaIndex VectorStoreIndex wrapper connected to Qdrant collection
    """
    if config.demo_mode or not QDRANT_AVAILABLE:
        return None
    try:
        from llama_index.core import VectorStoreIndex
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        
        client = QdrantClient(host=config.qdrant.host, port=config.qdrant.port)
        vector_store = QdrantVectorStore(client=client, collection_name=config.qdrant.collection_name)
        return VectorStoreIndex.from_vector_store(vector_store)
    except Exception as e:
        logger.error(f"Failed to load VectorStoreIndex from Qdrant: {e}")
        return None

def load_data_cache():
    """
    Loads static mock or fallback nodes from cache files
    """
    return []
