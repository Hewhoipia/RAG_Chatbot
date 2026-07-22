"""
Description: Ingestion and retrieval pipeline utilizing BGE Embedding Models
"""

import os
import logging
from config import config

logger = logging.getLogger(__name__)

def ingest_documents(data_dir: str = None):
    """
    Ingests all files (.pdf, .docx, etc.) in the data directory into Qdrant vector database
    """
    target_dir = data_dir or config.retrieval.data_dir
    if config.demo_mode:
        logger.info("[Ingestion] Running in DEMO MODE. Skipping document database ingestion.")
        return

    try:
        from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient

        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            logger.info(f"Created empty data directory: {target_dir}")
            return

        # Read directory files
        reader = SimpleDirectoryReader(input_dir=target_dir)
        documents = reader.load_data()
        if not documents:
            logger.info("No documents found in target folder. Nothing to ingest.")
            return

        # Setup vector store connection
        client = QdrantClient(host=config.qdrant.host, port=config.qdrant.port)
        vector_store = QdrantVectorStore(client=client, collection_name=config.qdrant.collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Parse and embed
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context
        )
        logger.info(f"Ingested {len(documents)} documents successfully into Qdrant collection '{config.qdrant.collection_name}'")
    except Exception as e:
        logger.error(f"Ingestion pipeline failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_documents()
