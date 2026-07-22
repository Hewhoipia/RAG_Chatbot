"""
FILE: connect_db_vllm.py
DESCRIPTION: Database connection manager for RAG supporting both real PostgreSQL and offline Demo Mode.

IMPLEMENTED CLASSES & FUNCTIONS:
- DatabaseManager: Manages PostgreSQL pool connection lifecycle or falls back to local logging.
  * init_pool(): Bootstraps the connection pool and creates chat_history/detailed_feedback schemas. Falls back to demo mode on connection failures.
  * close(): Disposes of the active connection pool.
  * health_check(): Performs ping SELECT 1 checks or reports local demo status.
  * save_chat_history(): Saves conversation metadata, user inputs, outputs, and thinking times to database or demo_logs/chat_history_demo.jsonl.
  * save_detailed_feedback(): Logs user satisfaction, ratings, and textual reviews to database or demo_logs/detailed_feedback_demo.jsonl.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
from config import config

logger = logging.getLogger(__name__)

# Try to import asyncpg for real Postgres connection
try:
    import asyncpg
    ASYNC_PG_AVAILABLE = True
except ImportError:
    ASYNC_PG_AVAILABLE = False
    logger.warning("asyncpg package not found. Real database operations will be unavailable.")

class DatabaseManager:
    """
    Manages connection pool to PostgreSQL or operates in Demo Mode
    without a database server.
    """
    def __init__(self):
        self.pool = None
        self.demo_mode = config.demo_mode or (not ASYNC_PG_AVAILABLE) or (not config.database.enabled)
        if self.demo_mode:
            logger.info("DatabaseManager initialized in DEMO MODE.")
        else:
            logger.info("DatabaseManager initialized in PRODUCTION/REAL DATABASE MODE.")
            
    async def init_pool(self):
        """Initializes PostgreSQL connection pool if not in demo mode"""
        if self.demo_mode:
            # Create files for logging locally in demo mode
            os.makedirs("demo_logs", exist_ok=True)
            return

        try:
            self.pool = await asyncpg.create_pool(
                dsn=config.database.dsn,
                min_size=config.database.min_size,
                max_size=config.database.max_size,
                command_timeout=60.0
            )
            # Create tables if they do not exist
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id SERIAL PRIMARY KEY,
                        conversation_id VARCHAR(255) UNIQUE NOT NULL,
                        user_ip VARCHAR(45),
                        user_name VARCHAR(255),
                        user_question TEXT,
                        llm_answer TEXT,
                        thinking_content TEXT,
                        sources TEXT,
                        query_type VARCHAR(50),
                        cache_hit BOOLEAN,
                        thinking_time_sec REAL,
                        total_time_sec REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS detailed_feedback (
                        id SERIAL PRIMARY KEY,
                        conversation_id VARCHAR(255) NOT NULL,
                        answer_satisfaction INT,
                        quality_rating INT,
                        speed_rating INT,
                        issues TEXT,
                        other_opinions TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            logger.info("Real PostgreSQL connection pool and tables initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}. Switching to DEMO MODE.")
            self.demo_mode = True
            os.makedirs("demo_logs", exist_ok=True)

    async def close(self):
        """Closes connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Real PostgreSQL connection pool closed.")

    async def health_check(self) -> Dict[str, Any]:
        """Checks status of database connection"""
        if self.demo_mode:
            return {"healthy": True, "mode": "demo_mode", "status": "running locally without DB"}

        if not self.pool:
            return {"healthy": False, "error": "Connection pool not initialized"}

        try:
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
            return {
                "healthy": True,
                "mode": "production",
                "pool_size": self.pool.get_size(),
                "free_connections": self.pool.get_idle_size()
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def save_chat_history(
        self,
        conversation_id: str,
        user_ip: str,
        user_name: str,
        user_question: str,
        llm_answer: str,
        thinking_content: Optional[str],
        sources: Optional[str],
        query_type: str,
        cache_hit: bool,
        thinking_time_sec: float,
        total_time_sec: float
    ):
        """Saves chat history record to PostgreSQL or local JSON file (demo)"""
        if self.demo_mode:
            # Save to demo JSON file
            record = {
                "timestamp": datetime.now().isoformat(),
                "conversation_id": conversation_id,
                "user_ip": user_ip,
                "user_name": user_name,
                "user_question": user_question,
                "llm_answer": llm_answer,
                "thinking_content": thinking_content,
                "sources": sources,
                "query_type": query_type,
                "cache_hit": cache_hit,
                "thinking_time_sec": thinking_time_sec,
                "total_time_sec": total_time_sec
            }
            try:
                log_path = "demo_logs/chat_history_demo.jsonl"
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                logger.info(f"[Demo DB] Saved chat history locally for conversation_id={conversation_id}")
            except Exception as e:
                logger.error(f"[Demo DB] Failed to save chat history locally: {e}")
            return

        # Real DB mode
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO chat_history (
                        conversation_id, user_ip, user_name, user_question, 
                        llm_answer, thinking_content, sources, query_type, 
                        cache_hit, thinking_time_sec, total_time_sec
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (conversation_id) DO UPDATE SET
                        llm_answer = EXCLUDED.llm_answer,
                        thinking_content = EXCLUDED.thinking_content,
                        sources = EXCLUDED.sources,
                        query_type = EXCLUDED.query_type,
                        cache_hit = EXCLUDED.cache_hit,
                        thinking_time_sec = EXCLUDED.thinking_time_sec,
                        total_time_sec = EXCLUDED.total_time_sec
                    """,
                    conversation_id, user_ip, user_name, user_question,
                    llm_answer, thinking_content, sources, query_type,
                    cache_hit, thinking_time_sec, total_time_sec
                )
            logger.info(f"Saved chat history to PostgreSQL for conversation_id={conversation_id}")
        except Exception as e:
            logger.error(f"Failed to save chat history to PostgreSQL: {e}")

    async def save_detailed_feedback(
        self,
        conversation_id: str,
        answer_satisfaction: Optional[int],
        quality_rating: Optional[int],
        speed_rating: Optional[int],
        issues: Optional[str],
        other_opinions: Optional[str]
    ):
        """Saves user feedback to PostgreSQL or local JSON file (demo)"""
        if self.demo_mode:
            record = {
                "timestamp": datetime.now().isoformat(),
                "conversation_id": conversation_id,
                "answer_satisfaction": answer_satisfaction,
                "quality_rating": quality_rating,
                "speed_rating": speed_rating,
                "issues": issues,
                "other_opinions": other_opinions
            }
            try:
                log_path = "demo_logs/detailed_feedback_demo.jsonl"
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                logger.info(f"[Demo DB] Saved detailed feedback locally for conversation_id={conversation_id}")
            except Exception as e:
                logger.error(f"[Demo DB] Failed to save detailed feedback locally: {e}")
            return

        # Real DB mode
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO detailed_feedback (
                        conversation_id, answer_satisfaction, quality_rating,
                        speed_rating, issues, other_opinions
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    conversation_id, answer_satisfaction, quality_rating,
                    speed_rating, issues, other_opinions
                )
            logger.info(f"Saved detailed feedback to PostgreSQL for conversation_id={conversation_id}")
        except Exception as e:
            logger.error(f"Failed to save detailed feedback to PostgreSQL: {e}")