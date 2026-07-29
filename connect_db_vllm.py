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
  * queue_conversation(): Simplified wrapper to save conversation history.
  * save_query_type(): Updates an existing chat history record with query type and domain.
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
        """
        Initializes PostgreSQL connection pool if not in demo mode.

        Args:
            None

        Returns:
            None
        """
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
                        domain TEXT,
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
        """
        Closes connection pool.

        Args:
            None

        Returns:
            None
        """
        if self.pool:
            await self.pool.close()
            logger.info("Real PostgreSQL connection pool closed.")

    async def health_check(self) -> Dict[str, Any]:
        """
        Checks status of database connection.

        Args:
            None

        Returns:
            Dict[str, Any]: Dictionary containing healthy status, mode, and other connection stats.
        """
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
        query_type: Optional[str],
        cache_hit: bool,
        thinking_time_sec: float,
        total_time_sec: float
    ):
        """
        Saves chat history record to PostgreSQL or local JSON file (demo).

        Args:
            conversation_id (str): Unique conversation identifier.
            user_ip (str): IP address of the user.
            user_name (str): Name of the user.
            user_question (str): The question asked by the user.
            llm_answer (str): The response generated by the LLM.
            thinking_content (Optional[str]): The thinking process of the model.
            sources (Optional[str]): The sources used for the response.
            query_type (Optional[str]): The categorized type of the query.
            cache_hit (bool): Whether the response was served from cache.
            thinking_time_sec (float): Time taken for the model to think.
            total_time_sec (float): Total response time.

        Returns:
            None
        """
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
        """
        Saves user feedback to PostgreSQL or local JSON file (demo).

        Args:
            conversation_id (str): Unique conversation identifier.
            answer_satisfaction (Optional[int]): Satisfaction rating (e.g. 1-5).
            quality_rating (Optional[int]): Quality rating (e.g. 1-5).
            speed_rating (Optional[int]): Speed rating (e.g. 1-5).
            issues (Optional[str]): Issues encountered by the user.
            other_opinions (Optional[str]): Additional comments or feedback.

        Returns:
            None
        """
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

    async def queue_conversation(
        self,
        conversation_id: str,
        question: str,
        response: str,
        response_time: Any,
        ip: str
    ):
        """
        Simplified wrapper around save_chat_history for queuing conversations.

        Args:
            conversation_id (str): Unique conversation identifier.
            question (str): The question asked by the user.
            response (str): The response generated by the LLM.
            response_time (Any): A timedelta object representing the response time.
            ip (str): IP address of the user.

        Returns:
            None
        """
        total_time_sec = response_time.total_seconds() if hasattr(response_time, 'total_seconds') else float(response_time)
        await self.save_chat_history(
            conversation_id=conversation_id,
            user_ip=ip,
            user_name='',
            user_question=question,
            llm_answer=response,
            thinking_content=None,
            sources=None,
            query_type=None,
            cache_hit=False,
            thinking_time_sec=0.0,
            total_time_sec=total_time_sec
        )

    async def save_query_type(
        self,
        conversation_id: str,
        query_type: str,
        domain: Optional[str] = None
    ):
        """
        Updates an existing chat history record to set query_type and domain.

        Args:
            conversation_id (str): Unique conversation identifier.
            query_type (str): The categorized type of the query.
            domain (Optional[str]): The domain associated with the query.

        Returns:
            None
        """
        if self.demo_mode:
            logger.info(f"[Demo DB] Updated query type for {conversation_id} to {query_type}, domain {domain}")
            return
            
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE chat_history 
                    SET query_type = $1, domain = $2
                    WHERE conversation_id = $3
                    """,
                    query_type, domain, conversation_id
                )
            logger.info(f"Updated query_type and domain for conversation_id={conversation_id}")
        except Exception as e:
            logger.error(f"Failed to update query_type and domain: {e}")