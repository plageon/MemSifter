import asyncio
import os
import random
import sqlite3
from typing import Dict, Optional, Any, Tuple

import aiosqlite
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class DBCache:
    def __init__(self, db_path: str):
        """
        初始化数据库缓存

        :param db_path: SQLite数据库文件路径
        """
        self.db_path = db_path
        # 初始化数据库（保持同步，因为只在初始化时执行一次）
        self.init_db()

        # 添加内存缓存，使用字典存储待写入的数据
        self.write_cache = {}  # 格式: {id: {"llm-resp": llm_response, "reasoning-content": reasoning_content}}
        self.cache_lock = asyncio.Lock()  # 用于保护缓存的锁

        # 新增：添加读写锁和写操作标记
        self.db_write_lock = asyncio.Lock()  # 数据库写锁
        self.is_writing = False  # 标记是否正在进行写操作

        # 标记批处理任务是否运行
        self.batch_writer_running = False
        asyncio.create_task(self.start_batch_writer())

    def init_db(self):
        """
        初始化数据库，创建extract-code表
        """
        # 确保数据库文件所在目录存在
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

        # 连接数据库并创建表
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建extract-code表，包含id、llm-resp和reasoning-content三个字段
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS "extract-code"
                       (
                           id
                           TEXT
                           PRIMARY
                           KEY,
                           "llm-resp"
                           TEXT
                           NOT
                           NULL,
                           "reasoning-content"
                           TEXT
                       )
                       ''')

        # 检查并添加新字段（如果表已存在但没有reasoning-content字段）
        cursor.execute("PRAGMA table_info('extract-code')")
        columns = [column[1] for column in cursor.fetchall()]
        if 'reasoning-content' not in columns:
            cursor.execute('''
                           ALTER TABLE "extract-code"
                           ADD COLUMN "reasoning-content" TEXT
                           ''')

        conn.commit()
        conn.close()

    def refresh_db(self):
        """
        刷新数据库，删除所有记录
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 删除所有记录
        cursor.execute('''
                       DELETE
                       FROM "extract-code"
                       ''')

        conn.commit()
        conn.close()

    async def get_llm_response(self, id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取LLM响应和推理内容（异步版本）

        :param id: 记录ID
        :return: 包含LLM响应和推理内容的字典，如果不存在则返回None
        """
        # 先检查内存缓存
        async with self.cache_lock:
            if id in self.write_cache:
                return self.write_cache[id]

        # 新增：等待写操作完成后再读取数据库
        while self.is_writing:
            await asyncio.sleep(0.1)  # 短暂等待后再次检查

        # 缓存未命中时查询数据库
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute('''
                                            SELECT "llm-resp", "reasoning-content"
                                            FROM "extract-code"
                                            WHERE id = ?
                                            ''', (id,))

                result = await cursor.fetchone()
                if result:
                    return {
                        "llm-resp": result[0],
                        "reasoning-content": result[1]
                    }
                return None
        except (aiosqlite.Error, sqlite3.Error) as e:
            logger.error(f"Database read error for ID {id}: {str(e)}")
            return None

    async def save_llm_response(self, id: str, llm_response: str, reasoning_content: Optional[str] = None):
        """
        保存LLM响应到内存缓存（异步版本）

        :param id: 记录ID
        :param llm_response: LLM的原始响应
        :param reasoning_content: LLM的推理内容
        """
        # 先保存到内存缓存，而不是直接写入数据库
        async with self.cache_lock:
            self.write_cache[id] = {
                "llm-resp": llm_response,
                "reasoning-content": reasoning_content,
            }

    async def exists(self, id: str) -> bool:
        """
        检查指定ID的记录是否存在（异步版本）

        :param id: 记录ID
        :return: 如果存在返回True，否则返回False
        """
        result = await self.get_llm_response(id)
        return result is not None

    async def start_batch_writer(self):
        """
        启动定期批量写入数据库的任务
        """
        self.batch_writer_running = True
        try:
            while self.batch_writer_running:
                await asyncio.sleep(1)  # 每隔1秒执行一次批量写入
                await self.batch_write_to_db()
        except asyncio.CancelledError:
            # 任务被取消时的处理
            pass
        finally:
            self.batch_writer_running = False
            # 确保在退出前将剩余的缓存数据写入数据库
            await self.batch_write_to_db()

    async def batch_write_to_db(self):
        """
        将内存缓存中的数据批量写入数据库
        """
        # 获取当前缓存中的数据并清空缓存
        batch_data = {}
        async with self.cache_lock:
            if self.write_cache:
                batch_data = self.write_cache.copy()
                self.write_cache = {}

        if not batch_data:
            return  # 没有数据需要写入

        # 新增：获取写锁并设置写标记
        async with self.db_write_lock:
            self.is_writing = True
            try:
                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    try:
                        async with aiosqlite.connect(self.db_path) as conn:
                            # 开始事务
                            await conn.execute('BEGIN TRANSACTION')

                            # 准备批量插入/更新的SQL语句
                            for id, data in batch_data.items():
                                await conn.execute('''
                                    INSERT OR REPLACE INTO "extract-code" (id, "llm-resp", "reasoning-content")
                                    VALUES (?, ?, ?)
                                ''', (id, data["llm-resp"], data["reasoning-content"]))

                            # 提交事务
                            await conn.commit()

                            # logger.debug(f"Successfully wrote {len(batch_data)} records to database")
                            break  # 成功写入后跳出循环
                    except (aiosqlite.Error, sqlite3.Error) as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error(
                                f"Failed to write batch data to database after {max_retries} retries: {str(e)}")
                            # 在最终失败的情况下，尝试将数据重新放回缓存
                            async with self.cache_lock:
                                for id, data in batch_data.items():
                                    if id not in self.write_cache:
                                        self.write_cache[id] = data
                        else:
                            # 生成随机等待时间（1-5秒）
                            wait_time = random.uniform(1, 5)
                            logger.warning(
                                f"Database write error, retrying in {wait_time:.2f}s (attempt {retry_count}/{max_retries}): {str(e)}")
                            await asyncio.sleep(wait_time)
            finally:
                self.is_writing = False  # 确保无论如何都会清除写标记

    async def stop_batch_writer(self):
        """
        停止批量写入任务
        """
        self.batch_writer_running = False
        # 确保在停止前将剩余数据写入数据库
        await self.batch_write_to_db()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=5, min=4, max=60))
async def get_chat_completion(client, model_config: Dict, message: list, request_semaphore: asyncio.Semaphore,
                              db_cache: DBCache = None, record_id: str = None, enable_thinking: bool = False) -> Tuple[str, Optional[str]]:
    try:
        async with request_semaphore:  # Use the shared semaphore to limit concurrent requests
            # loguru.logger.info(f"requesting model {model_config.model_name} with message: {message[-1]['content'][:50]}...\nWith params: {model_config.get('completions_params', {})})")
            # 如果提供了数据库缓存和记录ID，先尝试从数据库获取结果
            if db_cache and record_id:
                cached_response = await db_cache.get_llm_response(record_id)
                if cached_response:
                    if enable_thinking and cached_response["reasoning-content"]:
                        return cached_response["llm-resp"], cached_response["reasoning-content"]
                    # logger.info(f"Cache hit for record ID: {record_id}")
                    else:
                        return cached_response["llm-resp"]
                # else:
                # logger.info(f"Cache miss for record ID: {record_id}")

            response = await client.chat.completions.create(
                model=model_config["model_name"],
                messages=message,
                timeout=300,
                **model_config.get("completions_params", {}),  # temperature, max_tokens 等等
                # stream=True,
            )
            response_result = response.choices[0].message.content
            if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
                reasoning_content = response.choices[0].message.reasoning_content
            else:
                reasoning_content = None

            # 如果提供了数据库缓存和记录ID，将结果保存到数据库
            if db_cache and record_id and response_result:
                await db_cache.save_llm_response(record_id, response_result, reasoning_content)
                # logger.info(f"Saved response to cache for record ID: {record_id}")
            if enable_thinking and reasoning_content:
                return response_result, reasoning_content
            else:
                return response_result
    except Exception as e:
        # The @retry decorator will handle this exception
        print(f"Error in get_chat_completion: {type(e).__name__} - {str(e)}. Retrying...")
        raise