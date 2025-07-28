import asyncio
import logging
import os
from typing import Dict, List, Optional, Tuple, Set
from contextlib import asynccontextmanager
import time
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from telethon import TelegramClient
from telethon.errors import FloodWaitError, RPCError, AuthKeyPermEmptyError, SessionPasswordNeededError
from telethon import functions, types
from uuid import uuid4
import aiohttp
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logger.info("Using uvloop for better performance")
except ImportError:
    logger.warning("uvloop not available, using default event loop")

class BotInfoModel(BaseModel):
    first_name: str
    id: int
    username: Optional[str] = None

class ChatModel(BaseModel):
    id: int
    members_count: Optional[int] = None
    title: str
    type: str
    username: Optional[str] = None

class UserModel(BaseModel):
    id: int
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    username: Optional[str] = None
    is_premium: bool = False

class BotDataResponse(BaseModel):
    bot_info: BotInfoModel
    chats: List[ChatModel]
    users: List[UserModel]
    total_chats: int = Field(..., description="Total number of chats")
    total_users: int = Field(..., description="Total number of users")
    processing_time: float = Field(..., description="Processing time in seconds")

class ClientManager:
    def __init__(self):
        self.clients: Dict[str, TelegramClient] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        self.global_lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._session: Optional[aiohttp.ClientSession] = None

    async def get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def get_client(self, bot_token: str, api_id: int, api_hash: str) -> TelegramClient:
        async with self.global_lock:
            if bot_token not in self.locks:
                self.locks[bot_token] = asyncio.Lock()

        async with self.locks[bot_token]:
            if bot_token in self.clients:
                client = self.clients[bot_token]
                try:
                    if client.is_connected():
                        return client
                    else:
                        await client.disconnect()
                        del self.clients[bot_token]
                except Exception as e:
                    logger.warning(f"Client connection issue: {e}")
                    if bot_token in self.clients:
                        del self.clients[bot_token]

            session_name = f"bot_{uuid4().hex[:8]}"
            client = TelegramClient(
                session=session_name,
                api_id=api_id,
                api_hash=api_hash,
                base_logger=logger,
                flood_sleep_threshold=0
            )

            try:
                await asyncio.wait_for(client.start(bot_token=bot_token), timeout=30.0)
                async with self.global_lock:
                    self.clients[bot_token] = client
                logger.info(f"Client created successfully for token: {bot_token[:10]}...")
                return client
            except Exception as e:
                logger.error(f"Failed to create client: {e}")
                try:
                    await client.disconnect()
                except:
                    pass
                raise HTTPException(status_code=400, detail=f"Failed to connect to Telegram: {str(e)}")

    async def cleanup_client(self, bot_token: str):
        try:
            async with self.global_lock:
                if bot_token in self.clients:
                    client = self.clients[bot_token]
                    try:
                        if client.is_connected():
                            await asyncio.wait_for(client.disconnect(), timeout=10.0)
                    except Exception as e:
                        logger.warning(f"Error stopping client: {e}")
                    finally:
                        del self.clients[bot_token]
                if bot_token in self.locks:
                    del self.locks[bot_token]
                logger.info(f"Client cleaned up for token: {bot_token[:10]}...")
        except Exception as e:
            logger.error(f"Error cleaning up client {bot_token[:10]}...: {e}")

    async def shutdown(self):
        logger.info("Shutting down client manager...")
        try:
            async with self.global_lock:
                for bot_token in list(self.clients.keys()):
                    await self.cleanup_client(bot_token)
            if self._session and not self._session.closed:
                await self._session.close()
            self.executor.shutdown(wait=True)
            logger.info("Client manager shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

client_manager = ClientManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Enhanced Telegram Users API...")
    try:
        yield
    finally:
        await client_manager.shutdown()
        logger.info("API shutdown complete")

app = FastAPI(
    title="Enhanced Telegram Users API",
    description="High-performance async API for Telegram bot data with improved concurrency",
    version="2.2.0",
    lifespan=lifespan
)

def normalize_chat_type(raw_type: str) -> str:
    type_map = {
        "chat": "group",
        "channel": "channel", 
        "chatforbidden": "group",
        "channelforbidden": "channel",
        "user": "private"
    }
    return type_map.get(raw_type.lower(), raw_type.lower())

def merge_chat_data(existing: Optional[ChatModel], new: ChatModel) -> ChatModel:
    if not existing:
        return new
    return ChatModel(
        id=existing.id,
        members_count=new.members_count if new.members_count is not None else existing.members_count,
        title=new.title if new.title and new.title != "Unknown" else existing.title,
        type=new.type,
        username=new.username if new.username else existing.username
    )

async def process_entities_batch(entities: List, entity_type: str) -> Tuple[Dict[int, UserModel], Dict[int, ChatModel]]:
    """Process a batch of entities efficiently"""
    batch_users = {}
    batch_chats = {}
    
    for entity in entities:
        try:
            if entity_type == "users" and hasattr(entity, 'id'):
                if not entity.deleted and not getattr(entity, 'fake', False):
                    batch_users[entity.id] = UserModel(
                        id=entity.id,
                        first_name=getattr(entity, 'first_name', None),
                        last_name=getattr(entity, 'last_name', None),
                        username=getattr(entity, 'username', None),
                        is_premium=getattr(entity, 'premium', False)
                    )
            
            elif entity_type == "chats" and hasattr(entity, 'id'):
                chat_class_name = entity.__class__.__name__.lower()
                if chat_class_name not in ["chatforbidden", "channelforbidden"]:
                    chat_type = normalize_chat_type(chat_class_name)
                    title = (getattr(entity, 'title', None) or 
                            getattr(entity, 'first_name', None) or "Unknown")
                    
                    batch_chats[entity.id] = ChatModel(
                        id=entity.id,
                        members_count=getattr(entity, 'participants_count', None),
                        title=title,
                        type=chat_type,
                        username=getattr(entity, 'username', None)
                    )
        except Exception as e:
            logger.debug(f"Error processing {entity_type} entity: {e}")
            continue
    
    return batch_users, batch_chats

async def get_chats_and_users_optimized(client: TelegramClient) -> Tuple[List[ChatModel], List[UserModel]]:
    """Optimized version with better batching and error handling"""
    all_chats: Dict[int, ChatModel] = {}
    all_users: Dict[int, UserModel] = {}
    processed_user_ids: Set[int] = set()
    processed_chat_ids: Set[int] = set()
    
    # Get initial state with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            state = await asyncio.wait_for(client.get_state(), timeout=15.0)
            custom_pts = max(1, state.pts - 1000)  # Start slightly behind for safety
            custom_date = state.date - timedelta(days=1)  # Go back 1 day
            custom_qts = max(1, state.qts - 100)
            break
        except Exception as e:
            if attempt == max_retries - 1:
                logger.warning(f"Failed to get initial state after {max_retries} attempts: {e}")
                custom_pts = 1
                custom_date = datetime.now() - timedelta(days=7)
                custom_qts = 1
            else:
                await asyncio.sleep(2 ** attempt)
                continue

    start_time = time.time()
    max_duration = 600  # Increased to 10 minutes for large datasets
    batch_count = 0
    consecutive_empty_batches = 0
    max_empty_batches = 5
    
    logger.info(f"Starting optimized data collection with pts={custom_pts}, qts={custom_qts}")
    
    try:
        while time.time() - start_time < max_duration:
            batch_start = time.time()
            
            try:
                # Optimized parameters for large datasets
                diff = await asyncio.wait_for(
                    client(functions.updates.GetDifferenceRequest(
                        pts=custom_pts,
                        date=custom_date,
                        qts=custom_qts,
                        pts_limit=50000,      # Increased for better performance
                        pts_total_limit=10000000,  # Much higher limit
                        qts_limit=50000       # Increased for better performance
                    )),
                    timeout=120.0  # Increased timeout for large batches
                )
                
                batch_users_found = 0
                batch_chats_found = 0
                
                # Process users in batch
                if hasattr(diff, 'users') and diff.users:
                    batch_users, _ = await process_entities_batch(diff.users, "users")
                    for user_id, user in batch_users.items():
                        if user_id not in processed_user_ids:
                            all_users[user_id] = user
                            processed_user_ids.add(user_id)
                            batch_users_found += 1
                
                # Process chats in batch
                if hasattr(diff, 'chats') and diff.chats:
                    _, batch_chats = await process_entities_batch(diff.chats, "chats")
                    for chat_id, chat in batch_chats.items():
                        if chat_id not in processed_chat_ids:
                            all_chats[chat_id] = merge_chat_data(all_chats.get(chat_id), chat)
                            processed_chat_ids.add(chat_id)
                            batch_chats_found += 1
                
                # Process new messages for additional entities
                if hasattr(diff, 'new_messages') and diff.new_messages:
                    for message in diff.new_messages[:1000]:  # Limit message processing
                        try:
                            if hasattr(message, 'from_id') and message.from_id:
                                user_id = getattr(message.from_id, 'user_id', None)
                                if user_id and user_id not in processed_user_ids:
                                    # We'll get user details in the next difference call
                                    processed_user_ids.add(user_id)
                            
                            if hasattr(message, 'peer_id') and message.peer_id:
                                chat_id = None
                                if hasattr(message.peer_id, 'chat_id'):
                                    chat_id = message.peer_id.chat_id
                                elif hasattr(message.peer_id, 'channel_id'):
                                    chat_id = message.peer_id.channel_id
                                
                                if chat_id and chat_id not in processed_chat_ids:
                                    processed_chat_ids.add(chat_id)
                        except Exception as e:
                            logger.debug(f"Error processing message: {e}")
                            continue
                
                batch_time = time.time() - batch_start
                
                if batch_users_found > 0 or batch_chats_found > 0:
                    consecutive_empty_batches = 0
                    logger.info(f"Batch {batch_count}: Users +{batch_users_found}, Chats +{batch_chats_found} "
                              f"(Total: {len(all_users)} users, {len(all_chats)} chats) - {batch_time:.2f}s")
                else:
                    consecutive_empty_batches += 1
                    logger.debug(f"Batch {batch_count}: Empty batch ({consecutive_empty_batches}/{max_empty_batches})")
                
                batch_count += 1
                
                # Update state for next iteration
                if isinstance(diff, types.updates.DifferenceSlice):
                    custom_pts = diff.intermediate_state.pts
                    custom_date = diff.intermediate_state.date
                    custom_qts = diff.intermediate_state.qts
                elif isinstance(diff, types.updates.Difference):
                    logger.info("Reached end of differences")
                    break
                elif isinstance(diff, types.updates.DifferenceEmpty):
                    logger.info("No more differences available")
                    break
                elif isinstance(diff, types.updates.DifferenceTooLong):
                    logger.warning("Difference too long, adjusting parameters")
                    custom_pts = diff.pts
                    await asyncio.sleep(1)
                else:
                    logger.warning(f"Unknown diff type: {type(diff)}")
                    break
                
                # Break if we have consecutive empty batches
                if consecutive_empty_batches >= max_empty_batches:
                    logger.info(f"Stopping after {consecutive_empty_batches} consecutive empty batches")
                    break
                
                # Adaptive sleep based on batch size
                if batch_users_found + batch_chats_found > 100:
                    await asyncio.sleep(0.1)
                elif batch_users_found + batch_chats_found > 10:
                    await asyncio.sleep(0.05)
                else:
                    await asyncio.sleep(0.02)
                    
            except asyncio.TimeoutError:
                logger.warning(f"Batch {batch_count} timeout, continuing with collected data")
                consecutive_empty_batches += 1
                if consecutive_empty_batches >= max_empty_batches:
                    break
                await asyncio.sleep(1)
                continue
                
            except FloodWaitError as fw:
                if fw.seconds > 60:
                    logger.warning(f"FloodWait too long: {fw.seconds}s, breaking")
                    break
                logger.info(f"FloodWait: {fw.seconds}s")
                await asyncio.sleep(fw.seconds + 1)
                continue
                
            except Exception as e:
                logger.error(f"Error in batch {batch_count}: {str(e)}")
                consecutive_empty_batches += 1
                if consecutive_empty_batches >= max_empty_batches:
                    break
                await asyncio.sleep(2)
                continue
    
    except Exception as e:
        logger.error(f"Major error in data collection: {str(e)}")
    
    total_time = time.time() - start_time
    logger.info(f"Data collection completed in {total_time:.2f}s - "
                f"Total: {len(all_users)} users, {len(all_chats)} chats from {batch_count} batches")
    
    return list(all_chats.values()), list(all_users.values())

@app.get("/")
async def get_api_info():
    return {
        "api_name": "Enhanced Telegram Users API",
        "version": "2.2.0",
        "status": "running",
        "description": "High-performance API for Telegram bot data with improved concurrency and large dataset support",
        "owners": [
            {"username": "@ISmartCoder"},
            {"username": "@theSmartDev"}
        ],
        "endpoints": [
            {
                "path": "/tgusers",
                "method": "GET", 
                "description": "Fetch bot data including bot info, chats, and users",
                "parameters": [
                    {
                        "name": "token",
                        "type": "string",
                        "required": True,
                        "description": "Telegram Bot Token"
                    }
                ]
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check endpoint"
            },
            {
                "path": "/docs", 
                "method": "GET",
                "description": "Get interactive API documentation"
            }
        ],
        "contact": "Contact @ISmartCoder or @theSmartDev for support"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "active_clients": len(client_manager.clients)
    }

@app.get("/tgusers", response_model=BotDataResponse)
async def get_bot_data_optimized(
    background_tasks: BackgroundTasks,
    token: str = Query(..., description="Telegram Bot Token", min_length=10)
):
    start_time = time.time()
    
    if not token or len(token.strip()) < 10:
        raise HTTPException(status_code=400, detail="Invalid bot token format")

    try:
        logger.info(f"Processing optimized request for token: {token[:10]}...")
        
        client = await client_manager.get_client(
            bot_token=token.strip(),
            api_id=26512884,
            api_hash="c3f491cd59af263cfc249d3f93342ef8"
        )

        try:
            # Get bot info with timeout
            bot_info_task = asyncio.create_task(
                asyncio.wait_for(client.get_me(), timeout=20.0)
            )
            
            # Get chats and users with optimized method
            chats_users_task = asyncio.create_task(
                asyncio.wait_for(get_chats_and_users_optimized(client), timeout=720.0)  # 12 minutes
            )
            
            # Wait for both tasks
            results = await asyncio.gather(
                bot_info_task,
                chats_users_task,
                return_exceptions=True
            )
            
            me = results[0]
            chats_users_result = results[1]
            
            if isinstance(me, Exception):
                logger.error(f"Error getting bot info: {me}")
                raise me
                
            if isinstance(chats_users_result, Exception):
                logger.error(f"Error getting chats/users: {chats_users_result}")
                chats, users = [], []
            else:
                chats, users = chats_users_result
                
            # Ensure we have valid lists
            if not isinstance(chats, list):
                chats = []
            if not isinstance(users, list):
                users = []
                
            bot_info = BotInfoModel(
                first_name=me.first_name or "Unknown",
                id=me.id,
                username=me.username
            )
            
            processing_time = time.time() - start_time
            
            response = BotDataResponse(
                bot_info=bot_info,
                chats=chats,
                users=users,
                total_chats=len(chats),
                total_users=len(users),
                processing_time=round(processing_time, 3)
            )
            
            logger.info(f"Optimized request completed in {processing_time:.3f}s - "
                       f"Chats: {len(chats)}, Users: {len(users)}")
            
            # Schedule cleanup
            background_tasks.add_task(client_manager.cleanup_client, token.strip())
            
            return response
            
        except asyncio.TimeoutError:
            logger.error("Request timeout")
            await client_manager.cleanup_client(token.strip())
            raise HTTPException(status_code=408, detail="Request timeout - dataset too large, try again")
            
        except (AuthKeyPermEmptyError, SessionPasswordNeededError) as e:
            logger.error(f"Authentication error: {e}")
            await client_manager.cleanup_client(token.strip())
            raise HTTPException(status_code=401, detail="Invalid bot token or bot deactivated")
            
        except HTTPException:
            raise
            
        except RPCError as e:
            logger.error(f"Telegram API error: {e}")
            await client_manager.cleanup_client(token.strip())
            raise HTTPException(status_code=400, detail=f"Telegram API error: {str(e)}")
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            await client_manager.cleanup_client(token.strip())
            raise HTTPException(status_code=500, detail="Internal server error")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Critical server error")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting optimized server on {host}:{port}")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        workers=1,
        loop="uvloop" if 'uvloop' in globals() else "asyncio",
        access_log=True,
        reload=False,
        timeout_keep_alive=60,
        timeout_graceful_shutdown=60
    )
