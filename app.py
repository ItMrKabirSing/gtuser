import asyncio
import logging
import os
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
import time
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pyrogram import Client
from pyrogram.raw import functions, types
from pyrogram.errors import FloodWait, RPCError, AuthKeyUnregistered, UserDeactivated
from uuid import uuid4
import aiohttp

# Enhanced logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to use uvloop for better performance
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logger.info("Using uvloop for better performance")
except ImportError:
    logger.warning("uvloop not available, using default event loop")

# Pydantic models
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

# Connection management
class ClientManager:
    def __init__(self):
        self.clients: Dict[str, Client] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        self.global_lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)  # Reduced for stability
        self._session: Optional[aiohttp.ClientSession] = None

    async def get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def get_client(self, bot_token: str, api_id: int, api_hash: str) -> Client:
        async with self.global_lock:
            if bot_token not in self.locks:
                self.locks[bot_token] = asyncio.Lock()

        async with self.locks[bot_token]:
            if bot_token in self.clients:
                client = self.clients[bot_token]
                try:
                    if client.is_connected:
                        return client
                    else:
                        await client.stop()
                        del self.clients[bot_token]
                except Exception as e:
                    logger.warning(f"Client connection issue: {e}")
                    if bot_token in self.clients:
                        del self.clients[bot_token]

            # Create new client with valid parameters only
            session_name = f"bot_{uuid4().hex[:8]}"
            client = Client(
                name=session_name,
                bot_token=bot_token,
                api_id=api_id,
                api_hash=api_hash,
                in_memory=True,
                max_concurrent_transmissions=8,
                sleep_threshold=120,
                workers=4,
            )

            try:
                await asyncio.wait_for(client.start(), timeout=30.0)
                async with self.global_lock:
                    self.clients[bot_token] = client
                logger.info(f"Client created successfully for token: {bot_token[:10]}...")
                return client
            except Exception as e:
                logger.error(f"Failed to create client: {e}")
                try:
                    await client.stop()
                except:
                    pass
                raise HTTPException(status_code=400, detail=f"Failed to connect to Telegram: {str(e)}")

    async def cleanup_client(self, bot_token: str):
        try:
            async with self.global_lock:
                if bot_token in self.clients:
                    client = self.clients[bot_token]
                    try:
                        if client.is_connected:
                            await asyncio.wait_for(client.stop(), timeout=10.0)
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

# Application lifespan management
client_manager = ClientManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Enhanced Telegram Users API...")
    try:
        yield
    finally:
        await client_manager.shutdown()
        logger.info("API shutdown complete")

# FastAPI app
app = FastAPI(
    title="Enhanced Telegram Users API",
    description="High-performance async API for Telegram bot data with improved concurrency",
    version="2.1.0",
    lifespan=lifespan
)

# Chat type normalization
def normalize_chat_type(raw_type: str) -> str:
    """Normalize chat type strings"""
    type_map = {
        "chat": "group",
        "channel": "channel",
        "chatforbidden": "group",
        "channelforbidden": "channel",
        "user": "private"
    }
    return type_map.get(raw_type.lower(), raw_type.lower())

# Chat merge logic
def merge_chat_data(existing: Optional[ChatModel], new: ChatModel) -> ChatModel:
    """Merge chat data with existing data taking precedence for non-null values"""
    if not existing:
        return new
    return ChatModel(
        id=existing.id,
        members_count=new.members_count if new.members_count is not None else existing.members_count,
        title=new.title if new.title and new.title != "Unknown" else existing.title,
        type=new.type,
        username=new.username if new.username else existing.username
    )

# Enhanced chat and user fetching with better error handling
async def get_chats_and_users_fast(client: Client) -> Tuple[List[ChatModel], List[UserModel]]:
    """Fetch chats and users with improved error handling and performance"""
    chats: Dict[int, ChatModel] = {}
    users: Dict[int, UserModel] = {}
    inaccessible_chats = set()
    batch_count = 0
    custom_pts = 1
    custom_date = 1
    custom_qts = 1
    start_time = time.time()
    max_duration = 300  # 5 minutes timeout

    try:
        while time.time() - start_time < max_duration:
            try:
                diff = await asyncio.wait_for(
                    client.invoke(
                        functions.updates.GetDifference(
                            pts=custom_pts,
                            date=custom_date,
                            qts=custom_qts,
                            pts_limit=5000,
                            pts_total_limit=1000000,
                            qts_limit=5000
                        )
                    ),
                    timeout=60.0
                )

                # Process users
                batch_users = []
                for user in getattr(diff, 'users', []):
                    if user.id not in users:
                        users[user.id] = UserModel(
                            id=user.id,
                            first_name=getattr(user, 'first_name', None),
                            last_name=getattr(user, 'last_name', None),
                            username=getattr(user, 'username', None),
                            is_premium=getattr(user, 'premium', False)
                        )
                        batch_users.append({
                            "id": user.id,
                            "first_name": getattr(user, 'first_name', None),
                            "username": getattr(user, 'username', None)
                        })

                # Process chats
                batch_chats = []
                for chat in getattr(diff, 'chats', []):
                    if chat.id not in chats and chat.id not in inaccessible_chats:
                        chat_class_name = chat.__class__.__name__.lower()
                        if chat_class_name in ["chatforbidden", "channelforbidden"]:
                            inaccessible_chats.add(chat.id)
                            continue
                        chat_type = normalize_chat_type(chat_class_name)
                        chat_data = ChatModel(
                            id=chat.id,
                            members_count=getattr(chat, 'members_count', None),
                            title=getattr(chat, 'title', None) or getattr(chat, 'first_name', None) or "Unknown",
                            type=chat_type,
                            username=getattr(chat, 'username', None)
                        )
                        chats[chat.id] = merge_chat_data(chats.get(chat.id), chat_data)
                        batch_chats.append({
                            "id": chat_data.id,
                            "members_count": chat_data.members_count,
                            "title": chat_data.title,
                            "type": chat_data.type,
                            "username": chat_data.username
                        })

                # Process messages to extract additional chats
                for update in getattr(diff, 'new_messages', []):
                    if isinstance(update, (types.UpdateNewMessage, types.UpdateNewChannelMessage)):
                        if hasattr(update.message, 'chat') and update.message.chat:
                            chat = update.message.chat
                            if chat.id not in chats and chat.id not in inaccessible_chats:
                                chat_class_name = chat.__class__.__name__.lower()
                                if chat_class_name in ["chatforbidden", "channelforbidden"]:
                                    inaccessible_chats.add(chat.id)
                                    continue
                                chat_type = normalize_chat_type(chat_class_name)
                                chat_data = ChatModel(
                                    id=chat.id,
                                    members_count=getattr(chat, 'members_count', None),
                                    title=getattr(chat, 'title', None) or getattr(chat, 'first_name', None) or "Unknown",
                                    type=chat_type,
                                    username=getattr(chat, 'username', None)
                                )
                                chats[chat.id] = merge_chat_data(chats.get(chat.id), chat_data)
                                batch_chats.append({
                                    "id": chat_data.id,
                                    "members_count": chat_data.members_count,
                                    "title": chat_data.title,
                                    "type": chat_data.type,
                                    "username": chat_data.username
                                })

                # Log batch details
                if batch_users or batch_chats:
                    logger.info(f"Batch {batch_count}:")
                    logger.info(f"  Users fetched: {len(batch_users)}")
                    if batch_users:
                        logger.debug(f"  Users: {[user['id'] for user in batch_users]}")
                    logger.info(f"  Chats fetched: {len(batch_chats)}")
                    if batch_chats:
                        logger.debug(f"  Chats: {[chat['id'] for chat in batch_chats]}")
                    if inaccessible_chats:
                        logger.info(f"  Inaccessible chats: {len(inaccessible_chats)} IDs skipped")
                else:
                    logger.info(f"Batch {batch_count}: Skipped (no new users or chats)")

                batch_count += 1

                # Update state for next iteration
                if isinstance(diff, types.updates.DifferenceSlice):
                    custom_pts = diff.intermediate_state.pts
                    custom_date = diff.intermediate_state.date
                    custom_qts = diff.intermediate_state.qts
                elif isinstance(diff, types.updates.Difference):
                    break
                else:
                    logger.warning(f"Unknown diff type: {type(diff)}")
                    break

                # Small delay to prevent overwhelming the API
                await asyncio.sleep(0.05)

            except asyncio.TimeoutError:
                logger.warning("GetDifference timeout, continuing with collected data")
                break
            except FloodWait as fw:
                if fw.value > 30:
                    logger.warning(f"FloodWait too long: {fw.value}s, breaking")
                    break
                logger.info(f"FloodWait: {fw.value}s")
                await asyncio.sleep(fw.value)
            except Exception as e:
                logger.error(f"Error in iteration {batch_count}: {str(e)}")
                if batch_count > 50:
                    break
                await asyncio.sleep(1)

        # Check if we hit the timeout
        if time.time() - start_time >= max_duration:
            logger.warning("Reached 5-minute timeout, returning collected data")

    except Exception as e:
        logger.error(f"Major error fetching chats and users: {str(e)}")

    logger.info(f"Collected {len(chats)} chats and {len(users)} users in {batch_count} batches")
    return list(chats.values()), list(users.values())

# API Routes
@app.get("/")
async def get_api_info():
    """Get API information and available endpoints"""
    return {
        "api_name": "Enhanced Telegram Users API",
        "version": "2.1.0",
        "status": "running",
        "description": "High-performance API for Telegram bot data with improved concurrency",
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
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "active_clients": len(client_manager.clients)
    }

@app.get("/tgusers", response_model=BotDataResponse)
async def get_bot_data_fast(
    background_tasks: BackgroundTasks,
    token: str = Query(..., description="Telegram Bot Token", min_length=10)
):
    """
    Fetch comprehensive bot data including bot info, chats, and users
    """
    start_time = time.time()
    
    if not token or len(token.strip()) < 10:
        raise HTTPException(status_code=400, detail="Invalid bot token format")

    try:
        logger.info(f"Processing request for token: {token[:10]}...")
        
        # Create/get client with proper connection management
        client = await client_manager.get_client(
            bot_token=token.strip(),
            api_id=26512884,
            api_hash="c3f491cd59af263cfc249d3f93342ef8"
        )

        # Run tasks concurrently with timeouts
        try:
            bot_info_task = asyncio.create_task(
                asyncio.wait_for(client.get_me(), timeout=15.0)
            )
            chats_users_task = asyncio.create_task(
                asyncio.wait_for(get_chats_and_users_fast(client), timeout=300.0)  # 5-minute timeout
            )

            # Wait for both tasks
            results = await asyncio.gather(
                bot_info_task,
                chats_users_task,
                return_exceptions=True
            )
            
            me = results[0]
            chats_users = results[1]

            # Handle exceptions
            if isinstance(me, Exception):
                logger.error(f"Error getting bot info: {me}")
                raise me
                
            if isinstance(chats_users, Exception):
                logger.warning(f"Error getting chats/users: {chats_users}")
                chats, users = [], []
            else:
                chats, users = chats_users

            # Ensure we have valid data
            if not isinstance(chats, list):
                chats = []
            if not isinstance(users, list):
                users = []

            # Create response
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

            logger.info(f"Request completed in {processing_time:.3f}s - Chats: {len(chats)}, Users: {len(users)}")
            
            # Schedule cleanup
            background_tasks.add_task(client_manager.cleanup_client, token.strip())
            
            return response

        except asyncio.TimeoutError:
            logger.error("Request timeout")
            await client_manager.cleanup_client(token.strip())
            raise HTTPException(status_code=408, detail="Request timeout - try again")

    except (AuthKeyUnregistered, UserDeactivated) as e:
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

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    # Run with production settings
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        workers=1,
        loop="uvloop" if 'uvloop' in globals() else "asyncio",
        access_log=True,
        reload=False,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=30
    )
