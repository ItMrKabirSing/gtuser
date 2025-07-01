import asyncio
import logging
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
        self.executor = ThreadPoolExecutor(max_workers=4)  # For CPU-bound tasks
        self._session: Optional[aiohttp.ClientSession] = None  # Deferred initialization

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
                except:
                    if bot_token in self.clients:
                        del self.clients[bot_token]

            # Create new client
            session_name = f"bot_{uuid4().hex[:8]}"
            client = Client(
                name=session_name,
                bot_token=bot_token,
                api_id=api_id,
                api_hash=api_hash,
                in_memory=True,
                max_concurrent_transmissions=16,
                sleep_threshold=180,
                workers=8,
            )

            try:
                await asyncio.wait_for(client.start(), timeout=20.0)
                async with self.global_lock:
                    self.clients[bot_token] = client
                return client
            except Exception as e:
                logger.error(f"Failed to create client: {e}")
                raise

    async def cleanup_client(self, bot_token: str):
        async with self.global_lock:
            if bot_token in self.clients:
                try:
                    client = self.clients[bot_token]
                    if client.is_connected:
                        await client.stop()
                    del self.clients[bot_token]
                    del self.locks[bot_token]
                except Exception as e:
                    logger.error(f"Error cleaning up client {bot_token}: {e}")

    async def shutdown(self):
        async with self.global_lock:
            for bot_token in list(self.clients.keys()):
                await self.cleanup_client(bot_token)
            if self._session and not self._session.closed:
                await self._session.close()
        self.executor.shutdown()

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
    type_map = {
        "chat": "chat",
        "channel": "channel",
        "chatforbidden": "chat",
        "channelforbidden": "channel"
    }
    return type_map.get(raw_type.lower(), raw_type.lower())

# Chat merge logic
def merge_chat_data(existing: Optional[ChatModel], new: ChatModel) -> ChatModel:
    if not existing:
        return new
    return ChatModel(
        id=existing.id,
        members_count=new.members_count if new.members_count is not None else existing.members_count,
        title=new.title if new.title != "Unknown" else existing.title,
        type=new.type,
        username=new.username if new.username else existing.username
    )

# Enhanced chat and user fetching with parallel processing
async def get_chats_and_users_fast(client: Client) -> Tuple[List[ChatModel], List[UserModel]]:
    chats: Dict[int, ChatModel] = {}
    users: Dict[int, UserModel] = {}
    inaccessible_chats = set()
    custom_pts = 1
    custom_date = 1
    custom_qts = 1
    semaphore = asyncio.Semaphore(10)  # Control concurrent operations

    async def process_batch(diff):
        async with semaphore:
            batch_users = []
            batch_chats = []

            # Process users
            for user in getattr(diff, 'users', []):
                users[user.id] = UserModel(
                    id=user.id,
                    first_name=user.first_name,
                    last_name=user.last_name,
                    username=user.username,
                    is_premium=user.premium if hasattr(user, 'premium') else False
                )
                batch_users.append({
                    "id": user.id,
                    "first_name": user.first_name,
                    "username": user.username
                })

            # Process chats
            for chat in getattr(diff, 'chats', []):
                if chat.id not in chats and chat.id not in inaccessible_chats:
                    if chat.__class__.__name__.lower() in ["chatforbidden", "channelforbidden"]:
                        inaccessible_chats.add(chat.id)
                        continue
                    chat_type = normalize_chat_type(chat.__class__.__name__)
                    chat_data = ChatModel(
                        id=chat.id,
                        members_count=chat.members_count if hasattr(chat, "members_count") else None,
                        title=chat.title or getattr(chat, 'first_name', None) or "Unknown",
                        type=chat_type,
                        username=chat.username if hasattr(chat, "username") else None
                    )
                    chats[chat.id] = merge_chat_data(chats.get(chat.id), chat_data)
                    batch_chats.append({
                        "id": chat_data.id,
                        "members_count": chat_data.members_count,
                        "title": chat_data.title,
                        "type": chat_data.type,
                        "username": chat_data.username
                    })

            # Process messages
            for update in getattr(diff, 'new_messages', []):
                if isinstance(update, (types.UpdateNewMessage, types.UpdateNewChannelMessage)):
                    if hasattr(update, 'message') and hasattr(update.message, 'chat'):
                        chat = update.message.chat
                        if chat and chat.id not in chats and chat.id not in inaccessible_chats:
                            if chat.__class__.__name__.lower() in ["chatforbidden", "channelforbidden"]:
                                inaccessible_chats.add(chat.id)
                                continue
                            chat_type = normalize_chat_type(chat.type.name if hasattr(chat, 'type') and chat.type else chat.__class__.__name__)
                            chat_data = ChatModel(
                                id=chat.id,
                                members_count=chat.members_count if hasattr(chat, "members_count") else None,
                                title=chat.title or getattr(chat, 'first_name', None) or "Unknown",
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

            return batch_users, batch_chats

    try:
        while True:
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
                    timeout=15.0
                )

                # Process batch in parallel
                batch_users, batch_chats = await process_batch(diff)

                if not batch_users and not batch_chats:
                    if isinstance(diff, types.updates.Difference):
                        break
                    if isinstance(diff, types.updates.DifferenceSlice):
                        custom_pts = diff.intermediate_state.pts
                        custom_date = diff.intermediate_state.date
                        custom_qts = diff.intermediate_state.qts
                    continue

                if isinstance(diff, types.updates.DifferenceSlice):
                    custom_pts = diff.intermediate_state.pts
                    custom_date = diff.intermediate_state.date
                    custom_qts = diff.intermediate_state.qts
                elif isinstance(diff, types.updates.Difference):
                    break
                else:
                    break

                await asyncio.sleep(0.03)

            except asyncio.TimeoutError:
                logger.warning("GetDifference timeout, continuing")
                break
            except FloodWait as fw:
                if fw.value > 60:
                    logger.warning(f"FloodWait too long: {fw.value}s, breaking")
                    break
                await asyncio.sleep(fw.value)

    except Exception as e:
        logger.error(f"Error fetching chats and users: {str(e)}")
        return list(chats.values()), list(users.values())

    return list(chats.values()), list(users.values())

# API Routes
@app.get("/")
async def get_api_info():
    return {
        "api_name": "Enhanced Telegram Users API",
        "version": "2.1.0",
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
                "path": "/docs",
                "method": "GET",
                "description": "Get interactive API documentation",
                "parameters": []
            }
        ],
        "contact": "Contact @ISmartCoder or @theSmartDev for support"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/tgusers", response_model=BotDataResponse)
async def get_bot_data_fast(
    background_tasks: BackgroundTasks,
    token: str = Query(..., description="Telegram Bot Token", min_length=10)
):
    start_time = time.time()

    try:
        # Create/get client with proper connection management
        client = await client_manager.get_client(
            bot_token=token,
            api_id=26512884,
            api_hash="c3f491cd59af263cfc249d3f93342ef8"
        )

        # Run tasks concurrently
        bot_info_task = asyncio.create_task(client.get_me())
        chats_users_task = asyncio.create_task(get_chats_and_users_fast(client))

        try:
            me, (chats, users) = await asyncio.gather(
                bot_info_task,
                chats_users_task,
                return_exceptions=True
            )

            # Handle exceptions
            if isinstance(me, Exception):
                raise me
            if isinstance(chats, Exception) or isinstance(users, Exception):
                chats, users = [], []

            # Create response
            bot_info = BotInfoModel(
                first_name=me.first_name,
                id=me.id,
                username=me.username
            )

            processing_time = time.time() - start_time

            # Schedule cleanup for this specific client
            background_tasks.add_task(client_manager.cleanup_client, token)

            return BotDataResponse(
                bot_info=bot_info,
                chats=chats,
                users=users,
                total_chats=len(chats),
                total_users=len(users),
                processing_time=round(processing_time, 3)
            )

        finally:
            # Ensure client is properly handled for next request
            if client.is_connected:
                await client_manager.cleanup_client(token)

    except (AuthKeyUnregistered, UserDeactivated):
        raise HTTPException(status_code=401, detail="Invalid bot token")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timeout")
    except RPCError as e:
        raise HTTPException(status_code=400, detail=f"Telegram API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Run with maximum performance
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        loop="uvloop" if 'uvloop' in globals() else "asyncio",
        workers=2,
        access_log=False,
        reload=False,
        timeout_keep_alive=30
    )