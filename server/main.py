import logging
import os
import secrets
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from mem0 import Memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")

MIN_KEY_LENGTH = 16

if not ADMIN_API_KEY:
    logging.warning(
        "ADMIN_API_KEY not set - API endpoints are UNSECURED! "
        "Set ADMIN_API_KEY environment variable for production use."
    )
else:
    if len(ADMIN_API_KEY) < MIN_KEY_LENGTH:
        logging.warning(
            "ADMIN_API_KEY is shorter than %d characters - consider using a longer key for production.",
            MIN_KEY_LENGTH,
        )
    logging.info("API key authentication enabled")

QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mem0graph")

MEMGRAPH_URI = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USERNAME = os.environ.get("MEMGRAPH_USERNAME", "memgraph")
MEMGRAPH_PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "mem0graph")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/app/history/history.db")


CUSTOM_FACT_EXTRACTION_PROMPT = """
请从对话中提取关键事实和信息，使用中文输出。以下是一些示例：

输入: 你好
输出: {"facts" : []}

输入: 今天天气不错
输出: {"facts" : []}

输入: 我叫张三，我是一名软件工程师
输出: {"facts" : ["用户名字是张三", "职业是软件工程师"]}

输入: 我喜欢用 Python 写代码，最近在学 Rust
输出: {"facts" : ["喜欢使用 Python 编程", "正在学习 Rust"]}

输入: 我住在东京，之前在上海工作了五年
输出: {"facts" : ["目前居住在东京", "曾在上海工作五年"]}

输入: 我对机器学习和自然语言处理很感兴趣，特别是大语言模型
输出: {"facts" : ["对机器学习感兴趣", "对自然语言处理感兴趣", "特别关注大语言模型"]}

请以上述 JSON 格式返回提取的事实信息。只提取有意义的个人信息、偏好、经历和观点，忽略寒暄和无关内容。
"""


CUSTOM_UPDATE_MEMORY_PROMPT = """
你是一个智能记忆管理器，负责管理系统的记忆存储。你可以执行以下四种操作：
1. ADD（添加）：将新信息加入记忆
2. UPDATE（更新）：更新已有记忆中的信息
3. DELETE（删除）：删除与新信息矛盾的记忆
4. NONE（无操作）：信息已存在且无需变更

对比"已检索到的事实"和"已有记忆"，按以下规则判断：

- 如果检索到的事实包含记忆中没有的新信息，使用 ADD，并生成新的 ID。
- 如果检索到的事实与已有记忆内容相关但信息有更新或变化，使用 UPDATE，保留原有 ID，并在 old_memory 字段记录旧内容。
- 如果检索到的事实与已有记忆明确矛盾（如偏好改变、事实纠正），使用 DELETE 删除旧记忆，然后 ADD 新记忆。
- 如果检索到的事实与已有记忆语义相同，使用 NONE。

以下是一些示例：

已有记忆：
- ID: 1, 内容: "喜欢吃披萨"
- ID: 2, 内容: "住在东京"

检索到的事实: ["不喜欢吃披萨了，现在喜欢吃拉面", "在谷歌工作"]

输出:
{
  "memory": [
    {"id": "1", "text": "喜欢吃拉面", "event": "UPDATE", "old_memory": "喜欢吃披萨"},
    {"id": "3", "text": "在谷歌工作", "event": "ADD"}
  ]
}

请注意：
- 更详细的信息应覆盖简略信息
- 语义相同但表述不同的记忆不需要更新
- 始终使用中文输出
- 严格按照上述 JSON 格式返回结果
"""

DEFAULT_CONFIG = {
    "version": "v1.1",
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
            "collection_name": "memories",
            "embedding_model_dims": 3072,
            "api_key": QDRANT_API_KEY,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {"url": NEO4J_URI, "username": NEO4J_USERNAME, "password": NEO4J_PASSWORD},
    },
    "llm": {"provider": "gemini", "config": {"api_key": GOOGLE_API_KEY, "temperature": 0.2, "model": "gemini-3.1-flash-lite-preview"}},
    "embedder": {"provider": "gemini", "config": {"api_key": GOOGLE_API_KEY, "model": "gemini-embedding-2-preview", "embedding_dims": 3072}},
    "history_db_path": HISTORY_DB_PATH,
    "custom_fact_extraction_prompt": CUSTOM_FACT_EXTRACTION_PROMPT,
    "custom_update_memory_prompt": CUSTOM_UPDATE_MEMORY_PROMPT,
    "reranker": {
        "provider": "llm_reranker",
        "config": {
            "llm": {
                "provider": "gemini",
                "config": {
                    "model": "gemini-3.1-flash-lite-preview",
                    "api_key": GOOGLE_API_KEY,
                    "temperature": 0.0,
                },
            },
            "top_k": 5,
        },
    },
}


MEMORY_INSTANCE = Memory.from_config(DEFAULT_CONFIG)


from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import json

class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT") and request.url.path in ("/memories", "/search"):
            body = await request.body()
            try:
                data = json.loads(body)
                logging.info(f"[REQUEST] {request.method} {request.url.path} body={json.dumps(data, ensure_ascii=False)}")
            except:
                logging.info(f"[REQUEST] {request.method} {request.url.path} body={body[:500]}")
        elif request.method == "GET":
            logging.info(f"[REQUEST] {request.method} {request.url.path}?{request.query_params}")
        response = await call_next(request)
        return response

app = FastAPI(
    title="Mem0 REST APIs",
    description=(
        "A REST API for managing and searching memories for your AI Agents and Apps.\n\n"
        "## Authentication\n"
        "When the ADMIN_API_KEY environment variable is set, all endpoints require "
        "the `X-API-Key` header for authentication."
    ),
    version="1.0.0",
)

app.add_middleware(RequestLogMiddleware)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """Validate the API key when ADMIN_API_KEY is configured. No-op otherwise."""
    if ADMIN_API_KEY:
        if api_key is None:
            raise HTTPException(
                status_code=401,
                detail="X-API-Key header is required.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        if not secrets.compare_digest(api_key, ADMIN_API_KEY):
            raise HTTPException(
                status_code=401,
                detail="Invalid API key.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
    return api_key


class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")


class MemoryCreate(BaseModel):
    messages: List[Message] = Field(..., description="List of messages to store.")
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    infer: Optional[bool] = Field(None, description="Whether to extract facts from messages. Defaults to True.")
    memory_type: Optional[str] = Field(None, description="Type of memory to store (e.g. 'core').")
    prompt: Optional[str] = Field(None, description="Custom prompt to use for fact extraction.")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query.")
    user_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = Field(None, description="Maximum number of results to return.")
    threshold: Optional[float] = Field(None, description="Minimum similarity score for results.")


@app.post("/configure", summary="Configure Mem0")
def set_config(config: Dict[str, Any], _api_key: Optional[str] = Depends(verify_api_key)):
    """Set memory configuration."""
    global MEMORY_INSTANCE
    MEMORY_INSTANCE = Memory.from_config(config)
    return {"message": "Configuration set successfully"}


@app.post("/memories", summary="Create memories")
def add_memory(memory_create: MemoryCreate, _api_key: Optional[str] = Depends(verify_api_key)):
    """Store new memories."""
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    params = {k: v for k, v in memory_create.model_dump().items() if v is not None and k != "messages"}
    try:
        response = MEMORY_INSTANCE.add(messages=[m.model_dump() for m in memory_create.messages], **params)
        return JSONResponse(content=response)
    except Exception as e:
        logging.exception("Error in add_memory:")  # This will log the full traceback
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories", summary="Get memories")
def get_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Retrieve stored memories."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        return MEMORY_INSTANCE.get_all(**params)
    except Exception as e:
        logging.exception("Error in get_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}", summary="Get a memory")
def get_memory(memory_id: str, _api_key: Optional[str] = Depends(verify_api_key)):
    """Retrieve a specific memory by ID."""
    try:
        return MEMORY_INSTANCE.get(memory_id)
    except Exception as e:
        logging.exception("Error in get_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", summary="Search memories")
def search_memories(search_req: SearchRequest, _api_key: Optional[str] = Depends(verify_api_key)):
    """Search for memories based on a query."""
    try:
        params = {k: v for k, v in search_req.model_dump().items() if v is not None and k != "query"}
        return MEMORY_INSTANCE.search(query=search_req.query, **params)
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/memories/{memory_id}", summary="Update a memory")
def update_memory(memory_id: str, updated_memory: Dict[str, Any], _api_key: Optional[str] = Depends(verify_api_key)):
    """Update an existing memory with new content.
    
    Args:
        memory_id (str): ID of the memory to update
        updated_memory (str): New content to update the memory with
        
    Returns:
        dict: Success message indicating the memory was updated
    """
    try:
        return MEMORY_INSTANCE.update(memory_id=memory_id, data=updated_memory)
    except Exception as e:
        logging.exception("Error in update_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}/history", summary="Get memory history")
def memory_history(memory_id: str, _api_key: Optional[str] = Depends(verify_api_key)):
    """Retrieve memory history."""
    try:
        return MEMORY_INSTANCE.history(memory_id=memory_id)
    except Exception as e:
        logging.exception("Error in memory_history:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories/{memory_id}", summary="Delete a memory")
def delete_memory(memory_id: str, _api_key: Optional[str] = Depends(verify_api_key)):
    """Delete a specific memory by ID."""
    try:
        MEMORY_INSTANCE.delete(memory_id=memory_id)
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        logging.exception("Error in delete_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories", summary="Delete all memories")
def delete_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Delete all memories for a given identifier."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        MEMORY_INSTANCE.delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Error in delete_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", summary="Reset all memories")
def reset_memory(_api_key: Optional[str] = Depends(verify_api_key)):
    """Completely reset stored memories."""
    try:
        MEMORY_INSTANCE.reset()
        return {"message": "All memories reset"}
    except Exception as e:
        logging.exception("Error in reset_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", summary="Redirect to the OpenAPI documentation", include_in_schema=False)
def home():
    """Redirect to the OpenAPI documentation."""
    return RedirectResponse(url="/docs")
