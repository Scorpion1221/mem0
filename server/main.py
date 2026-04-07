import logging
import os
import re as _re
import secrets
import threading
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
_prompt_lock = threading.Lock()


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
    custom_instructions: Optional[str] = Field(None, description="Alias for prompt (openclaw compatibility).")
    custom_categories: Optional[List[Dict[str, str]]] = Field(
        None,
        description='Category definitions, e.g. [{"identity": "Name, location..."}, {"projects": "Active projects..."}]',
    )


class MemoryUpdate(BaseModel):
    text: str = Field(..., description="New content to update the memory with.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata to update.")


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


import json as _json


def _backfill_categories(results: List[Dict[str, Any]], categories: List[Dict[str, str]]):
    """Classify memories and store category in metadata via a single LLM call.

    After mem0 add() returns, the memory text is clean (no [tag] prefix) because
    mem0's internal update-memory LLM strips any prefix during deduplication.
    So we do a separate batch classification: send all new memories + category
    definitions to the LLM, get back a mapping, then call update() per memory.
    """
    actionable = [
        item for item in results
        if item.get("event") in ("ADD", "UPDATE") and item.get("id") and item.get("memory")
    ]
    if not actionable:
        return

    # Also try [tag] prefix match first (works when update-memory LLM preserves it)
    tag_pattern = _re.compile(r"^\[([a-z_]+)\]\s*")
    remaining = []
    for item in actionable:
        match = tag_pattern.match(item["memory"])
        if match:
            category = match.group(1)
            clean_text = item["memory"][match.end():]
            try:
                MEMORY_INSTANCE.update(item["id"], data=clean_text, metadata={"category": category})
                item["memory"] = clean_text
                item["category"] = category
            except Exception as e:
                logging.warning(f"Failed to backfill category for {item['id']}: {e}")
        else:
            remaining.append(item)

    if not remaining:
        return

    # Batch LLM classification for memories without [tag] prefix
    cat_names = [k for cat in categories for k in cat.keys()]
    cat_desc = ", ".join(f"{k}({v})" for cat in categories for k, v in cat.items())
    memories_list = "\n".join(f"[{i}] {item['memory']}" for i, item in enumerate(remaining))

    prompt = (
        f"将以下记忆分类到最匹配的类别中。可用类别: {cat_desc}\n\n"
        f"{memories_list}\n\n"
        f'仅输出 JSON 数组，元素为类别名，顺序对应上方编号。如 ["{cat_names[0]}", "{cat_names[-1]}"]'
    )

    try:
        response = MEMORY_INSTANCE.llm.generate_response(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        # Parse: might be a raw array or {"categories": [...]} or {"result": [...]}
        parsed = _json.loads(response)
        if isinstance(parsed, list):
            labels = parsed
        elif isinstance(parsed, dict):
            labels = parsed.get("categories") or parsed.get("result") or list(parsed.values())[0]
        else:
            labels = []

        for i, item in enumerate(remaining):
            if i < len(labels) and labels[i] in cat_names:
                category = labels[i]
                try:
                    MEMORY_INSTANCE.update(item["id"], data=item["memory"], metadata={"category": category})
                    item["category"] = category
                except Exception as e:
                    logging.warning(f"Failed to update category for {item['id']}: {e}")
    except Exception as e:
        logging.warning(f"Batch category classification failed: {e}")


@app.post("/memories", summary="Create memories")
def add_memory(memory_create: MemoryCreate, _api_key: Optional[str] = Depends(verify_api_key)):
    """Store new memories."""
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    params = {k: v for k, v in memory_create.model_dump().items() if v is not None and k not in ("messages", "prompt", "custom_instructions", "custom_categories")}

    # Build effective prompt: per-request prompt > server default.
    # If custom_categories provided, append category tagging instructions.
    effective_prompt = memory_create.prompt or memory_create.custom_instructions
    categories = memory_create.custom_categories

    if categories:
        cat_lines = "\n".join(f"- {k}: {v}" for cat in categories for k, v in cat.items())
        cat_suffix = (
            "\n\n## 分类标注要求（覆盖上方输出格式）\n"
            "请为每条提取的事实标注最匹配的分类。可用分类:\n"
            f"{cat_lines}\n\n"
            '【输出格式】每条 fact 必须是包含 "text" 和 "category" 的对象:\n'
            '{"facts": [{"text": "事实内容", "category": "分类名"}, ...]}\n\n'
            "以下是几个完整示例:\n\n"
            "输入: 我叫张三，是一名软件工程师\n"
            '输出: {"facts": [{"text": "用户名叫张三", "category": "identity"}, '
            '{"text": "职业是软件工程师", "category": "identity"}]}\n\n'
            "输入: 我最近在用 Rust 重写项目，比较喜欢函数式编程\n"
            '输出: {"facts": [{"text": "正在用 Rust 重写项目", "category": "projects"}, '
            '{"text": "喜欢函数式编程", "category": "preferences"}]}\n\n'
            "输入: 今天天气不错\n"
            '输出: {"facts": []}\n\n'
            "输入: 我住在东京，周末喜欢跑步，正在准备明年的马拉松\n"
            '输出: {"facts": [{"text": "居住在东京", "category": "identity"}, '
            '{"text": "周末喜欢跑步", "category": "preferences"}, '
            '{"text": "正在准备明年的马拉松", "category": "projects"}]}\n\n'
            "如果没有匹配的分类，category 设为 \"misc\"。"
        )
        base_prompt = effective_prompt or CUSTOM_FACT_EXTRACTION_PROMPT
        effective_prompt = base_prompt + cat_suffix

    if effective_prompt:
        with _prompt_lock:
            original_prompt = MEMORY_INSTANCE.custom_fact_extraction_prompt
            MEMORY_INSTANCE.custom_fact_extraction_prompt = effective_prompt
            MEMORY_INSTANCE.config.custom_fact_extraction_prompt = effective_prompt
            try:
                response = MEMORY_INSTANCE.add(messages=[m.model_dump() for m in memory_create.messages], **params)
            except Exception as e:
                logging.exception("Error in add_memory:")
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                MEMORY_INSTANCE.custom_fact_extraction_prompt = original_prompt
                MEMORY_INSTANCE.config.custom_fact_extraction_prompt = original_prompt
    else:
        try:
            response = MEMORY_INSTANCE.add(messages=[m.model_dump() for m in memory_create.messages], **params)
        except Exception as e:
            logging.exception("Error in add_memory:")
            raise HTTPException(status_code=500, detail=str(e))

    # Post-process: classify memories and store category in metadata.
    if categories and response.get("results"):
        _backfill_categories(response["results"], categories)

    return JSONResponse(content=response)


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
def update_memory(memory_id: str, updated_memory: MemoryUpdate, _api_key: Optional[str] = Depends(verify_api_key)):
    """Update an existing memory with new content.

    Args:
        memory_id (str): ID of the memory to update
        updated_memory (MemoryUpdate): New content and optional metadata to update the memory with

    Returns:
        dict: Success message indicating the memory was updated
    """
    try:
        return MEMORY_INSTANCE.update(memory_id=memory_id, data=updated_memory.text, metadata=updated_memory.metadata)
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
