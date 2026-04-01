"""Tests for custom_instructions, custom prompt override, and custom_categories.

Verifies that:
- The `prompt` and `custom_instructions` fields temporarily override the
  extraction prompt on MEMORY_INSTANCE (with lock safety).
- The `custom_categories` field appends category tagging instructions to the
  prompt and triggers _backfill_categories post-processing.
- _backfill_categories correctly parses [tag] prefixes and calls update().
- Fields that should not leak into Memory.add() kwargs are excluded.
"""

import importlib
import os
from unittest.mock import MagicMock, patch, call

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def _mock_memory():
    """Patch Memory.from_config so the server imports without a real backend."""
    mock_instance = MagicMock()
    mock_instance.custom_fact_extraction_prompt = "default server prompt"
    mock_instance.config = MagicMock()
    mock_instance.config.custom_fact_extraction_prompt = "default server prompt"
    mock_instance.add.return_value = {
        "results": [{"id": "mem-1", "event": "ADD", "memory": "test fact"}]
    }
    mock_instance.update.return_value = {"message": "Memory updated"}
    mock_instance.search.return_value = [{"id": "mem-1", "memory": "test", "score": 0.9}]

    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key", "ADMIN_API_KEY": ""}):
        with patch("mem0.Memory.from_config", return_value=mock_instance):
            yield mock_instance


@pytest.fixture
def client(_mock_memory):
    """Return a TestClient wired to the server app with mocked Memory."""
    try:
        import server.main as server_main
    except ModuleNotFoundError:
        import main as server_main  # When running from /app (docker)
    with patch.dict(os.environ, {"ADMIN_API_KEY": ""}):
        importlib.reload(server_main)
    return TestClient(server_main.app)


@pytest.fixture
def mock_memory(_mock_memory):
    return _mock_memory


# ===========================================================================
# custom_instructions as alias for prompt
# ===========================================================================

class TestCustomInstructions:
    """Verify that custom_instructions is accepted as an alias for prompt."""

    def test_custom_instructions_accepted(self, client, mock_memory):
        """custom_instructions should not cause a 422 validation error."""
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "I like pizza"}],
            "user_id": "u1",
            "custom_instructions": "Extract food preferences only",
        })
        assert resp.status_code == 200

    def test_custom_instructions_not_in_add_kwargs(self, client, mock_memory):
        """custom_instructions should NOT be forwarded as a kwarg to Memory.add()."""
        client.post("/memories", json={
            "messages": [{"role": "user", "content": "I like pizza"}],
            "user_id": "u1",
            "custom_instructions": "Extract food preferences only",
        })
        _, kwargs = mock_memory.add.call_args
        assert "custom_instructions" not in kwargs
        assert "prompt" not in kwargs

    def test_custom_instructions_overrides_extraction_prompt(self, client, mock_memory):
        """When custom_instructions is provided, it should temporarily set
        the extraction prompt on MEMORY_INSTANCE during the add() call."""
        custom_prompt = "Extract food preferences only"
        prompts_seen = []

        original_add = mock_memory.add

        def capture_prompt(*args, **kwargs):
            prompts_seen.append(mock_memory.custom_fact_extraction_prompt)
            return original_add.return_value

        mock_memory.add.side_effect = capture_prompt

        client.post("/memories", json={
            "messages": [{"role": "user", "content": "I like pizza"}],
            "user_id": "u1",
            "custom_instructions": custom_prompt,
        })

        assert len(prompts_seen) == 1
        assert prompts_seen[0] == custom_prompt

    def test_extraction_prompt_restored_after_request(self, client, mock_memory):
        """After the request completes, the extraction prompt should be
        restored to its original value."""
        original = mock_memory.custom_fact_extraction_prompt

        client.post("/memories", json={
            "messages": [{"role": "user", "content": "I like pizza"}],
            "user_id": "u1",
            "custom_instructions": "Temporary prompt",
        })

        assert mock_memory.custom_fact_extraction_prompt == original

    def test_extraction_prompt_restored_on_error(self, client, mock_memory):
        """Even if Memory.add() raises, the prompt should be restored."""
        original = mock_memory.custom_fact_extraction_prompt
        mock_memory.add.side_effect = RuntimeError("boom")

        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "I like pizza"}],
            "user_id": "u1",
            "custom_instructions": "Temporary prompt",
        })

        assert resp.status_code == 500
        assert mock_memory.custom_fact_extraction_prompt == original


# ===========================================================================
# prompt field (existing, should still work)
# ===========================================================================

class TestPromptField:
    """Verify that the prompt field overrides the extraction prompt."""

    def test_prompt_overrides_extraction_prompt(self, client, mock_memory):
        custom_prompt = "My custom extraction prompt"
        prompts_seen = []

        def capture_prompt(*args, **kwargs):
            prompts_seen.append(mock_memory.custom_fact_extraction_prompt)
            return {"results": [{"id": "mem-1", "event": "ADD", "memory": "test"}]}

        mock_memory.add.side_effect = capture_prompt

        client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            "user_id": "u1",
            "prompt": custom_prompt,
        })

        assert prompts_seen[0] == custom_prompt

    def test_prompt_takes_precedence_over_custom_instructions(self, client, mock_memory):
        """When both prompt and custom_instructions are provided, prompt wins."""
        prompts_seen = []

        def capture_prompt(*args, **kwargs):
            prompts_seen.append(mock_memory.custom_fact_extraction_prompt)
            return {"results": [{"id": "mem-1", "event": "ADD", "memory": "test"}]}

        mock_memory.add.side_effect = capture_prompt

        client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            "user_id": "u1",
            "prompt": "from prompt field",
            "custom_instructions": "from custom_instructions field",
        })

        assert prompts_seen[0] == "from prompt field"

    def test_no_prompt_uses_server_default(self, client, mock_memory):
        """When neither prompt nor custom_instructions is provided,
        the extraction prompt should remain unchanged."""
        original = mock_memory.custom_fact_extraction_prompt
        prompts_seen = []

        def capture_prompt(*args, **kwargs):
            prompts_seen.append(mock_memory.custom_fact_extraction_prompt)
            return {"results": [{"id": "mem-1", "event": "ADD", "memory": "test"}]}

        mock_memory.add.side_effect = capture_prompt

        client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            "user_id": "u1",
        })

        assert prompts_seen[0] == original


# ===========================================================================
# custom_categories field
# ===========================================================================

class TestCustomCategories:
    """Verify that custom_categories appends tagging instructions and triggers backfill."""

    def test_custom_categories_accepted(self, client, mock_memory):
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "I like pizza"}],
            "user_id": "u1",
            "custom_categories": [
                {"identity": "Name, location, occupation"},
                {"preferences": "Likes, dislikes, opinions"},
            ],
        })
        assert resp.status_code == 200

    def test_custom_categories_not_in_add_kwargs(self, client, mock_memory):
        """custom_categories should NOT leak into Memory.add() kwargs."""
        client.post("/memories", json={
            "messages": [{"role": "user", "content": "I like pizza"}],
            "user_id": "u1",
            "custom_categories": [{"identity": "Name"}],
        })
        _, kwargs = mock_memory.add.call_args
        assert "custom_categories" not in kwargs

    def test_custom_categories_appends_to_prompt(self, client, mock_memory):
        """When custom_categories is provided, the extraction prompt should
        contain category instructions including the category names."""
        prompts_seen = []

        def capture_prompt(*args, **kwargs):
            prompts_seen.append(mock_memory.custom_fact_extraction_prompt)
            return {"results": [{"id": "mem-1", "event": "ADD", "memory": "test"}]}

        mock_memory.add.side_effect = capture_prompt

        client.post("/memories", json={
            "messages": [{"role": "user", "content": "I like pizza"}],
            "user_id": "u1",
            "custom_categories": [
                {"identity": "Name, location"},
                {"preferences": "Likes and dislikes"},
            ],
        })

        assert len(prompts_seen) == 1
        prompt = prompts_seen[0]
        assert "identity" in prompt
        assert "preferences" in prompt
        assert "分类" in prompt  # Chinese category instructions

    def test_custom_categories_with_custom_instructions(self, client, mock_memory):
        """When both custom_instructions and custom_categories are provided,
        categories should be appended to the custom_instructions prompt."""
        prompts_seen = []

        def capture_prompt(*args, **kwargs):
            prompts_seen.append(mock_memory.custom_fact_extraction_prompt)
            return {"results": [{"id": "mem-1", "event": "ADD", "memory": "test"}]}

        mock_memory.add.side_effect = capture_prompt

        client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            "user_id": "u1",
            "custom_instructions": "My base prompt",
            "custom_categories": [{"projects": "Active projects"}],
        })

        prompt = prompts_seen[0]
        assert prompt.startswith("My base prompt")
        assert "projects" in prompt


# ===========================================================================
# _backfill_categories
# ===========================================================================

class TestBackfillCategories:
    """Verify that _backfill_categories correctly parses [tag] prefixes
    and calls MEMORY_INSTANCE.update() to store categories in metadata."""

    def test_backfill_strips_tag_and_updates(self, client, mock_memory):
        """When Memory.add() returns a memory with [tag] prefix,
        _backfill_categories should strip the tag and call update()."""
        mock_memory.add.return_value = {
            "results": [
                {"id": "mem-1", "event": "ADD", "memory": "[identity] 用户叫张三"},
                {"id": "mem-2", "event": "ADD", "memory": "[projects] 正在开发新功能"},
            ]
        }

        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "我叫张三，正在开发新功能"}],
            "user_id": "u1",
            "custom_categories": [
                {"identity": "Name, location"},
                {"projects": "Active projects"},
            ],
        })

        assert resp.status_code == 200

        # Verify update() was called for each tagged memory
        assert mock_memory.update.call_count == 2

        # Check first call
        update_calls = mock_memory.update.call_args_list
        assert update_calls[0] == call("mem-1", data="用户叫张三", metadata={"category": "identity"})
        assert update_calls[1] == call("mem-2", data="正在开发新功能", metadata={"category": "projects"})

    def test_backfill_batch_classifies_untagged_memories(self, client, mock_memory):
        """Memories without [tag] prefix should trigger batch LLM classification."""
        mock_memory.add.return_value = {
            "results": [
                {"id": "mem-1", "event": "ADD", "memory": "用户叫张三"},
            ]
        }
        # Mock the LLM for batch classification
        mock_memory.llm = MagicMock()
        mock_memory.llm.generate_response.return_value = '["identity"]'

        client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            "user_id": "u1",
            "custom_categories": [{"identity": "Name"}, {"projects": "Projects"}],
        })

        # LLM should be called for classification
        mock_memory.llm.generate_response.assert_called_once()
        # update() should be called with the classified category
        mock_memory.update.assert_called_once_with("mem-1", data="用户叫张三", metadata={"category": "identity"})

    def test_backfill_skips_non_add_events(self, client, mock_memory):
        """Only ADD and UPDATE events should trigger backfill."""
        mock_memory.add.return_value = {
            "results": [
                {"id": "mem-1", "event": "DELETE", "memory": "[identity] 旧记忆"},
                {"id": "mem-2", "event": "NOOP", "memory": "[identity] 已存在"},
            ]
        }

        client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            "user_id": "u1",
            "custom_categories": [{"identity": "Name"}],
        })

        mock_memory.update.assert_not_called()

    def test_backfill_updates_response_body(self, client, mock_memory):
        """The response should contain cleaned memory text and category field."""
        mock_memory.add.return_value = {
            "results": [
                {"id": "mem-1", "event": "ADD", "memory": "[preferences] 喜欢吃披萨"},
            ]
        }

        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "I like pizza"}],
            "user_id": "u1",
            "custom_categories": [{"preferences": "Likes"}],
        })

        data = resp.json()
        result = data["results"][0]
        assert result["memory"] == "喜欢吃披萨"
        assert result["category"] == "preferences"

    def test_no_backfill_without_custom_categories(self, client, mock_memory):
        """When custom_categories is not provided, no backfill should happen
        even if the memory text happens to contain [tag] prefix."""
        mock_memory.add.return_value = {
            "results": [
                {"id": "mem-1", "event": "ADD", "memory": "[identity] 张三"},
            ]
        }

        client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            "user_id": "u1",
        })

        mock_memory.update.assert_not_called()

    def test_backfill_handles_update_failure_gracefully(self, client, mock_memory):
        """If update() fails, the request should still succeed."""
        mock_memory.add.return_value = {
            "results": [
                {"id": "mem-1", "event": "ADD", "memory": "[identity] 张三"},
            ]
        }
        mock_memory.update.side_effect = RuntimeError("update failed")

        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            "user_id": "u1",
            "custom_categories": [{"identity": "Name"}],
        })

        # Request should still succeed despite update failure
        assert resp.status_code == 200

    def test_backfill_handles_llm_failure_gracefully(self, client, mock_memory):
        """If batch LLM classification fails, the request should still succeed."""
        mock_memory.add.return_value = {
            "results": [
                {"id": "mem-1", "event": "ADD", "memory": "用户叫张三"},
            ]
        }
        mock_memory.llm = MagicMock()
        mock_memory.llm.generate_response.side_effect = RuntimeError("LLM down")

        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            "user_id": "u1",
            "custom_categories": [{"identity": "Name"}],
        })

        assert resp.status_code == 200
        mock_memory.update.assert_not_called()


# ===========================================================================
# Search: filters parameter (existing, should pass through)
# ===========================================================================

class TestSearchFilters:
    """Verify that the filters parameter is forwarded to Memory.search()."""

    def test_category_filter_forwarded(self, client, mock_memory):
        resp = client.post("/search", json={
            "query": "projects",
            "user_id": "u1",
            "filters": {"category": "projects"},
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert kwargs["filters"] == {"category": "projects"}

    def test_category_in_filter_forwarded(self, client, mock_memory):
        resp = client.post("/search", json={
            "query": "test",
            "user_id": "u1",
            "filters": {"category": {"in": ["identity", "projects"]}},
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert kwargs["filters"] == {"category": {"in": ["identity", "projects"]}}

    def test_no_filters_omitted(self, client, mock_memory):
        resp = client.post("/search", json={"query": "test", "user_id": "u1"})
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert "filters" not in kwargs


# ===========================================================================
# OpenAPI schema validation
# ===========================================================================

class TestOpenAPISchema:
    """Verify that new fields appear in the OpenAPI schema."""

    def test_custom_instructions_in_schema(self, client):
        resp = client.get("/openapi.json")
        schemas = resp.json()["components"]["schemas"]
        props = schemas["MemoryCreate"]["properties"]
        assert "custom_instructions" in props

    def test_custom_categories_in_schema(self, client):
        resp = client.get("/openapi.json")
        schemas = resp.json()["components"]["schemas"]
        props = schemas["MemoryCreate"]["properties"]
        assert "custom_categories" in props

    def test_filters_in_search_schema(self, client):
        resp = client.get("/openapi.json")
        schemas = resp.json()["components"]["schemas"]
        props = schemas["SearchRequest"]["properties"]
        assert "filters" in props
