import json
import re
from typing import Any, Dict, List, Union

from mem0.configs.rerankers.base import BaseRerankerConfig
from mem0.configs.rerankers.llm import LLMRerankerConfig
from mem0.reranker.base import BaseReranker
from mem0.utils.factory import LlmFactory


class LLMReranker(BaseReranker):
    def __init__(self, config: Union[BaseRerankerConfig, LLMRerankerConfig, Dict]):
        if isinstance(config, dict):
            config = LLMRerankerConfig(**config)
        elif isinstance(config, BaseRerankerConfig) and not isinstance(config, LLMRerankerConfig):
            config = LLMRerankerConfig(
                provider=getattr(config, "provider", "openai"),
                model=getattr(config, "model", "gpt-4o-mini"),
                api_key=getattr(config, "api_key", None),
                top_k=getattr(config, "top_k", None),
                temperature=0.0,
                max_tokens=500,
            )

        self.config = config

        if self.config.llm:
            nested = self.config.llm
            llm_provider = nested.get("provider", self.config.provider)
            llm_config: dict = dict(nested.get("config") or {})
            llm_config.setdefault("model", self.config.model)
            llm_config.setdefault("temperature", self.config.temperature)
            llm_config.setdefault("max_tokens", self.config.max_tokens)
            if self.config.api_key:
                llm_config.setdefault("api_key", self.config.api_key)
        else:
            llm_provider = self.config.provider
            llm_config = {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            if self.config.api_key:
                llm_config["api_key"] = self.config.api_key

        self.llm = LlmFactory.create(llm_provider, llm_config)

    def _get_doc_text(self, doc: Dict[str, Any]) -> str:
        if "memory" in doc:
            return doc["memory"]
        elif "text" in doc:
            return doc["text"]
        elif "content" in doc:
            return doc["content"]
        return str(doc)

    def _build_batch_prompt(self, query: str, doc_texts: List[str]) -> str:
        docs_section = chr(10).join(
            f"[{i}] {text}" for i, text in enumerate(doc_texts)
        )
        return (
            "对每条记忆与查询的相关性打分（0.00-1.00），要求分数有区分度，避免全部相同。\n"
            "\n"
            "评分标准：\n"
            "- 0.90-1.00：直接回答查询，高度匹配\n"
            "- 0.70-0.89：主题相关，包含有用信息\n"
            "- 0.40-0.69：间接相关或部分匹配\n"
            "- 0.10-0.39：弱相关，仅有微弱联系\n"
            "- 0.00-0.09：完全无关\n"
            "\n"
            "示例：\n"
            '查询："项目部署状态"\n'
            "[0] 昨天完成了生产环境部署 -> 0.92\n"
            "[1] 项目使用 Docker 容器化 -> 0.55\n"
            "[2] 今天天气不错 -> 0.02\n"
            "输出：[0.92, 0.55, 0.02]\n"
            "\n"
            f'现在请打分：\n'
            f'查询："{query}"\n'
            f"\n"
            f"记忆：\n"
            f"{docs_section}\n"
            f"\n"
            "仅输出 JSON 数组，如 [0.85, 0.42, 0.15]"
        )

    def _parse_scores(self, response: str, n: int) -> List[float]:
        match = re.search(r"\[([\d.,\s]+)\]", response)
        if match:
            try:
                scores = json.loads("[" + match.group(1) + "]")
                if len(scores) == n:
                    return [min(max(float(s), 0.0), 1.0) for s in scores]
            except (json.JSONDecodeError, ValueError):
                pass

        pattern = r"\b([01](?:\.\d+)?)\b"
        matches = re.findall(pattern, response)
        scores = [float(m) for m in matches[:n]]

        while len(scores) < n:
            scores.append(0.5)
        return [min(max(s, 0.0), 1.0) for s in scores[:n]]

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        if not documents:
            return documents

        doc_texts = [self._get_doc_text(doc) for doc in documents]

        try:
            prompt = self._build_batch_prompt(query, doc_texts)
            response = self.llm.generate_response(
                messages=[{"role": "user", "content": prompt}]
            )
            scores = self._parse_scores(response, len(documents))
        except Exception:
            scores = [0.5] * len(documents)

        scored_docs = []
        for doc, score in zip(documents, scores):
            scored_doc = doc.copy()
            scored_doc["rerank_score"] = score
            scored_docs.append(scored_doc)

        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        limit = top_k or self.config.top_k
        if limit:
            scored_docs = scored_docs[:limit]

        return scored_docs

