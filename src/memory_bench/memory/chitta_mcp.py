"""Chitta MCP memory provider for Agent Memory Benchmark.

Talks to chitta-rs over streamable HTTP MCP (JSON-RPC).
Default endpoint: http://127.0.0.1:3100/mcp
"""

import json
import os
import uuid
from pathlib import Path

import httpx

from ..models import Document
from .base import MemoryProvider

_DEFAULT_URL = "http://127.0.0.1:3100/mcp"
_BEARER_TOKEN_PATH = os.path.expanduser("~/.config/chitta/bearer-token.txt")


def _auth_header() -> dict[str, str]:
    token_path = os.environ.get("CHITTA_TOKEN_FILE", _BEARER_TOKEN_PATH)
    token = os.environ.get("CHITTA_BEARER_TOKEN", "")
    if not token and os.path.isfile(token_path):
        token = Path(token_path).read_text().strip()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


class _McpClient:
    """Thin JSON-RPC client for chitta-rs streamable HTTP MCP."""

    def __init__(self, url: str, timeout: float = 120):
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        headers.update(_auth_header())
        self._url = url
        self._client = httpx.Client(headers=headers, timeout=timeout)
        self._id = 0
        self._session_id: str | None = None
        self._initialized = False

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _post(self, payload: dict) -> dict:
        resp = self._client.post(self._url, json=payload, headers=self._session_headers())
        if sid := resp.headers.get("mcp-session-id"):
            self._session_id = sid
        ct = resp.headers.get("content-type", "")
        if "text/event-stream" in ct:
            return self._parse_sse(resp.text)
        resp.raise_for_status()
        if not resp.content:
            return {}
        return resp.json()

    def _session_headers(self) -> dict[str, str]:
        if self._session_id:
            return {"Mcp-Session-Id": self._session_id}
        return {}

    @staticmethod
    def _parse_sse(text: str) -> dict:
        for line in reversed(text.splitlines()):
            if line.startswith("data: "):
                return json.loads(line[6:])
        raise RuntimeError(f"No data line in SSE response: {text[:500]}")

    def initialize(self) -> None:
        if self._initialized:
            return
        resp = self._post({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "amb", "version": "0.1.0"},
            },
        })
        if "error" in resp:
            raise RuntimeError(f"MCP initialize failed: {resp['error']}")
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self._initialized = True

    def call_tool(self, name: str, arguments: dict) -> dict:
        self.initialize()
        resp = self._post({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        if "error" in resp:
            raise RuntimeError(f"MCP tool {name} error: {resp['error']}")
        result = resp.get("result", {})
        content = result.get("content", [])
        if content and content[0].get("type") == "text":
            return json.loads(content[0]["text"])
        return result

    def close(self) -> None:
        self._client.close()


class ChittaMCPMemoryProvider(MemoryProvider):
    name = "chitta-mcp"
    description = (
        "Chitta MCP: hybrid vector + BM25 search with entity overlap boost. "
        "Local Postgres + pgvector via chitta-rs HTTP MCP server."
    )
    kind = "local"
    provider = "chitta-mcp"
    variant = "local"
    link = "https://github.com/josh/chitta"
    concurrency = 8

    def __init__(
        self,
        k: int = 20,
        extract_facts: bool = False,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        turns_per_chunk: int | None = None,
        overlap_turns: int | None = None,
    ):
        self.k = int(os.environ.get("CHITTA_K") or k)
        self._profile_prefix = "amb_"
        self._extract_facts_enabled = extract_facts
        self._extractor_client = None
        self.chunk_size = int(os.environ.get("CHITTA_CHUNK_SIZE") or chunk_size or 512)
        self.chunk_overlap = int(
            os.environ.get("CHITTA_CHUNK_OVERLAP") or chunk_overlap or 64
        )
        self.turns_per_chunk = int(
            os.environ.get("CHITTA_TURNS_PER_CHUNK") or turns_per_chunk or 4
        )
        self.overlap_turns = int(
            os.environ.get("CHITTA_OVERLAP_TURNS") or overlap_turns or 1
        )
        self._url = os.environ.get("CHITTA_RS_URL", _DEFAULT_URL)
        self._mcp: _McpClient | None = None

    @property
    def mcp(self) -> _McpClient:
        if self._mcp is None:
            self._mcp = _McpClient(self._url)
            self._mcp.initialize()
        return self._mcp

    def initialize(self) -> None:
        _ = self.mcp

    def prepare(
        self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True
    ) -> None:
        pass

    def cleanup(self) -> None:
        if self._mcp is not None:
            self._mcp.close()
            self._mcp = None

    def _profile(self, user_id: str | None) -> str:
        return f"{self._profile_prefix}{user_id or 'default'}"

    @staticmethod
    def _extract_messages(doc: Document) -> list[dict] | None:
        messages = doc.messages
        if not messages and doc.content.strip().startswith("["):
            try:
                messages = json.loads(doc.content)
            except (json.JSONDecodeError, TypeError):
                pass
        if messages and isinstance(messages, list):
            return messages
        return None

    def _chunk_text(self, text: str) -> list[str]:
        """Simple character-based chunking. Approximates token-based chunking
        at ~4 chars/token."""
        char_size = self.chunk_size * 4
        char_overlap = self.chunk_overlap * 4
        if len(text) <= char_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + char_size
            chunks.append(text[start:end])
            start = end - char_overlap
        return chunks

    def _chunk_messages(self, messages: list[dict]) -> list[str]:
        """Chunk conversation messages by turns."""
        if len(messages) <= self.turns_per_chunk:
            return [self._format_messages(messages)]
        chunks = []
        start = 0
        while start < len(messages):
            end = min(start + self.turns_per_chunk, len(messages))
            chunk_msgs = messages[start:end]
            chunks.append(self._format_messages(chunk_msgs))
            start = end - self.overlap_turns
            if start >= len(messages) or start <= (end - self.turns_per_chunk):
                break
        return chunks

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _store(self, profile: str, content: str, tags: list[str],
               metadata: dict, prefix: str | None = None) -> None:
        text = f"{prefix}\n{content}" if prefix else content
        chunks = self._chunk_text(text)
        for i, chunk in enumerate(chunks):
            key = f"{metadata.get('doc_id', '')}_{i}"
            try:
                self.mcp.call_tool("store_memory", {
                    "profile": profile,
                    "content": chunk,
                    "idempotency_key": key,
                    "source": "amb",
                    "tags": tags or None,
                    "metadata": metadata,
                })
            except RuntimeError as e:
                err = str(e)
                if "content_too_long" in err or "byte_length" in err:
                    sub_chunks = self._chunk_text(chunk)
                    for j, sub in enumerate(sub_chunks):
                        self.mcp.call_tool("store_memory", {
                            "profile": profile,
                            "content": sub,
                            "idempotency_key": f"{key}_{j}",
                            "source": "amb",
                            "tags": tags or None,
                            "metadata": metadata,
                        })
                else:
                    raise

    def ingest(self, documents: list[Document]) -> None:
        for d in documents:
            profile = self._profile(d.user_id)
            tags = []
            if d.timestamp:
                tags.append(f"date:{d.timestamp}")
            prefix = f"[Date: {d.timestamp}]" if d.timestamp else None
            metadata = {"doc_id": d.id}

            messages = self._extract_messages(d)
            if messages:
                chunks = self._chunk_messages(messages)
                for i, chunk in enumerate(chunks):
                    text = f"{prefix}\n{chunk}" if prefix else chunk
                    key = f"{d.id}_msg_{i}"
                    try:
                        self.mcp.call_tool("store_memory", {
                            "profile": profile,
                            "content": text,
                            "idempotency_key": key,
                            "source": "amb",
                            "tags": tags or None,
                            "metadata": metadata,
                        })
                    except RuntimeError as e:
                        if "content_too_long" in e.args[0] or "byte_length" in e.args[0]:
                            sub_chunks = self._chunk_text(text)
                            for j, sub in enumerate(sub_chunks):
                                self.mcp.call_tool("store_memory", {
                                    "profile": profile,
                                    "content": sub,
                                    "idempotency_key": f"{key}_{j}",
                                    "source": "amb",
                                    "tags": tags or None,
                                    "metadata": metadata,
                                })
                        else:
                            raise
            else:
                self._store(profile, d.content, tags, metadata, prefix)

    def _get_extractor(self):
        if self._extractor_client is None:
            provider = os.environ.get("CHITTA_EXTRACTOR_PROVIDER", "gemini")
            if provider == "openai":
                from openai import OpenAI
                self._extractor_client = ("openai", OpenAI(api_key=os.environ.get("OPENAI_API_KEY")))
            else:
                from google import genai
                api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                self._extractor_client = ("gemini", genai.Client(api_key=api_key))
        return self._extractor_client

    def _extract_facts(self, query: str, raw_content: str) -> str:
        prompt = f"""Given a user's question and conversation history, extract the facts most relevant to answering the question.

Question: {query}

Conversation history:
{raw_content}

Extract relevant facts as a concise bulleted list. Preserve specific details: names, numbers, dates, locations. If the history contains no relevant information, respond with "NO RELEVANT FACTS"."""

        try:
            provider, client = self._get_extractor()
            if provider == "openai":
                model = os.environ.get("CHITTA_EXTRACTOR_MODEL", "gpt-4.1-mini")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content or raw_content
            else:
                model = os.environ.get("CHITTA_EXTRACTOR_MODEL", "gemini-2.5-flash")
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                return response.text or raw_content
        except Exception:
            return raw_content

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        profile = self._profile(user_id)

        search_result = self.mcp.call_tool("search_memories", {
            "profile": profile,
            "query": query,
            "k": self.k,
        })

        hits = search_result.get("results", [])
        if not hits:
            return [], None

        # search returns 200-char snippets; fetch full content for each hit
        full_docs = []
        for hit in hits:
            mem_id = hit.get("id", "")
            try:
                full = self.mcp.call_tool("get_memory", {
                    "profile": profile,
                    "id": mem_id,
                })
                content = full.get("content", hit.get("snippet", ""))
            except Exception:
                content = hit.get("snippet", "")

            similarity = hit.get("similarity")
            content_parts = [content]
            if similarity is not None:
                content_parts.append(f"relevance: {similarity:.3f}")
            full_docs.append(Document(
                id=str(mem_id),
                content="\n".join(content_parts),
            ))

        if self._extract_facts_enabled:
            raw_bundle_parts = []
            for i, doc in enumerate(full_docs):
                raw_bundle_parts.append(f"## Memory {i + 1}\n{doc.content}")
            raw_bundle = "\n\n".join(raw_bundle_parts)
            facts = self._extract_facts(query, raw_bundle)
            return [Document(id="chitta-extracted-facts", content=facts)], None

        return full_docs, None
