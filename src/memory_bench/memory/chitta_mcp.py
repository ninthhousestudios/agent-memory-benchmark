"""Chitta MCP memory provider for Agent Memory Benchmark.

Uses Chitta's hybrid search (vector + BM25 + entity overlap boost)
via the local Python API.
"""

import os
import sys
from pathlib import Path

from ..models import Document
from .base import MemoryProvider

_CHITTA_REPO = os.environ.get("CHITTA_REPO", "")


def _ensure_chitta():
    if _CHITTA_REPO:
        src_path = os.path.join(_CHITTA_REPO, "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
    os.environ.setdefault("DATABASE_BACKEND", "postgres")
    os.environ.setdefault("EMBEDDING_PROVIDER", "onnx")


class ChittaMCPMemoryProvider(MemoryProvider):
    name = "chitta-mcp"
    description = (
        "Chitta MCP: hybrid vector + BM25 search with entity overlap boost. "
        "Local Postgres + pgvector. Stores verbatim conversations and retrieves "
        "via Reciprocal Rank Fusion with optional read-time fact extraction."
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
        self.k = k
        self._profile_prefix = "amb_"
        self._extract_facts_enabled = extract_facts
        self._extractor_client = None
        # CHITTA_CHUNK_SIZE/CHUNK_OVERLAP override kwargs so the sweep
        # harness can vary them without editing the provider class.
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

    def initialize(self) -> None:
        _ensure_chitta()
        from chitta.config import settings
        from chitta import database
        settings._reset()
        database._reset_backend()

    def prepare(
        self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True
    ) -> None:
        _ensure_chitta()

    def cleanup(self) -> None:
        from chitta.database import _reset_backend

        _reset_backend()

    def _profile(self, user_id: str | None) -> str:
        return f"{self._profile_prefix}{user_id or 'default'}"

    @staticmethod
    def _extract_messages(doc: Document) -> list[dict] | None:
        import json

        messages = doc.messages
        if not messages and doc.content.strip().startswith("["):
            try:
                messages = json.loads(doc.content)
            except (json.JSONDecodeError, TypeError):
                pass
        if messages and isinstance(messages, list):
            return messages
        return None

    def ingest(self, documents: list[Document]) -> None:
        from chitta.service import store_memory_chunked

        for d in documents:
            profile = self._profile(d.user_id)
            tags = []
            if d.timestamp:
                tags.append(f"date:{d.timestamp}")
            prefix = f"[Date: {d.timestamp}]" if d.timestamp else None
            metadata = {"doc_id": d.id}

            messages = self._extract_messages(d)
            if messages:
                store_memory_chunked(
                    profile=profile,
                    messages=messages,
                    source="amb",
                    tags=tags,
                    metadata=metadata,
                    target_tokens=self.chunk_size,
                    overlap_tokens=self.chunk_overlap,
                    turns_per_chunk=self.turns_per_chunk,
                    overlap_turns=self.overlap_turns,
                    prefix=prefix,
                    auto_link=False,
                    follows_edges=True,
                )
            else:
                store_memory_chunked(
                    profile=profile,
                    content=d.content,
                    source="amb",
                    tags=tags,
                    metadata=metadata,
                    target_tokens=self.chunk_size,
                    overlap_tokens=self.chunk_overlap,
                    prefix=prefix,
                    auto_link=False,
                    follows_edges=True,
                )

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
        from chitta.service import search_memories_enriched

        profile = self._profile(user_id)
        results = search_memories_enriched(
            query=query,
            profile=profile,
            limit=k or self.k,
        )

        if not results:
            return [], None

        if self._extract_facts_enabled:
            raw_bundle_parts = []
            for i, r in enumerate(results):
                content = r.get("content", "")
                raw_bundle_parts.append(f"## Memory {i + 1}\n{content}")
            raw_bundle = "\n\n".join(raw_bundle_parts)

            facts = self._extract_facts(query, raw_bundle)
            return [Document(id="chitta-extracted-facts", content=facts)], None

        docs = []
        for r in results:
            content_parts = [r.get("content", "")]
            if r.get("relevance") is not None:
                content_parts.append(f"relevance: {r['relevance']:.3f}")
            docs.append(
                Document(
                    id=str(r.get("id", "")),
                    content="\n".join(content_parts),
                )
            )
        return docs, None
