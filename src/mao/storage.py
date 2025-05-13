"""
KnowledgeTree and ExperienceTree: Qdrant-based vector stores for agent knowledge and experience.
"""

from typing import List, Dict, Any, Optional, Tuple, TypedDict, Callable
import logging
import os
import uuid
import time
import asyncio
from contextlib import asynccontextmanager

# Type definitions
from typing_extensions import NotRequired  # Python 3.12+ standard

# Modern Qdrant client with async API
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from qdrant_client.http.exceptions import (
    UnexpectedResponse,
    ApiException,
    ResponseHandlingException,
)

# OpenAI embeddings with batched requests, async API and dimensions parameter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.embeddings import Embeddings
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

# Environment variables with typing
EMBED_MODEL: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
QDRANT_URL: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
VECTOR_NAME: str = "default"
BATCH_SIZE: int = int(os.environ.get("QDRANT_BATCH_SIZE", "32"))


# Typed dictionary for search results
class SearchResult(TypedDict):
    id: str
    score: float
    page_content: str
    tags: List[str]
    relations: NotRequired[List[Dict[str, Any]]]


# --- Embedding Model Factory ---
class EmbeddingProvider:
    """Factory for embedding models with proper dimension handling"""

    @staticmethod
    async def create_embeddings() -> Tuple[Embeddings, int]:
        """Create embedding model and return it with its dimension"""
        embed_dim = None

        # First try OpenAI models (most reliable dimensions)
        try:
            if "text-embedding-3" in EMBED_MODEL:
                embed = OpenAIEmbeddings(model=EMBED_MODEL, dimensions=1536)
                embed_dim = embed.dimensions
                logging.info(
                    f"Using OpenAI embeddings: {EMBED_MODEL} with dimensions {embed_dim}"
                )
                return embed, embed_dim
        except Exception as e:
            logging.info(f"Could not use OpenAI embeddings: {e}")

        # Finally try FastEmbed
        try:
            embed = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", parallel=0)
            # Try to get dimension from model properties
            if hasattr(embed, "embedding_size") and embed.embedding_size:
                embed_dim = embed.embedding_size
            elif hasattr(embed, "dim") and embed.dim:
                embed_dim = embed.dim
            else:
                # Infer from a test embedding
                embed_dim = len(embed.embed_query("test"))
            logging.info(
                f"Using FastEmbed embeddings: {embed.model_name} with dimensions {embed_dim}"
            )
            return embed, embed_dim
        except Exception as e:
            logging.error(f"Failed to initialize any embedding model: {e}")
            raise RuntimeError(f"Could not initialize any embedding model: {e}")


class QdrantOperationError(Exception):
    """Custom exception for Qdrant operations that fail after retries."""

    pass


# Improved retry configuration
DEFAULT_RETRY_STOP = stop_after_attempt(3)
DEFAULT_RETRY_WAIT = wait_exponential(multiplier=1, min=2, max=10)
# Nur tatsächlich vorhandene Exception-Typen
QDRANT_RETRY_EXCEPTION = (UnexpectedResponse, ApiException, ResponseHandlingException)


class VectorStoreBase:
    """Base class with common vector store functionality and async methods"""

    def __init__(
        self,
        url: str = QDRANT_URL,
        collection_name: str = "default_collection",
        recreate_on_dim_mismatch: bool = False,
        embedding_provider: Optional[Callable[[], Tuple[Embeddings, int]]] = None,
    ):
        self.collection_name = collection_name
        self.qdrant_url = url
        self.vector_name = VECTOR_NAME
        self.recreate_on_dim_mismatch = recreate_on_dim_mismatch

        # Synchronous client for backward compatibility
        self.client = QdrantClient(url=self.qdrant_url)

        # Async client for modern async ops
        self.async_client = AsyncQdrantClient(url=self.qdrant_url)

        # Initialize embeddings (will be set in async_init)
        self.embed = None
        self.embed_dim = None
        self._embedding_provider = (
            embedding_provider or EmbeddingProvider.create_embeddings
        )

    async def async_init(self) -> "VectorStoreBase":
        """Async initialization method"""
        self.embed, self.embed_dim = await self._embedding_provider()

        # Ensure collection exists
        await self._ensure_collection_async()

        logging.info(
            f"{self.__class__.__name__}: Using embedding dim {self.embed_dim} "
            f"for collection '{self.collection_name}' at {self.qdrant_url}"
        )
        return self

    @classmethod
    async def create(
        cls,
        url: str = QDRANT_URL,
        collection_name: str = "default_collection",
        recreate_on_dim_mismatch: bool = False,
        embedding_provider: Optional[Callable[[], Tuple[Embeddings, int]]] = None,
    ) -> "VectorStoreBase":
        """Factory method for async initialization"""
        instance = cls(
            url, collection_name, recreate_on_dim_mismatch, embedding_provider
        )
        return await instance.async_init()

    async def _ensure_collection_async(self) -> None:
        """Ensure collection exists with correct vector dimension"""
        try:
            collections = await self.async_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            exists = self.collection_name in collection_names

            if exists:
                info = await self.async_client.get_collection(self.collection_name)
                vectors_cfg = info.config.params.vectors
                col_dim = None

                if isinstance(vectors_cfg, dict) and self.vector_name in vectors_cfg:
                    col_dim = vectors_cfg[self.vector_name].size
                elif hasattr(vectors_cfg, "size"):
                    col_dim = vectors_cfg.size

                if col_dim is None:
                    logging.warning(
                        f"Could not determine vector dimension for existing collection "
                        f"'{self.collection_name}'. Assuming compatible."
                    )
                elif col_dim != self.embed_dim:
                    msg = f"Collection '{self.collection_name}' has dimension {col_dim}, but model expects {self.embed_dim}."
                    if self.recreate_on_dim_mismatch:
                        logging.warning(
                            f"{msg} RECREATING collection based on recreate_on_dim_mismatch=True."
                        )
                        await self.async_client.recreate_collection(
                            collection_name=self.collection_name,
                            vectors_config={
                                self.vector_name: VectorParams(
                                    size=self.embed_dim, distance=Distance.COSINE
                                )
                            },
                        )
                    else:
                        logging.error(
                            f"{msg} NOT recreating automatically. Manual intervention may be required."
                        )
            else:
                logging.info(
                    f"Collection '{self.collection_name}' does not exist. Creating now with dim {self.embed_dim}."
                )
                await self.async_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        self.vector_name: VectorParams(
                            size=self.embed_dim, distance=Distance.COSINE
                        )
                    },
                )
            logging.info(
                f"Collection '{self.collection_name}' is ready with target dim {self.embed_dim}."
            )
        except Exception as e:
            logging.error(
                f"Error during collection setup for '{self.collection_name}': {e}"
            )
            raise QdrantOperationError(
                f"Failed to ensure collection '{self.collection_name}': {e}"
            ) from e

    def _ensure_collection(self) -> None:
        """Synchronous wrapper for _ensure_collection_async"""
        # This is only used for backward compatibility in the old __init__
        if self.embed is None or self.embed_dim is None:
            # Initialize embeddings synchronously if needed
            self.embed, self.embed_dim = asyncio.run(
                EmbeddingProvider.create_embeddings()
            )

        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            exists = self.collection_name in collection_names

            if exists:
                info = self.client.get_collection(self.collection_name)
                vectors_cfg = info.config.params.vectors
                col_dim = None

                if isinstance(vectors_cfg, dict) and self.vector_name in vectors_cfg:
                    col_dim = vectors_cfg[self.vector_name].size
                elif hasattr(vectors_cfg, "size"):
                    col_dim = vectors_cfg.size

                if col_dim is None:
                    logging.warning(
                        f"Could not determine vector dimension for existing collection "
                        f"'{self.collection_name}'. Assuming compatible."
                    )
                elif col_dim != self.embed_dim:
                    msg = f"Collection '{self.collection_name}' has dimension {col_dim}, but model expects {self.embed_dim}."
                    if self.recreate_on_dim_mismatch:
                        logging.warning(
                            f"{msg} RECREATING collection based on recreate_on_dim_mismatch=True."
                        )
                        self.client.recreate_collection(
                            collection_name=self.collection_name,
                            vectors_config={
                                self.vector_name: VectorParams(
                                    size=self.embed_dim, distance=Distance.COSINE
                                )
                            },
                        )
                    else:
                        logging.error(
                            f"{msg} NOT recreating automatically. Manual intervention may be required."
                        )
            else:
                logging.info(
                    f"Collection '{self.collection_name}' does not exist. Creating now with dim {self.embed_dim}."
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        self.vector_name: VectorParams(
                            size=self.embed_dim, distance=Distance.COSINE
                        )
                    },
                )
            logging.info(
                f"Collection '{self.collection_name}' is ready with target dim {self.embed_dim}."
            )
        except Exception as e:
            logging.error(
                f"Error during collection setup for '{self.collection_name}': {e}"
            )
            raise QdrantOperationError(
                f"Failed to ensure collection '{self.collection_name}': {e}"
            ) from e

    async def wait_for_index(
        self, timeout: float = 2.0, check_interval: float = 0.1
    ) -> bool:
        """
        Wait for Qdrant index to be fully updated.

        Args:
            timeout: Maximum wait time in seconds
            check_interval: Check interval in seconds

        Returns:
            bool: True if index is updated, False otherwise
        """
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Check collection status
                collection_info = await self.async_client.get_collection(
                    self.collection_name
                )
                if (
                    hasattr(collection_info, "status")
                    and collection_info.status == "green"
                ):
                    return True

                # Alternatively check health
                try:
                    health = await self.async_client.health()
                    if health and health.status == "ok":
                        return True
                except Exception:
                    pass

                await asyncio.sleep(check_interval)

            return False
        except Exception as e:
            logging.warning(f"Error checking index status: {e}")
            return False

    @retry(
        stop=DEFAULT_RETRY_STOP,
        wait=DEFAULT_RETRY_WAIT,
        retry=retry_if_exception_type(QDRANT_RETRY_EXCEPTION),
    )
    async def add_entry_async(self, text: str, tags: Optional[List[str]] = None) -> str:
        """
        Add entry to vector store asynchronously.

        Args:
            text: Text content to embed and store
            tags: Optional tags to attach to entry

        Returns:
            str: ID of the created entry
        """
        point_id = str(uuid.uuid4())
        try:
            # Embed query (still sync due to model limitations)
            vector = self.embed.embed_query(text)
            if len(vector) != self.embed_dim:
                logging.error(
                    f"CRITICAL: Embedding vector for entry '{point_id}' has dim {len(vector)}, "
                    f"but collection expects {self.embed_dim}!"
                )

            payload = {"text": text, "tags": tags or []}

            # Async upsert
            await self.async_client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id, vector={self.vector_name: vector}, payload=payload
                    )
                ],
                wait=True,
            )

            # Small delay to ensure index is updated
            await asyncio.sleep(0.1)

            logging.info(f"Added entry {point_id} to {self.collection_name}.")
            return point_id
        except RetryError as e:
            logging.error(
                f"Failed to add entry to '{self.collection_name}' after retries: {e}"
            )
            raise QdrantOperationError(f"Failed to add entry after retries: {e}") from e
        except Exception as e:
            logging.error(f"Failed to add entry to '{self.collection_name}': {e}")
            raise QdrantOperationError(f"Failed to add entry: {e}") from e

    # Synchronous wrapper for backward compatibility
    def add_entry(self, text: str, tags: Optional[List[str]] = None) -> str:
        """Synchronous wrapper for add_entry_async"""
        return asyncio.run(self.add_entry_async(text, tags))

    @retry(
        stop=DEFAULT_RETRY_STOP,
        wait=DEFAULT_RETRY_WAIT,
        retry=retry_if_exception_type(QDRANT_RETRY_EXCEPTION),
    )
    async def search_async(self, query: str, k: int = 3) -> List[SearchResult]:
        """
        Search for similar vectors asynchronously.

        Args:
            query: Query text to search for
            k: Number of results to return

        Returns:
            List[SearchResult]: List of search results with scores and metadata
        """
        # Ensure index is updated
        await self.wait_for_index(timeout=1.0)

        try:
            # Embed query (still sync due to model limitations)
            vector = self.embed.embed_query(query)

            # Try multiple query formats with fallbacks
            try:
                # Modern format for vector name
                results = await self.async_client.query_points(
                    collection_name=self.collection_name,
                    query_vector=(self.vector_name, vector),
                    limit=k,
                    with_payload=True,
                )
            except Exception as e1:
                try:
                    # Legacy dict format
                    results = await self.async_client.query_points(
                        collection_name=self.collection_name,
                        query_vector={self.vector_name: vector},
                        limit=k,
                        with_payload=True,
                    )
                except Exception as e2:
                    try:
                        # Scroll as last resort (not vector search but returns latest)
                        scroll_results = await self.async_client.scroll(
                            collection_name=self.collection_name,
                            limit=k,
                            with_payload=True,
                        )
                        if hasattr(scroll_results, "points"):
                            results = scroll_results.points
                        else:
                            # Handle different scroll result formats
                            results = scroll_results[0] if scroll_results else []
                    except Exception as e3:
                        logging.error(f"All search attempts failed: {e1}, {e2}, {e3}")
                        return []

            # Format results
            formatted_hits: List[SearchResult] = []
            for r in results:
                try:
                    payload = r.payload if hasattr(r, "payload") and r.payload else {}
                    formatted_hit: SearchResult = {
                        "id": str(r.id),
                        "score": getattr(r, "score", 1.0),
                        "page_content": payload.get("text", ""),
                        "tags": payload.get("tags", []),
                        "relations": payload.get("relations", []),
                    }
                    formatted_hits.append(formatted_hit)
                except Exception as e:
                    logging.warning(f"Skipping malformed search result: {e}")

            return formatted_hits
        except Exception as e:
            logging.error(f"Search failed in '{self.collection_name}': {e}")
            return []

    # Synchronous wrapper for backward compatibility
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Synchronous wrapper for search_async"""
        return asyncio.run(self.search_async(query, k))

    @retry(
        stop=DEFAULT_RETRY_STOP,
        wait=DEFAULT_RETRY_WAIT,
        retry=retry_if_exception_type(QDRANT_RETRY_EXCEPTION),
    )
    async def delete_entry_async(self, point_id: str) -> bool:
        """Delete entry asynchronously"""
        try:
            result = await self.async_client.delete(
                collection_name=self.collection_name, points_selector=[point_id]
            )
            status = getattr(result, "status", None)
            success = status == "completed" if status else True
            if success:
                logging.info(f"Deleted entry {point_id} from {self.collection_name}")
            return success
        except Exception as e:
            logging.error(f"Failed to delete entry {point_id}: {e}")
            return False

    # Synchronous wrapper
    def delete_entry(self, point_id: str) -> bool:
        """Synchronous wrapper for delete_entry_async"""
        return asyncio.run(self.delete_entry_async(point_id))

    @retry(
        stop=DEFAULT_RETRY_STOP,
        wait=DEFAULT_RETRY_WAIT,
        retry=retry_if_exception_type(QDRANT_RETRY_EXCEPTION),
    )
    async def get_entry_async(self, point_id: str) -> Optional[Dict[str, Any]]:
        """Get entry by ID asynchronously"""
        try:
            res_list = await self.async_client.retrieve(
                collection_name=self.collection_name, ids=[point_id], with_payload=True
            )
            if not res_list:
                return None

            point_data = res_list[0]
            payload = point_data.payload if hasattr(point_data, "payload") else {}
            if payload is None:
                payload = {}

            entry = {
                "id": point_data.id,
                "page_content": payload.get("text", ""),
                **payload,
            }
            return entry
        except Exception as e:
            if "not found" in str(e).lower():
                return None
            logging.error(f"Error retrieving entry {point_id}: {e}")
            return None

    # Synchronous wrapper
    def get_entry(self, point_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous wrapper for get_entry_async"""
        return asyncio.run(self.get_entry_async(point_id))

    @retry(
        stop=DEFAULT_RETRY_STOP,
        wait=DEFAULT_RETRY_WAIT,
        retry=retry_if_exception_type(QDRANT_RETRY_EXCEPTION),
    )
    async def clear_all_points_async(self) -> None:
        """Clear all points from collection asynchronously"""
        try:
            await self.async_client.delete(
                collection_name=self.collection_name,
                points_selector=models.Filter(must=[]),
            )
        except Exception as e:
            logging.error(f"Failed to clear all points: {e}")
            raise

    # Synchronous wrapper
    def clear_all_points(self) -> bool:
        """Synchronous wrapper for clear_all_points_async"""
        return asyncio.run(self.clear_all_points_async())

    async def add_tag_async(self, point_id: str, tag: str) -> bool:
        """Add tag to entry asynchronously"""
        entry = await self.get_entry_async(point_id)
        if not entry:
            logging.warning(f"Cannot add tag to non-existent entry {point_id}")
            return False

        tags = set(entry.get("tags", []))
        tags.add(tag)

        try:
            await self.async_client.set_payload(
                collection_name=self.collection_name,
                payload={"tags": list(tags)},
                points=[point_id],
            )
            return True
        except Exception as e:
            logging.error(f"Failed to add tag to {point_id}: {e}")
            return False

    # Synchronous wrapper
    def add_tag(self, point_id: str, tag: str) -> bool:
        """Synchronous wrapper for add_tag_async"""
        return asyncio.run(self.add_tag_async(point_id, tag))

    async def get_tags_async(self, point_id: str) -> List[str]:
        """Get tags for entry asynchronously"""
        entry = await self.get_entry_async(point_id)
        return entry.get("tags", []) if entry else []

    # Synchronous wrapper
    def get_tags(self, point_id: str) -> List[str]:
        """Synchronous wrapper for get_tags_async"""
        return asyncio.run(self.get_tags_async(point_id))

    async def add_relation_async(
        self, from_id: str, to_id: str, rel_type: str = "related"
    ) -> bool:
        """Add relation between entries asynchronously"""
        entry = await self.get_entry_async(from_id)
        if not entry:
            logging.warning(f"Cannot add relation from non-existent entry {from_id}")
            return False

        rels = entry.get("relations", [])
        if not any(r.get("id") == to_id and r.get("type") == rel_type for r in rels):
            rels.append({"id": to_id, "type": rel_type})

            try:
                await self.async_client.set_payload(
                    collection_name=self.collection_name,
                    payload={"relations": rels},
                    points=[from_id],
                )
                return True
            except Exception as e:
                logging.error(f"Failed to add relation from {from_id} to {to_id}: {e}")
                return False
        return True  # Relation already exists

    # Synchronous wrapper
    def add_relation(self, from_id: str, to_id: str, rel_type: str = "related") -> bool:
        """Synchronous wrapper for add_relation_async"""
        return asyncio.run(self.add_relation_async(from_id, to_id, rel_type))

    async def get_relations_async(
        self, point_id: str, rel_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get relations for entry asynchronously"""
        entry = await self.get_entry_async(point_id)
        if not entry:
            return []

        rels = entry.get("relations", [])
        if rel_type:
            return [r for r in rels if r.get("type") == rel_type]
        return rels

    # Synchronous wrapper
    def get_relations(
        self, point_id: str, rel_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for get_relations_async"""
        return asyncio.run(self.get_relations_async(point_id, rel_type))

    async def remove_relation_async(
        self, from_id: str, to_id: str, rel_type: Optional[str] = None
    ) -> bool:
        """Remove relation between entries asynchronously"""
        entry = await self.get_entry_async(from_id)
        if not entry:
            return False

        rels = entry.get("relations", [])
        new_rels = [
            r
            for r in rels
            if not (
                r.get("id") == to_id and (rel_type is None or r.get("type") == rel_type)
            )
        ]

        if len(new_rels) < len(rels):
            try:
                await self.async_client.set_payload(
                    collection_name=self.collection_name,
                    payload={"relations": new_rels},
                    points=[from_id],
                )
                return True
            except Exception as e:
                logging.error(
                    f"Failed to remove relation from {from_id} to {to_id}: {e}"
                )
                return False
        return True  # No matching relation to remove

    # Synchronous wrapper
    def remove_relation(
        self, from_id: str, to_id: str, rel_type: Optional[str] = None
    ) -> bool:
        """Synchronous wrapper for remove_relation_async"""
        return asyncio.run(self.remove_relation_async(from_id, to_id, rel_type))

    # Additional helper methods
    @retry(
        stop=DEFAULT_RETRY_STOP,
        wait=DEFAULT_RETRY_WAIT,
        retry=retry_if_exception_type(QDRANT_RETRY_EXCEPTION),
    )
    async def add_entries_batch_async(
        self, texts: List[str], tags_list: Optional[List[List[str]]] = None
    ) -> List[str]:
        """Add multiple entries in batch asynchronously"""
        if not texts:
            return []

        if tags_list and len(texts) != len(tags_list):
            raise ValueError(
                f"Number of texts ({len(texts)}) must match number of tag lists ({len(tags_list)})"
            )

        point_ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        try:
            # Use batch embedding
            vectors = list(self.embed.embed_documents(texts))

            # Create batch points
            points = []
            for i, (text, vector) in enumerate(zip(texts, vectors)):
                tags = tags_list[i] if tags_list and i < len(tags_list) else []
                payload = {"text": text, "tags": tags}
                points.append(
                    PointStruct(
                        id=point_ids[i],
                        vector={self.vector_name: vector},
                        payload=payload,
                    )
                )

            # Process in batches
            for i in range(0, len(points), BATCH_SIZE):
                batch = points[i : i + BATCH_SIZE]
                await self.async_client.upsert(
                    collection_name=self.collection_name, points=batch, wait=True
                )

            logging.info(
                f"Added {len(texts)} entries in batch to {self.collection_name}"
            )
            return point_ids

        except Exception as e:
            logging.error(f"Failed to add entries in batch: {e}")
            raise QdrantOperationError(f"Failed to add entries in batch: {e}") from e

    # Synchronous wrapper
    def add_entries_batch(
        self, texts: List[str], tags_list: Optional[List[List[str]]] = None
    ) -> List[str]:
        """Synchronous wrapper for add_entries_batch_async"""
        return asyncio.run(self.add_entries_batch_async(texts, tags_list))

    async def traverse_async(
        self, start_id: str, depth: int = 1, rel_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Traverse the graph from a starting point asynchronously"""
        if depth <= 0:
            return []

        visited_ids = set()
        queue = [(start_id, 0)]
        result_nodes = []
        processed_edges = set()

        head = 0
        while head < len(queue):
            current_id, d = queue[head]
            head += 1

            if d > depth:
                continue

            if current_id not in visited_ids:
                visited_ids.add(current_id)
                current_entry = await self.get_entry_async(current_id)
                if not current_entry:
                    continue

                current_entry["depth"] = d
                result_nodes.append(current_entry)

            if d < depth:
                relations = await self.get_relations_async(current_id)
                for rel in relations:
                    target_id = rel.get("id")
                    relation_type = rel.get("type")

                    if not target_id:
                        continue

                    if rel_types is not None and relation_type not in rel_types:
                        continue

                    edge = (current_id, target_id, relation_type)
                    if edge in processed_edges:
                        continue

                    processed_edges.add(edge)
                    if target_id not in visited_ids:
                        queue.append((target_id, d + 1))

        return result_nodes

    # Synchronous wrapper
    def traverse(
        self, start_id: str, depth: int = 1, rel_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for traverse_async"""
        return asyncio.run(self.traverse_async(start_id, depth, rel_types))

    def add(self, text: str, tags: Optional[List[str]] = None) -> str:
        """Alias for add_entry"""
        return self.add_entry(text, tags)

    async def summarize_entry_async(self, entry_id: str) -> str:
        """Summarize entry asynchronously"""
        entry = await self.get_entry_async(entry_id)
        return entry.get("text", "") if entry else ""

    def summarize_entry(self, entry_id: str) -> str:
        """Synchronous wrapper for summarize_entry_async"""
        return asyncio.run(self.summarize_entry_async(entry_id))

    @asynccontextmanager
    async def batch_operations(self):
        """Context manager for batch operations with proper error handling"""
        try:
            yield self
        finally:
            # Ensure index is updated after batch operations
            await self.wait_for_index(timeout=1.0)


class KnowledgeTree(VectorStoreBase):
    """
    Qdrant-based vector store for knowledge entries.
    This class inherits all functionality from VectorStoreBase.
    """

    def __init__(
        self,
        url: str = QDRANT_URL,
        collection_name: str = "knowledge_tree",
        recreate_on_dim_mismatch: bool = False,
    ):
        super().__init__(
            url=url,
            collection_name=collection_name,
            recreate_on_dim_mismatch=recreate_on_dim_mismatch,
        )

    @classmethod
    async def create(
        cls,
        url: str = QDRANT_URL,
        collection_name: str = "knowledge_tree",
        recreate_on_dim_mismatch: bool = False,
        embedding_provider: Optional[Callable[[], Tuple[Embeddings, int]]] = None,
    ) -> "KnowledgeTree":
        """Factory method for async initialization"""
        instance = cls(url, collection_name, recreate_on_dim_mismatch)
        instance._embedding_provider = (
            embedding_provider or EmbeddingProvider.create_embeddings
        )
        return await instance.async_init()

    async def learn_from_experience_async(
        self,
        text: str,
        related_knowledge_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Learn from experience asynchronously"""
        exp_id = await self.add_entry_async(text, tags)

        if related_knowledge_id:
            knowledge_entry = await self.get_entry_async(related_knowledge_id)
            if knowledge_entry:
                await self.add_relation_async(
                    exp_id, related_knowledge_id, rel_type="knowledge"
                )
            else:
                logging.warning(
                    f"Could not link experience {exp_id} to non-existent knowledge entry {related_knowledge_id}"
                )

        return exp_id

    def learn_from_experience(
        self,
        text: str,
        related_knowledge_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Synchronous wrapper for learn_from_experience_async"""
        return asyncio.run(
            self.learn_from_experience_async(text, related_knowledge_id, tags)
        )

    async def learn_from_entry_async(self, entry_id: str, new_text: str) -> str:
        """Learn from entry asynchronously"""
        original_entry = await self.get_entry_async(entry_id)
        if not original_entry:
            logging.warning(
                f"Cannot learn from non-existent entry {entry_id}. Creating new entry directly."
            )
            return await self.add_entry_async(new_text)

        new_id = await self.add_entry_async(new_text)
        await self.add_relation_async(entry_id, new_id, rel_type="learned_from_this")
        await self.add_relation_async(new_id, entry_id, rel_type="learned_this_from")
        return new_id

    def learn_from_entry(self, entry_id: str, new_text: str) -> str:
        """Synchronous wrapper for learn_from_entry_async"""
        return asyncio.run(self.learn_from_entry_async(entry_id, new_text))


class ExperienceTree(KnowledgeTree):
    """
    Qdrant-based vector store for experience entries.
    Inherits all functionality from KnowledgeTree.
    """

    def __init__(
        self,
        url: str = QDRANT_URL,
        collection_name: str = "experience_tree",
        recreate_on_dim_mismatch: bool = False,
    ):
        super().__init__(
            url=url,
            collection_name=collection_name,
            recreate_on_dim_mismatch=recreate_on_dim_mismatch,
        )

    @classmethod
    async def create(
        cls,
        url: str = QDRANT_URL,
        collection_name: str = "experience_tree",
        recreate_on_dim_mismatch: bool = False,
        embedding_provider: Optional[Callable[[], Tuple[Embeddings, int]]] = None,
    ) -> "ExperienceTree":
        """Factory method for async initialization"""
        instance = cls(url, collection_name, recreate_on_dim_mismatch)
        instance._embedding_provider = (
            embedding_provider or EmbeddingProvider.create_embeddings
        )
        return await instance.async_init()


# Example usage (optional, for testing or demonstration)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # It's good practice to ensure QDRANT_URL and potentially EMBEDDING_MODEL_API_KEY are set
    # in the environment if not using defaults, especially for OpenAIEmbeddings.

    print(f"Attempting to use Qdrant at: {QDRANT_URL}")
    print(f"Embedding model: {EMBED_MODEL}")

    async def run_tests():
        """Run tests asynchronously"""
        try:
            # Test KnowledgeTree with async initialization
            kt = await KnowledgeTree.create(
                collection_name="test_knowledge_tree_prod",
                recreate_on_dim_mismatch=True,
            )
            await kt.clear_all_points_async()  # Start clean for test

            entry_id1 = await kt.add_entry_async(
                "The sky is blue.", tags=["nature", "color"]
            )
            entry_id2 = await kt.add_entry_async(
                "The sun is bright.", tags=["nature", "light"]
            )
            await kt.add_relation_async(entry_id1, entry_id2, "related_fact")

            print(f"KT Entry 1: {await kt.get_entry_async(entry_id1)}")
            search_results_kt = await kt.search_async("celestial bodies")
            print(f"KT Search for 'celestial bodies': {search_results_kt}")

            traversed_kt = await kt.traverse_async(entry_id1, depth=1)
            print(f"KT Traversal from {entry_id1} (depth 1): {traversed_kt}")

            # Test ExperienceTree with async initialization
            et = await ExperienceTree.create(
                collection_name="test_experience_tree_prod",
                recreate_on_dim_mismatch=True,
            )
            await et.clear_all_points_async()

            exp_id1 = await et.learn_from_experience_async(
                "User asked about weather. Agent said it is sunny.",
                related_knowledge_id=entry_id2,
                tags=["conversation"],
            )
            print(f"ET Experience 1: {await et.get_entry_async(exp_id1)}")

            search_results_et = await et.search_async("sunny weather conversation")
            print(f"ET Search for 'sunny weather conversation': {search_results_et}")

            # Test batch operations context manager
            async with kt.batch_operations():
                batch_ids = await kt.add_entries_batch_async(
                    ["Fact 1", "Fact 2", "Fact 3"],
                    [["batch", "test"], ["batch", "test"], ["batch", "test"]],
                )
                print(f"Added batch entries: {batch_ids}")

            print("Tests completed successfully.")

        except QdrantOperationError as qe:
            logging.error(f"A Qdrant operation failed during tests: {qe}")
        except Exception as e:
            logging.error(
                f"An unexpected error occurred during tests: {e}", exc_info=True
            )

    # Run tests with proper async event loop
    asyncio.run(run_tests())
