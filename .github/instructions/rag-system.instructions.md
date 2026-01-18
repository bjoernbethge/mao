---
applyTo: "**/rag-system.py"
description: "Guidelines for RAG (Retrieval-Augmented Generation) system development"
---

# RAG System Development Guidelines

When working with the RAG system in `mao/rag-system.py`, follow these guidelines:

## Document Processing

1. **Implement robust chunking strategies** for different document types
2. **Preserve context** across chunk boundaries
3. **Handle metadata** (source, timestamps, authors) consistently
4. **Support multiple document formats** (text, markdown, PDF, etc.)

Example:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    async def process_document(self, content: str, metadata: dict) -> list[dict]:
        """
        Split document into chunks with metadata.
        
        Args:
            content: Document text content
            metadata: Document metadata (source, type, etc.)
        
        Returns:
            List of chunk dicts with content and metadata
        """
        chunks = self.splitter.split_text(content)
        
        return [
            {
                "content": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            }
            for i, chunk in enumerate(chunks)
        ]
```

## Embeddings

1. **Use consistent embedding models** across the system
2. **Cache embeddings** to reduce API calls and latency
3. **Batch embedding operations** for efficiency
4. **Handle embedding failures** gracefully

Example:
```python
from langchain_openai import OpenAIEmbeddings
from functools import lru_cache

class EmbeddingService:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(model=model)
        self._cache = {}
    
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts with caching.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors
        """
        # Check cache
        uncached_texts = []
        uncached_indices = []
        results = [None] * len(texts)
        
        for i, text in enumerate(texts):
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = await self.embeddings.aembed_documents(uncached_texts)
            for idx, embedding in zip(uncached_indices, new_embeddings):
                self._cache[texts[idx]] = embedding
                results[idx] = embedding
        
        return results
```

## Vector Storage

1. **Use Qdrant** for vector storage and similarity search
2. **Create collection schemas** with appropriate vector dimensions
3. **Implement proper indexing** for fast retrieval
4. **Use filters** for metadata-based search

Example:
```python
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class VectorStore:
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int = 1536
    ):
        """Create a collection for storing embeddings."""
        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
    
    async def add_documents(
        self,
        collection_name: str,
        documents: list[dict]
    ):
        """Add documents with embeddings to collection."""
        points = []
        for doc in documents:
            point = PointStruct(
                id=doc["id"],
                vector=doc["embedding"],
                payload={
                    "content": doc["content"],
                    "metadata": doc["metadata"]
                }
            )
            points.append(point)
        
        await self.client.upsert(
            collection_name=collection_name,
            points=points
        )
```

## Retrieval Strategies

1. **Implement semantic search** using vector similarity
2. **Support hybrid search** (vector + keyword when needed)
3. **Use reranking** for improved relevance
4. **Configure top_k** appropriately for context window

Example:
```python
async def retrieve(
    self,
    query: str,
    collection_name: str,
    top_k: int = 5,
    filters: dict | None = None
) -> list[dict]:
    """
    Retrieve relevant documents for a query.
    
    Args:
        query: Search query
        collection_name: Vector collection to search
        top_k: Number of results to return
        filters: Optional metadata filters
    
    Returns:
        List of relevant documents with scores
    """
    # Generate query embedding
    query_embedding = await self.embeddings.embed_query(query)
    
    # Search with filters
    results = await self.vector_store.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
        query_filter=filters
    )
    
    return [
        {
            "content": hit.payload["content"],
            "metadata": hit.payload["metadata"],
            "score": hit.score
        }
        for hit in results
    ]
```

## Context Assembly

1. **Assemble retrieved chunks** into coherent context
2. **Respect LLM context window** limits
3. **Prioritize by relevance score** when truncating
4. **Include source attribution** in assembled context

Example:
```python
def assemble_context(
    self,
    retrieved_docs: list[dict],
    max_tokens: int = 4000
) -> str:
    """
    Assemble retrieved documents into a context string.
    
    Args:
        retrieved_docs: List of retrieved documents with scores
        max_tokens: Maximum tokens for context (approximate)
    
    Returns:
        Formatted context string with sources
    """
    context_parts = []
    token_count = 0
    
    for doc in retrieved_docs:
        # Approximate tokens (4 chars per token)
        doc_tokens = len(doc["content"]) // 4
        
        if token_count + doc_tokens > max_tokens:
            break
        
        source = doc["metadata"].get("source", "Unknown")
        context_parts.append(
            f"[Source: {source}]\n{doc['content']}\n"
        )
        token_count += doc_tokens
    
    return "\n".join(context_parts)
```

## Response Generation

1. **Provide retrieved context** to LLM prompts
2. **Include source citations** in generated responses
3. **Handle cases** where no relevant documents are found
4. **Stream responses** for better UX

Example:
```python
async def generate_response(
    self,
    query: str,
    context: str,
    llm: ChatAnthropic
) -> str:
    """
    Generate a response using RAG.
    
    Args:
        query: User query
        context: Retrieved context
        llm: Language model instance
    
    Returns:
        Generated response with citations
    """
    prompt = f"""Answer the following question using the provided context.
Include citations to sources where appropriate.

Context:
{context}

Question: {query}

Answer:"""
    
    response = await llm.ainvoke(prompt)
    return response.content
```

## Performance Optimization

1. **Batch operations** where possible
2. **Use async operations** for I/O-bound tasks
3. **Implement caching** for frequently accessed documents
4. **Monitor retrieval latency** and optimize

Example:
```python
import asyncio

async def batch_retrieve(
    self,
    queries: list[str],
    collection_name: str
) -> list[list[dict]]:
    """Retrieve documents for multiple queries concurrently."""
    tasks = [
        self.retrieve(query, collection_name)
        for query in queries
    ]
    results = await asyncio.gather(*tasks)
    return results
```

## Testing

1. **Test document processing** with various formats
2. **Test retrieval quality** using evaluation metrics
3. **Mock LLM calls** for deterministic tests
4. **Test edge cases** (empty documents, no results, etc.)

Example:
```python
@pytest.mark.asyncio
async def test_rag_retrieval():
    rag_system = RAGSystem()
    
    # Add test documents
    docs = [
        {"content": "Python is a programming language", "metadata": {"topic": "python"}},
        {"content": "FastAPI is a web framework", "metadata": {"topic": "fastapi"}}
    ]
    await rag_system.add_documents("test_collection", docs)
    
    # Test retrieval
    results = await rag_system.retrieve("What is FastAPI?", "test_collection")
    
    assert len(results) > 0
    assert "FastAPI" in results[0]["content"]
    assert results[0]["score"] > 0.5
```

## Evaluation

1. **Measure retrieval quality** using metrics (precision, recall, MRR)
2. **Evaluate response quality** using LLM-as-judge or human eval
3. **Track retrieval latency** and throughput
4. **A/B test** different chunking and retrieval strategies
