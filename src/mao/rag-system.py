import os
from typing import List, Optional
import logging
import time

# LangChain Core Imports
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

# LangChain Retrievers and Transformers
from langchain.retrievers import ContextualCompressionRetriever, ParentDocumentRetriever
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    DocumentCompressorPipeline,
)
from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter
from langchain.retrievers import MultiQueryRetriever

# LangChain Document Loaders and Text Splitters
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

# Vector Store and Embeddings
from langchain_community.vectorstores import Qdrant
import qdrant_client
from qdrant_client.http import models
from fastembed import TextEmbedding

# MCP (Message Creation Protocol) Adapter for LangChain
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# LiteLLM for Multi-Provider Support
import litellm

# LangChain LiteLLM Integration
from langchain_community.chat_models import ChatLiteLLM

# For Hybrid Search
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Cache Implementation
import diskcache
from threading import Lock
from contextlib import contextmanager

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FastEmbeddingsModel(Embeddings):
    """
    Implementation of LangChain Embeddings interface using FastEmbed.

    FastEmbed is a lightweight, efficient embedding library optimized for production use,
    supporting various multilingual embedding models.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the FastEmbed-based embedding model.

        Args:
            model_name: The name of the embedding model to use. Default is 'BAAI/bge-small-en-v1.5'.
            cache_dir: Directory to cache the model files. Default is None (uses default cache location).
            max_retries: Maximum number of retries for embedding operations. Default is 3.
            retry_delay: Delay between retries in seconds. Default is 1.0.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize the embedding model
        self.embedding_model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

        # Get the embedding dimension for initializing vector stores
        sample_text = "Sample text for dimension detection"
        sample_embedding = list(self.embedding_model.embed([sample_text]))[0]
        self.embedding_dimension = len(sample_embedding)

        logger.info(
            f"Initialized FastEmbed model {model_name} with dimension {self.embedding_dimension}"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embeddings, each embedding is a list of floats.
        """
        for attempt in range(self.max_retries):
            try:
                # FastEmbed returns a generator, convert to list of numpy arrays
                embeddings = list(self.embedding_model.embed(texts))
                # Convert numpy arrays to lists
                return [embedding.tolist() for embedding in embeddings]
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Embedding attempt {attempt+1} failed: {str(e)}. Retrying in {self.retry_delay}s..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error(
                        f"Failed to generate embeddings after {self.max_retries} attempts: {str(e)}"
                    )
                    raise
        return []

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embeddings for a single query text.

        Args:
            text: The query text to embed.

        Returns:
            A single embedding as a list of floats.
        """
        for attempt in range(self.max_retries):
            try:
                # FastEmbed returns a generator, extract the first item
                embedding = list(self.embedding_model.embed([text]))[0]
                return embedding.tolist()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Query embedding attempt {attempt+1} failed: {str(e)}. Retrying in {self.retry_delay}s..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error(
                        f"Failed to generate query embedding after {self.max_retries} attempts: {str(e)}"
                    )
                    raise
        return []


class EmbeddingCache:
    """
    Cache for embeddings to reduce computation and API calls.

    Implements a disk-based cache for document and query embeddings
    with thread-safe operations.
    """

    def __init__(self, cache_dir: str = "./.cache/embeddings"):
        """
        Initialize the embedding cache.

        Args:
            cache_dir: Directory to store the cache. Default is './.cache/embeddings'.
        """
        self.cache = diskcache.Cache(cache_dir)
        self.lock = Lock()

    @contextmanager
    def _acquire(self):
        """Context manager for thread-safe cache operations."""
        self.lock.acquire()
        try:
            yield
        finally:
            self.lock.release()

    def get(self, key: str) -> Optional[List[float]]:
        """
        Retrieve an embedding from the cache.

        Args:
            key: The cache key (typically a text or document ID).

        Returns:
            The cached embedding or None if not found.
        """
        with self._acquire():
            return self.cache.get(key)

    def set(
        self, key: str, embedding: List[float], expire: Optional[int] = None
    ) -> None:
        """
        Store an embedding in the cache.

        Args:
            key: The cache key (typically a text or document ID).
            embedding: The embedding to cache.
            expire: Optional expiration time in seconds.
        """
        with self._acquire():
            self.cache.set(key, embedding, expire=expire)

    def clear(self) -> None:
        """Clear the entire cache."""
        with self._acquire():
            self.cache.clear()


class CachedEmbeddings(Embeddings):
    """
    Wrapper around an embedding model that caches results.

    This reduces API calls and computation for frequently embedded texts.
    """

    def __init__(
        self,
        underlying_embeddings: Embeddings,
        cache: Optional[EmbeddingCache] = None,
        namespace: str = "default",
    ):
        """
        Initialize the cached embeddings wrapper.

        Args:
            underlying_embeddings: The actual embedding model to use.
            cache: Optional embedding cache instance. If None, a new one will be created.
            namespace: Namespace to use for cache keys to avoid collisions.
        """
        self.underlying_embeddings = underlying_embeddings
        self.cache = cache or EmbeddingCache()
        self.namespace = namespace

        # For compatibility with vector stores that need to know the embedding dimension
        if hasattr(underlying_embeddings, "embedding_dimension"):
            self.embedding_dimension = underlying_embeddings.embedding_dimension

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text."""
        # Simple hash function for cache keys
        import hashlib

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self.namespace}:{text_hash}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents, using cache when available.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embeddings, each embedding is a list of floats.
        """
        result: List[List[float]] = []
        texts_to_embed: List[str] = []
        indices_to_embed: List[int] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cached_embedding = self.cache.get(cache_key)

            if cached_embedding is not None:
                result.append(cached_embedding)
            else:
                # Need to compute this embedding
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        # If we have texts that need embedding
        if texts_to_embed:
            # Generate embeddings for texts not in cache
            new_embeddings = self.underlying_embeddings.embed_documents(texts_to_embed)

            # Cache the new embeddings
            for text, embedding in zip(texts_to_embed, new_embeddings):
                cache_key = self._get_cache_key(text)
                self.cache.set(cache_key, embedding)

            # Insert the new embeddings into the result at the correct positions
            result_with_new = list(result)  # Copy the result list
            for i, embedding in zip(indices_to_embed, new_embeddings):
                # Ensure the result list is long enough
                while len(result_with_new) <= i:
                    result_with_new.append([])
                result_with_new[i] = embedding

            # Filter out any None values (should not happen if indices are continuous)
            result = [e for e in result_with_new if e is not None]

        return result

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embeddings for a single query text, using cache when available.

        Args:
            text: The query text to embed.

        Returns:
            A single embedding as a list of floats.
        """
        cache_key = self._get_cache_key(text)
        cached_embedding = self.cache.get(cache_key)

        if cached_embedding is not None:
            return cached_embedding

        # Cache miss - generate the embedding
        embedding = self.underlying_embeddings.embed_query(text)

        # Cache the result
        self.cache.set(cache_key, embedding)

        return embedding


class EnhancedRAGSystem:
    """
    An advanced RAG system leveraging LangChain and Qdrant with modern techniques.

    Features:
    - Multiple embedding models via FastEmbed
    - Embedding caching for efficiency
    - Contextual compression and hybrid search
    - Multiple retrieval strategies
    - Multi-provider support via LiteLLM
    - Parent-child chunking
    """

    def __init__(
        self,
        collection_name: str,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        context_window_size: int = 16000,
        enable_caching: bool = True,
        enable_hybrid_search: bool = True,
        hybrid_search_ratio: float = 0.5,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the enhanced RAG system.

        Args:
            collection_name: Name of the Qdrant collection
            embedding_model_name: Name of the FastEmbed model
            llm_provider: LLM provider (openai, anthropic, etc.)
            llm_model: Specific model to use with the provider
            qdrant_url: URL to Qdrant instance (optional, default: local instance)
            qdrant_api_key: API key for Qdrant (optional)
            api_key: API key for the LLM provider
            context_window_size: Size of the context window for the LLM
            enable_caching: Whether to enable embedding caching
            enable_hybrid_search: Whether to enable hybrid search
            hybrid_search_ratio: Ratio between semantic and keyword search (0-1)
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
        """
        # Core settings
        self.collection_name = collection_name
        self.llm_provider = llm_provider
        self.llm_model = llm_model

        # API keys and connection parameters
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key or os.environ.get("QDRANT_API_KEY")
        self.api_key = api_key or os.environ.get(f"{llm_provider.upper()}_API_KEY")

        # Feature flags
        self.enable_caching = enable_caching
        self.enable_hybrid_search = enable_hybrid_search
        self.hybrid_search_ratio = max(
            0.0, min(1.0, hybrid_search_ratio)
        )  # Ensure value is between 0 and 1

        # Context and chunking parameters
        self.context_window_size = context_window_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize components
        self._setup_embedding_model(embedding_model_name)
        self._setup_qdrant_client()
        self._setup_vector_store()
        self._setup_llm()

        # Text splitters for document processing
        self._setup_text_splitters()

        # Retrievers
        self.retriever = self._create_standard_retriever()

        # BM25 for hybrid search
        if self.enable_hybrid_search:
            self.bm25_index: Optional[BM25Okapi] = None
            self.bm25_corpus: List[List[str]] = []
            self.document_lookup: dict[str, Document] = {}

        logger.info(
            f"Enhanced RAG System initialized with collection '{collection_name}'"
        )

    def _setup_embedding_model(self, model_name: str):
        """Set up the embedding model with optional caching."""
        base_embeddings = FastEmbeddingsModel(model_name=model_name)

        if self.enable_caching:
            self.embedding_model: Embeddings = CachedEmbeddings(
                underlying_embeddings=base_embeddings,
                namespace=f"fastembed:{model_name}",
            )
            self.embedding_dimension = base_embeddings.embedding_dimension
            logger.info(
                f"Using cached {model_name} embeddings with dimension {self.embedding_dimension}"
            )
        else:
            self.embedding_model = base_embeddings
            self.embedding_dimension = base_embeddings.embedding_dimension
            logger.info(
                f"Using {model_name} embeddings with dimension {self.embedding_dimension}"
            )

    def _setup_qdrant_client(self):
        """Initialize the Qdrant client."""
        if self.qdrant_url:
            self.qdrant_client = qdrant_client.QdrantClient(
                url=self.qdrant_url, api_key=self.qdrant_api_key
            )
            logger.info(f"Connected to Qdrant at {self.qdrant_url}")
        else:
            # Local instance
            self.qdrant_client = qdrant_client.QdrantClient(location=":memory:")
            logger.info("Using in-memory Qdrant instance")

    def _setup_vector_store(self):
        """Initialize the vector store."""
        # Check if collection exists
        collections = self.qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            # Create collection if it doesn't exist
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dimension, distance=models.Distance.COSINE
                ),
            )
            logger.info(f"Created new Qdrant collection: {self.collection_name}")

        # Initialize vector store
        self.vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.embedding_model,
        )
        logger.info(
            f"Initialized Qdrant vector store with collection: {self.collection_name}"
        )

    def _setup_llm(self):
        """Set up the LLM using LiteLLM."""
        # Configure LiteLLM
        litellm.api_key = self.api_key

        # Create ChatLiteLLM instance
        self.llm = ChatLiteLLM(
            model=f"{self.llm_provider}/{self.llm_model}",
            temperature=0,
            max_tokens=1024,
        )

        logger.info(f"Initialized LLM: {self.llm_provider}/{self.llm_model}")

    def _setup_text_splitters(self):
        """Set up text splitters for document processing."""
        # Standard recursive text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        # Parent document splitter for hierarchical chunking
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 5,  # Larger chunks for parents
            chunk_overlap=self.chunk_overlap * 2,
        )

        # Child document splitter for detailed chunks
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        # Markdown header splitter for structured documents
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
            ]
        )

    def _create_standard_retriever(self, k: int = 5) -> BaseRetriever:
        """
        Create a standard retriever from the vector store.

        Args:
            k: Number of documents to retrieve. Default is 5.

        Returns:
            A retriever configured for similarity search.
        """
        return self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )

    def _create_contextual_retriever(self, k: int = 5) -> BaseRetriever:
        """
        Create a contextual compression retriever.

        Args:
            k: Number of documents to retrieve initially. Default is 5.

        Returns:
            A contextual compression retriever.
        """
        # Create a document compressor using the LLM
        document_compressor = LLMChainExtractor.from_llm(self.llm)

        # Create embeddings filter to remove irrelevant documents
        embeddings_filter = EmbeddingsFilter(
            embeddings=self.embedding_model,
            similarity_threshold=0.76,  # Adjust based on your requirements
        )

        # Create a compression pipeline
        compression_pipeline = DocumentCompressorPipeline(
            transformers=[embeddings_filter, document_compressor]
        )

        # Create the base retriever
        base_retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )

        # Create and return the contextual retriever
        return ContextualCompressionRetriever(
            base_compressor=compression_pipeline, base_retriever=base_retriever
        )

    def _create_parent_child_retriever(self, k: int = 5) -> BaseRetriever:
        """
        Create a parent-child retriever for hierarchical document retrieval.

        Args:
            k: Number of documents to retrieve. Default is 5.

        Returns:
            A parent-child retriever.
        """
        # We need to create a separate vector store for the child documents
        child_collection_name = f"{self.collection_name}_children"

        # Check if collection exists
        collections = self.qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if child_collection_name not in collection_names:
            # Create collection if it doesn't exist
            self.qdrant_client.create_collection(
                collection_name=child_collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dimension, distance=models.Distance.COSINE
                ),
            )

        # Initialize child vector store
        child_vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=child_collection_name,
            embeddings=self.embedding_model,
        )

        # Create and return parent-child retriever
        return ParentDocumentRetriever(
            vectorstore=child_vector_store,
            parent_splitter=self.parent_splitter,
            child_splitter=self.child_splitter,
            k=k,
        )

    def _create_multi_query_retriever(self, k: int = 5) -> BaseRetriever:
        """
        Create a multi-query retriever that generates variations of the query.

        Args:
            k: Number of documents to retrieve per query. Default is 5.

        Returns:
            A multi-query retriever.
        """
        # Create the base retriever
        base_retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )

        # Create and return the multi-query retriever
        return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=self.llm)

    def add_documents(
        self,
        documents: List[Document],
        generate_contextual_embeddings: bool = True,
        retriever_type: str = "standard",
    ) -> int:
        """
        Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects.
            generate_contextual_embeddings: Whether to generate contextual embeddings.
            retriever_type: Type of retriever to use ("standard", "contextual",
                           "parent_child", "multi_query").

        Returns:
            Number of chunks added to the vector store.
        """
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)

        # Generate contextual embeddings if requested
        if generate_contextual_embeddings:
            chunks = self._generate_contextual_embeddings(chunks)

        # Add documents to vector store
        self.vector_store.add_documents(chunks)

        # Update BM25 index for hybrid search if enabled
        if self.enable_hybrid_search:
            self._update_bm25_index(chunks)

        # Create appropriate retriever based on type
        if retriever_type == "contextual":
            self.retriever = self._create_contextual_retriever()
        elif retriever_type == "parent_child":
            self.retriever = self._create_parent_child_retriever()
        elif retriever_type == "multi_query":
            self.retriever = self._create_multi_query_retriever()
        else:
            self.retriever = self._create_standard_retriever()

        return len(chunks)

    def _generate_contextual_embeddings(self, chunks: List[Document]) -> List[Document]:
        """
        Generate contextual embeddings for document chunks.

        Args:
            chunks: List of document chunks.

        Returns:
            List of documents with contextualized content.
        """
        CONTEXT_PROMPT = ChatPromptTemplate.from_messages([HumanMessage(content="""
            Given the following document chunk, provide a concise context (2-3 sentences) 
            that situates this chunk within a broader scope. Focus on key entities, relationships, 
            and the main topic to improve search retrieval.
            
            DOCUMENT CHUNK:
            {chunk_content}
            
            CONCISE CONTEXT:
            """)])

        contextualized_chunks = []

        for chunk in chunks:
            try:
                # Generate context for the chunk
                context_response = self.llm.invoke(
                    CONTEXT_PROMPT.format(chunk_content=chunk.page_content)
                )

                # Create a new document with contextualized content
                contextualized_chunk = Document(
                    page_content=f"{chunk.page_content}\n\nCONTEXT: {context_response.content}",
                    metadata=chunk.metadata.copy(),
                )
                # Store original content in metadata
                contextualized_chunk.metadata["original_content"] = chunk.page_content
                contextualized_chunk.metadata["context"] = context_response.content

                contextualized_chunks.append(contextualized_chunk)
            except Exception as e:
                logger.warning(f"Failed to generate context for chunk: {str(e)}")
                # Fall back to original chunk if contextualization fails
                contextualized_chunks.append(chunk)

        return contextualized_chunks

    def _update_bm25_index(self, chunks: List[Document]):
        """
        Update the BM25 index for hybrid search.

        Args:
            chunks: List of document chunks.
        """
        # Load NLTK resources if not already loaded
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        # Get English stopwords
        stop_words = set(stopwords.words("english"))

        # Process documents for BM25
        for i, doc in enumerate(chunks):
            # Get the document content
            content = doc.metadata.get("original_content", doc.page_content)

            # Tokenize and remove stopwords
            tokens = [
                word.lower()
                for word in word_tokenize(content)
                if word.isalnum() and word.lower() not in stop_words
            ]

            # Add to corpus
            self.bm25_corpus.append(tokens)

            # Store document in lookup dictionary
            doc_id = str(i)
            self.document_lookup[doc_id] = doc

        # Create or update BM25 index
        self.bm25_index = BM25Okapi(self.bm25_corpus)

        logger.info(f"Updated BM25 index with {len(self.bm25_corpus)} documents")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        use_hybrid_search: Optional[bool] = None,
        retriever_type: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The query text.
            k: Number of documents to retrieve. Default is 5.
            use_hybrid_search: Whether to use hybrid search. Overrides class setting
                              if provided.
            retriever_type: Type of retriever to use for this query. If None, uses the
                           current retriever.

        Returns:
            List of retrieved documents.
        """
        use_hybrid = (
            self.enable_hybrid_search
            if use_hybrid_search is None
            else use_hybrid_search
        )

        # Select retriever based on type
        if retriever_type:
            if retriever_type == "standard":
                active_retriever = self._create_standard_retriever(k)
            elif retriever_type == "contextual":
                active_retriever = self._create_contextual_retriever(k)
            elif retriever_type == "parent_child":
                active_retriever = self._create_parent_child_retriever(k)
            elif retriever_type == "multi_query":
                active_retriever = self._create_multi_query_retriever(k)
            else:
                active_retriever = self.retriever
        else:
            active_retriever = self.retriever

        # Perform retrieval
        if use_hybrid and self.bm25_index is not None:
            return self._hybrid_retrieval(query, k)
        else:
            return active_retriever.invoke(query)

    def _hybrid_retrieval(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform hybrid retrieval combining vector search and BM25.

        Args:
            query: The query text.
            k: Number of documents to retrieve. Default is 5.

        Returns:
            List of retrieved documents.
        """
        if self.bm25_index is None:
            logger.warning(
                "BM25 index is not initialized. Falling back to vector search."
            )
            return self.vector_store.similarity_search(query, k=k)
        # Tokenize query for BM25
        tokens = [
            word.lower()
            for word in word_tokenize(query)
            if word.isalnum() and word.lower() not in stopwords.words("english")
        ]

        # Get BM25 scores
        bm25_scores = self.bm25_index.get_scores(tokens)

        # Get top BM25 results
        bm25_k = int(k * 2)  # Get more results to have a good pool for fusion
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:bm25_k]
        bm25_results = [self.document_lookup[str(idx)] for idx in top_bm25_indices]

        # Get vector search results
        vector_k = int(k * 2)  # Get more results to have a good pool for fusion
        vector_results = self.vector_store.similarity_search(query, k=vector_k)

        # Perform reciprocal rank fusion
        return self._reciprocal_rank_fusion(
            query,
            bm25_results,
            vector_results,
            k=k,
            weight_vector=self.hybrid_search_ratio,
            weight_bm25=1.0 - self.hybrid_search_ratio,
        )

    def _reciprocal_rank_fusion(
        self,
        query: str,
        bm25_results: List[Document],
        vector_results: List[Document],
        k: int = 5,
        weight_vector: float = 0.5,
        weight_bm25: float = 0.5,
    ) -> List[Document]:
        """
        Combine results using Reciprocal Rank Fusion.

        Args:
            query: The original query.
            bm25_results: Results from BM25 search.
            vector_results: Results from vector search.
            k: Number of results to return. Default is 5.
            weight_vector: Weight for vector search results (0-1). Default is 0.5.
            weight_bm25: Weight for BM25 results (0-1). Default is 0.5.

        Returns:
            Combined list of documents.
        """
        # Create a map of document content to scores and rank
        doc_scores = {}

        # Process BM25 results
        for rank, doc in enumerate(bm25_results):
            content = doc.page_content
            if content not in doc_scores:
                doc_scores[content] = {
                    "doc": doc,
                    "bm25_rank": rank + 1,
                    "vector_rank": None,
                }
            else:
                doc_scores[content]["bm25_rank"] = rank + 1

        # Process vector results
        for rank, doc in enumerate(vector_results):
            content = doc.page_content
            if content not in doc_scores:
                doc_scores[content] = {
                    "doc": doc,
                    "bm25_rank": None,
                    "vector_rank": rank + 1,
                }
            else:
                doc_scores[content]["vector_rank"] = rank + 1

        # Calculate fusion scores
        fusion_scores = []
        for content, score_data in doc_scores.items():
            bm25_contribution: float = 0.0
            raw_bm25_rank = score_data["bm25_rank"]
            if raw_bm25_rank is not None:
                bm25_contribution = weight_bm25 * (1.0 / (raw_bm25_rank + 60))  # type: ignore[operator]

            vector_contribution: float = 0.0
            raw_vector_rank = score_data["vector_rank"]
            if raw_vector_rank is not None:
                vector_contribution = weight_vector * (
                    1.0 / (raw_vector_rank + 60)  # type: ignore[operator]
                )

            fusion_score = bm25_contribution + vector_contribution
            fusion_scores.append((fusion_score, score_data["doc"]))

        # Sort by fusion score (descending)
        fusion_scores.sort(reverse=True)

        # Return top k documents
        return [doc for _, doc in fusion_scores[:k]]  # type: ignore[misc]

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        LangChain BaseRetriever compatible method to get relevant documents.

        Args:
            query: The query string.
            **kwargs: Additional arguments for the retrieve method.

        Returns:
            List of relevant documents.
        """
        return self.retrieve(query, **kwargs)

    def rag_query(
        self,
        query: str,
        k: int = 5,
        use_hybrid_search: Optional[bool] = None,
        retriever_type: Optional[str] = None,
        query_transformation: bool = False,
        rerank_results: bool = False,
        stream: bool = False,
    ) -> str:
        """
        Perform a RAG query using the configured system.

        Args:
            query: The user query.
            k: Number of documents to retrieve. Default is 5.
            use_hybrid_search: Whether to use hybrid search.
            retriever_type: Type of retriever to use.
            query_transformation: Whether to transform the query for better retrieval.
            rerank_results: Whether to rerank results using the LLM.
            stream: Whether to stream the response.

        Returns:
            The generated response.
        """
        # Transform query if requested
        if query_transformation:
            query = self._transform_query(query)

        # Retrieve relevant documents
        docs = self.retrieve(
            query=query,
            k=k,
            use_hybrid_search=use_hybrid_search,
            retriever_type=retriever_type,
        )

        # Rerank if requested
        if rerank_results and len(docs) > 1:
            docs = self._rerank_documents(query, docs)

        # Format context from documents
        context = self._format_context(docs)

        # Create RAG prompt
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessage(
                    content="""You are a helpful assistant that answers questions based on the provided context. 
If the context doesn't contain relevant information, acknowledge this and provide a general response.

CONTEXT:
{context}

USER QUESTION:
{query}"""
                )
            ]
        )

        # Generate response
        if stream:
            # Stream the response
            return self.llm.stream(rag_prompt.format(context=context, query=query))
        else:
            # Return the full response
            return self.llm.invoke(
                rag_prompt.format(context=context, query=query)
            ).content

    def _transform_query(self, query: str) -> str:
        """
        Transform a user query to improve retrieval performance.

        Args:
            query: The original user query.

        Returns:
            The transformed query.
        """
        transform_prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessage(
                    content="""Please rewrite the following user query to make it more effective for retrieving relevant information from a knowledge base. 
Add important keywords and make it more specific, while preserving the original intent. Do not add artificial constraints not implied by the original query.

USER QUERY:
{query}

REWRITTEN QUERY:"""
                )
            ]
        )

        try:
            response = self.llm.invoke(transform_prompt.format(query=query))
            transformed_query = response.content.strip()
            logger.info(f"Transformed query: '{query}' -> '{transformed_query}'")
            return transformed_query
        except Exception as e:
            logger.warning(f"Query transformation failed: {str(e)}")
            return query

    def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Rerank retrieved documents based on relevance to the query.

        Args:
            query: The user query.
            docs: List of retrieved documents.

        Returns:
            Reranked list of documents.
        """
        rerank_prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessage(
                    content="""Analyze these retrieved documents and rank them by relevance to the user's query. \
Return a comma-separated list of document indices (0-based) from most to least relevant.\n\nUSER QUERY:\n{query}\n\nDOCUMENTS:\n{documents}\n\nRANKED INDICES (comma-separated):"""
                )
            ]
        )

        # Format documents for the prompt
        documents_text = "\n\n".join(
            [f"DOCUMENT {i}:\n{doc.page_content}" for i, doc in enumerate(docs)]
        )

        try:
            response = self.llm.invoke(
                rerank_prompt.format(query=query, documents=documents_text)
            )

            # Parse the response
            ranked_indices_str = response.content.strip()
            try:
                # Handle potential formatting issues
                ranked_indices_str = ranked_indices_str.replace("[", "").replace(
                    "]", ""
                )
                ranked_indices = [
                    int(idx.strip()) for idx in ranked_indices_str.split(",")
                ]

                # Ensure all indices are valid
                valid_indices = [idx for idx in ranked_indices if 0 <= idx < len(docs)]

                # Add any missing indices at the end
                all_indices = set(range(len(docs)))
                missing_indices = all_indices - set(valid_indices)
                valid_indices.extend(missing_indices)

                # Rerank the documents
                reranked_docs = [docs[idx] for idx in valid_indices]
                return reranked_docs
            except Exception as e:
                logger.warning(
                    f"Failed to parse reranking response, returning original order: {e}"
                )
                return docs
        except Exception as e:
            logger.warning(f"Reranking failed: {str(e)}")
            return docs

    def _format_context(self, docs: List[Document]) -> str:
        """
        Format retrieved documents into a context string.

        Args:
            docs: List of retrieved documents.

        Returns:
            Formatted context string.
        """
        formatted_context = ""

        for i, doc in enumerate(docs):
            # Extract metadata
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page", "")
            page_info = f", Page {page}" if page else ""

            # Format document
            formatted_context += (
                f"Document {i+1} [Source: {source}{page_info}]:\n{doc.page_content}\n\n"
            )

        return formatted_context.strip()


# Example usage
if __name__ == "__main__":
    # Create the RAG system
    rag_system = EnhancedRAGSystem(
        collection_name="my_knowledge_base",
        embedding_model_name="BAAI/bge-small-en-v1.5",  # Fast and effective
        llm_provider="openai",  # Can be changed to anthropic, etc.
        llm_model="gpt-4o",
        enable_hybrid_search=True,
    )

    # Load sample documents
    sample_docs = [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models.",
            metadata={"source": "langchain_docs.pdf", "page": 1},
        ),
        Document(
            page_content="Qdrant is a vector database for storing embeddings and performing similarity search.",
            metadata={"source": "vector_db_comparison.pdf", "page": 15},
        ),
    ]

    # Add documents
    rag_system.add_documents(sample_docs, generate_contextual_embeddings=True)

    # Perform a query
    response = rag_system.rag_query(
        "How can I use LangChain with vector databases?", k=3, use_hybrid_search=True
    )

    print(response)
