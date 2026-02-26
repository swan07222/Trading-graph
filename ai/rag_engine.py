"""RAG engine for real-time knowledge grounding.

Fixes:
- No real-time knowledge: RAG provides up-to-date context
- Hallucinations: Ground responses in retrieved documents
- Context limits: Retrieve only relevant information
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


class DocumentSource(Enum):
    """Sources of knowledge documents."""
    MARKET_DATA = auto()
    NEWS = auto()
    SENTIMENT = auto()
    MODEL_OUTPUT = auto()
    USER_PROVIDED = auto()
    SYSTEM = auto()


@dataclass
class KnowledgeDocument:
    """A knowledge document for RAG retrieval."""
    doc_id: str
    content: str
    source: DocumentSource
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None
    created_at: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "source": self.source.name,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "created_at": self.created_at.isoformat(),
            "relevance_score": self.relevance_score,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeDocument:
        return cls(
            doc_id=data["doc_id"],
            content=data["content"],
            source=DocumentSource[data["source"]],
            metadata=data.get("metadata", {}),
            embedding=np.array(data["embedding"]) if data.get("embedding") else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            relevance_score=data.get("relevance_score", 0.0),
        )
    
    @property
    def token_estimate(self) -> int:
        """Estimate token count."""
        zh_chars = sum(1 for c in self.content if '\u4e00' <= c <= '\u9fff')
        other_chars = len(self.content) - zh_chars
        return (zh_chars // 2) + (other_chars // 4)


@dataclass
class RetrievalResult:
    """Result of a RAG retrieval."""
    document: KnowledgeDocument
    score: float
    rank: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "rank": self.rank,
        }


class RAGEngine:
    """Retrieval-Augmented Generation engine.
    
    Features:
    - Real-time knowledge injection
    - Multi-source document retrieval
    - Semantic search with embeddings
    - Automatic document expiration
    - Context-aware retrieval
    """
    
    def __init__(
        self,
        storage_dir: Path | None = None,
        embedding_dim: int = 384,  # Sentence-BERT default
        max_documents: int = 10000,
        default_top_k: int = 5,
        ttl_hours: int = 24,
    ) -> None:
        self.storage_dir = storage_dir or (CONFIG.cache_dir / "rag")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_dim = embedding_dim
        self.max_documents = max_documents
        self.default_top_k = default_top_k
        self.ttl_hours = ttl_hours
        
        self._documents: dict[str, KnowledgeDocument] = {}
        self._index: dict[str, list[str]] = {}  # Source -> doc_ids
        self._embedding_matrix: np.ndarray | None = None
        self._doc_ids: list[str] = []
        self._lock = threading.RLock()
        
        # Embedding model (lazy loaded)
        self._embedding_model: Any = None
        
        # Load existing documents
        self._load_documents()
        
        log.info(f"RAGEngine initialized: {self.storage_dir}")
    
    def _load_embedding_model(self) -> None:
        """Load sentence transformer for embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use multilingual model for Chinese/English support
            self._embedding_model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                cache_folder=str(CONFIG.cache_dir / "models"),
            )
            log.info("Loaded embedding model")
        except ImportError:
            log.warning("sentence-transformers not available, using keyword search")
        except Exception as e:
            log.warning(f"Failed to load embedding model: {e}")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        if self._embedding_model is None:
            # Fallback: simple keyword-based vector
            return self._keyword_vector(text)
        
        try:
            embedding = self._embedding_model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embedding.astype(np.float32)
        except Exception as e:
            log.warning(f"Embedding failed: {e}")
            return self._keyword_vector(text)
    
    def _keyword_vector(self, text: str) -> np.ndarray:
        """Create a simple keyword-based vector as fallback."""
        # Extract key terms and create a simple vector
        text_lower = text.lower()
        
        # Financial keywords
        keywords = [
            "buy", "sell", "hold", "stock", "price", "market",
            "上涨", "下跌", "买入", "卖出", "股票", "价格", "市场",
        ]
        
        vector = np.zeros(self.embedding_dim, dtype=np.float32)
        
        for i, kw in enumerate(keywords[:self.embedding_dim]):
            if kw.lower() in text_lower:
                vector[i] = 1.0
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def add_document(
        self,
        content: str,
        source: DocumentSource,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
        compute_embedding: bool = True,
    ) -> KnowledgeDocument:
        """Add a document to the knowledge base.
        
        Args:
            content: Document content
            source: Document source type
            metadata: Optional metadata
            doc_id: Optional document ID (auto-generated if None)
            compute_embedding: Whether to compute embedding
            
        Returns:
            Created KnowledgeDocument
        """
        import uuid
        
        with self._lock:
            doc_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"
            
            doc = KnowledgeDocument(
                doc_id=doc_id,
                content=content,
                source=source,
                metadata=metadata or {},
            )
            
            # Compute embedding
            if compute_embedding:
                doc.embedding = self._get_embedding(content)
            
            # Add to index
            self._documents[doc_id] = doc
            
            source_key = source.name
            if source_key not in self._index:
                self._index[source_key] = []
            self._index[source_key].append(doc_id)
            
            # Update embedding matrix
            self._rebuild_index()
            
            # Enforce max documents
            self._enforce_max_documents()
            
            log.debug(f"Added document: {doc_id} ({source.name})")
            return doc
    
    def add_market_data(
        self,
        symbol: str,
        data: dict[str, Any],
    ) -> KnowledgeDocument:
        """Add market data as a document.
        
        Args:
            symbol: Stock symbol
            data: Market data (price, volume, etc.)
            
        Returns:
            Created KnowledgeDocument
        """
        content = (
            f"Stock {symbol}: "
            f"Price: {data.get('price', 'N/A')}, "
            f"Change: {data.get('change', 'N/A')}, "
            f"Volume: {data.get('volume', 'N/A')}, "
            f"High: {data.get('high', 'N/A')}, "
            f"Low: {data.get('low', 'N/A')}, "
            f"Open: {data.get('open', 'N/A')}, "
            f"Close: {data.get('close', 'N/A')}"
        )
        
        return self.add_document(
            content=content,
            source=DocumentSource.MARKET_DATA,
            metadata={
                "symbol": symbol,
                **data,
            },
            doc_id=f"market_{symbol}_{int(time.time())}",
        )
    
    def add_news(
        self,
        title: str,
        content: str,
        source: str = "",
        category: str = "",
        symbols: list[str] | None = None,
    ) -> KnowledgeDocument:
        """Add news article as a document."""
        full_content = f"{title}\n\n{content}"
        
        return self.add_document(
            content=full_content,
            source=DocumentSource.NEWS,
            metadata={
                "title": title,
                "source": source,
                "category": category,
                "symbols": symbols or [],
            },
        )
    
    def add_sentiment(
        self,
        symbol: str,
        sentiment_score: float,
        analysis: str,
    ) -> KnowledgeDocument:
        """Add sentiment analysis as a document."""
        content = (
            f"Sentiment analysis for {symbol}: "
            f"Score: {sentiment_score:.2f}. {analysis}"
        )
        
        return self.add_document(
            content=content,
            source=DocumentSource.SENTIMENT,
            metadata={
                "symbol": symbol,
                "sentiment_score": sentiment_score,
            },
        )
    
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        sources: list[DocumentSource] | None = None,
        min_score: float = 0.3,
        max_age_hours: float | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results (default: default_top_k)
            sources: Filter by sources
            min_score: Minimum relevance score
            max_age_hours: Maximum document age
            
        Returns:
            List of RetrievalResult sorted by relevance
        """
        top_k = top_k or self.default_top_k
        
        with self._lock:
            if not self._documents:
                return []
            
            # Get query embedding
            query_embedding = self._get_embedding(query)
            
            results = []
            now = datetime.now()
            
            for doc_id, doc in self._documents.items():
                # Filter by source
                if sources and doc.source not in sources:
                    continue
                
                # Filter by age
                if max_age_hours:
                    age = now - doc.created_at
                    if age > timedelta(hours=max_age_hours):
                        continue
                
                # Compute similarity
                if doc.embedding is not None:
                    score = float(np.dot(query_embedding, doc.embedding))
                    # Normalize to 0-1 range
                    score = (score + 1) / 2
                else:
                    score = self._keyword_similarity(query, doc.content)
                
                if score >= min_score:
                    results.append(RetrievalResult(
                        document=doc,
                        score=score,
                        rank=0,  # Will be set after sorting
                    ))
            
            # Sort by score
            results.sort(key=lambda r: r.score, reverse=True)
            
            # Set ranks and limit
            for i, r in enumerate(results[:top_k]):
                r.rank = i + 1
            
            return results[:top_k]
    
    def _keyword_similarity(self, query: str, content: str) -> float:
        """Compute keyword-based similarity as fallback."""
        query_words = set(re.findall(r'\w+|[\u4e00-\u9fff]', query.lower()))
        content_words = set(re.findall(r'\w+|[\u4e00-\u9fff]', content.lower()))
        
        if not query_words or not content_words:
            return 0.0
        
        intersection = query_words & content_words
        union = query_words | content_words
        
        return len(intersection) / len(union) if union else 0.0
    
    def _rebuild_index(self) -> None:
        """Rebuild the embedding matrix for efficient search."""
        if not self._documents:
            self._embedding_matrix = None
            self._doc_ids = []
            return
        
        embeddings = []
        doc_ids = []
        
        for doc_id, doc in self._documents.items():
            if doc.embedding is not None:
                embeddings.append(doc.embedding)
                doc_ids.append(doc_id)
        
        if embeddings:
            self._embedding_matrix = np.stack(embeddings)
            self._doc_ids = doc_ids
        else:
            self._embedding_matrix = None
            self._doc_ids = []
    
    def _enforce_max_documents(self) -> None:
        """Remove old documents if over limit."""
        if len(self._documents) <= self.max_documents:
            return
        
        # Sort by created_at
        sorted_docs = sorted(
            self._documents.values(),
            key=lambda d: d.created_at,
        )
        
        # Remove oldest
        to_remove = len(self._documents) - self.max_documents
        for doc in sorted_docs[:to_remove]:
            self._documents.pop(doc.doc_id, None)
            if doc.doc_id in self._index.get(doc.source.name, []):
                self._index[doc.source.name].remove(doc.doc_id)
        
        self._rebuild_index()
    
    def _load_documents(self) -> None:
        """Load documents from disk."""
        doc_file = self.storage_dir / "documents.json"
        
        if not doc_file.exists():
            return
        
        try:
            with open(doc_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            now = datetime.now()
            for doc_data in data:
                doc = KnowledgeDocument.from_dict(doc_data)
                
                # Check TTL
                age = now - doc.created_at
                if age < timedelta(hours=self.ttl_hours):
                    self._documents[doc.doc_id] = doc
                    
                    source_key = doc.source.name
                    if source_key not in self._index:
                        self._index[source_key] = []
                    self._index[source_key].append(doc.doc_id)
            
            self._rebuild_index()
            log.info(f"Loaded {len(self._documents)} RAG documents")
            
        except Exception as e:
            log.warning(f"Failed to load RAG documents: {e}")
    
    def save_documents(self) -> None:
        """Save documents to disk."""
        with self._lock:
            doc_file = self.storage_dir / "documents.json"
            doc_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = [doc.to_dict() for doc in self._documents.values()]
            
            with open(doc_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            log.debug(f"Saved {len(self._documents)} RAG documents")
    
    def get_context_for_query(
        self,
        query: str,
        top_k: int | None = None,
        include_sources: bool = True,
    ) -> str:
        """Get formatted context string for LLM prompt.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            include_sources: Include source citations
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k=top_k)
        
        if not results:
            return "No relevant context found."
        
        context_parts = []
        
        for result in results:
            doc = result.document
            context = doc.content
            
            if include_sources:
                source_info = f"[Source: {doc.source.name}]"
                if doc.metadata.get("symbol"):
                    source_info += f" Symbol: {doc.metadata['symbol']}"
                if doc.metadata.get("title"):
                    source_info += f" Title: {doc.metadata['title']}"
                context = f"{source_info}\n{context}"
            
            context_parts.append(f"[{result.rank}]. {context}")
        
        return "\n\n".join(context_parts)
    
    def clear_expired(self) -> int:
        """Clear expired documents."""
        with self._lock:
            now = datetime.now()
            expired = []
            
            for doc_id, doc in self._documents.items():
                age = now - doc.created_at
                if age > timedelta(hours=self.ttl_hours):
                    expired.append(doc_id)
            
            for doc_id in expired:
                doc = self._documents.get(doc_id)
                if doc:
                    self._documents.pop(doc_id, None)
                    if doc_id in self._index.get(doc.source.name, []):
                        self._index[doc.source.name].remove(doc_id)
            
            if expired:
                self._rebuild_index()
                log.info(f"Cleared {len(expired)} expired RAG documents")
            
            return len(expired)
    
    def clear_all(self) -> None:
        """Clear all documents."""
        with self._lock:
            self._documents.clear()
            self._index.clear()
            self._embedding_matrix = None
            self._doc_ids = []
            
            # Clear disk storage
            doc_file = self.storage_dir / "documents.json"
            if doc_file.exists():
                doc_file.unlink()
            
            log.info("Cleared all RAG documents")
    
    def get_stats(self) -> dict[str, Any]:
        """Get RAG engine statistics."""
        with self._lock:
            source_counts = {
                source.name: len(doc_ids)
                for source, doc_ids in self._index.items()
            }
            
            return {
                "total_documents": len(self._documents),
                "sources": source_counts,
                "embedding_dim": self.embedding_dim,
                "embedding_model": (
                    self._embedding_model.__class__.__name__
                    if self._embedding_model else "None (keyword fallback)"
                ),
                "max_documents": self.max_documents,
                "ttl_hours": self.ttl_hours,
            }
    
    def shutdown(self) -> None:
        """Shutdown and save."""
        self.save_documents()
        log.info("RAGEngine shutdown complete")


# Singleton instance
_engine_instance: RAGEngine | None = None


def get_rag_engine(
    storage_dir: Path | None = None,
    **kwargs: Any,
) -> RAGEngine:
    """Get or create the singleton RAGEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngine(storage_dir, **kwargs)
    return _engine_instance
