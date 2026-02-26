# models/news_embeddings.py
"""Enhanced text embeddings for news and policy analysis.

This module provides:
- Multiple embedding strategies (TF-IDF, Word2Vec, BERT, etc.)
- Chinese text support
- Sentiment-aware embeddings
- Multi-granularity text representations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TextEmbeddingResult:
    """Result of text embedding computation."""
    
    # Primary embedding
    embedding: np.ndarray
    
    # Metadata
    text_length: int = 0
    embedding_dim: int = 0
    method: str = ""
    
    # Quality metrics
    confidence: float = 1.0
    language: str = "zh"
    
    # Additional features
    sentiment_score: float = 0.0
    topic_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    keywords: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "embedding": self.embedding.tolist() if len(self.embedding) > 0 else [],
            "text_length": self.text_length,
            "embedding_dim": self.embedding_dim,
            "method": self.method,
            "confidence": float(self.confidence),
            "language": self.language,
            "sentiment_score": float(self.sentiment_score),
            "keywords": self.keywords,
        }


class EmbeddingModel:
    """Text embedding model with multiple strategies."""
    
    def __init__(
        self,
        method: str = "tfidf",
        embedding_dim: int = 256,
        max_vocab_size: int = 30000,
        model_path: Path | None = None,
        use_sentiment: bool = True,
        language: str = "zh",
    ) -> None:
        """
        Args:
            method: Embedding method ('tfidf', 'word2vec', 'fasttext', 'bert', 'sentence_transformers')
            embedding_dim: Embedding dimension
            max_vocab_size: Maximum vocabulary size
            model_path: Path to pre-trained model
            use_sentiment: Include sentiment features
            language: Primary language ('zh' for Chinese, 'en' for English)
        """
        self.method = method
        self.embedding_dim = embedding_dim
        self.max_vocab_size = max_vocab_size
        self.model_path = model_path
        self.use_sentiment = use_sentiment
        self.language = language
        
        self._model: Any = None
        self._vocab: dict[str, int] | None = None
        self._idf: np.ndarray | None = None
        self._is_fitted = False
        
        # Chinese tokenization
        self._tokenizer = None
        if language == "zh":
            try:
                import jieba
                self._tokenizer = jieba
            except ImportError:
                log.warning("jieba not installed - Chinese tokenization unavailable")
        
        # Sentiment lexicon
        self._sentiment_lexicon: dict[str, float] = {}
        if use_sentiment:
            self._load_sentiment_lexicon()
    
    def _load_sentiment_lexicon(self) -> None:
        """Load sentiment lexicon for Chinese/English."""
        # Simple sentiment lexicon - in production, use more comprehensive resources
        if self.language == "zh":
            self._sentiment_lexicon = {
                # Positive
                "上涨": 0.8, "增长": 0.7, "利好": 0.9, "突破": 0.8,
                "向好": 0.7, "复苏": 0.6, "强劲": 0.7, "乐观": 0.8,
                "支持": 0.5, "促进": 0.6, "改善": 0.6, "提升": 0.7,
                # Negative
                "下跌": -0.8, "下滑": -0.7, "利空": -0.9, "暴跌": -1.0,
                "恶化": -0.8, "风险": -0.6, "担忧": -0.7, "悲观": -0.8,
                "衰退": -0.9, "危机": -1.0, "亏损": -0.8, "下滑": -0.7,
                # Neutral
                "市场": 0.0, "股票": 0.0, "交易": 0.0, "分析": 0.0,
            }
        else:
            self._sentiment_lexicon = {
                # Positive
                "rise": 0.8, "growth": 0.7, "positive": 0.9, "breakthrough": 0.8,
                "bullish": 0.8, "recovery": 0.6, "strong": 0.7, "optimistic": 0.8,
                # Negative
                "fall": -0.8, "decline": -0.7, "negative": -0.9, "crash": -1.0,
                "bearish": -0.8, "recession": -0.9, "crisis": -1.0, "loss": -0.8,
            }
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text."""
        if self.language == "zh" and self._tokenizer is not None:
            return list(self._tokenizer.cut(text))
        else:
            # Simple whitespace tokenization for English
            return text.lower().split()
    
    def fit(self, texts: list[str]) -> "EmbeddingModel":
        """Fit the embedding model on texts.
        
        Args:
            texts: List of texts to fit on
            
        Returns:
            Self for chaining
        """
        if self.method == "tfidf":
            self._fit_tfidf(texts)
        elif self.method == "word2vec":
            self._fit_word2vec(texts)
        elif self.method in ("bert", "sentence_transformers"):
            self._load_pretrained()
        
        self._is_fitted = True
        log.info(f"EmbeddingModel fitted with {len(texts)} texts using {self.method}")
        return self
    
    def _fit_tfidf(self, texts: list[str]) -> None:
        """Fit TF-IDF model."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Preprocess texts
            processed_texts = []
            for text in texts:
                tokens = self._tokenize(text)
                processed_texts.append(" ".join(tokens))
            
            self._model = TfidfVectorizer(
                max_features=self.max_vocab_size,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
            )
            self._model.fit(processed_texts)
            self._vocab = {word: idx for idx, word in enumerate(self._model.get_feature_names_out())}
            self.embedding_dim = len(self._vocab)
            self._is_fitted = True
            
            log.info(f"TF-IDF model fitted with vocab size {len(self._vocab)}")
            
        except ImportError:
            log.warning("sklearn not available - using fallback embeddings")
            self.embedding_dim = self.embedding_dim
            self._is_fitted = True
    
    def _fit_word2vec(self, texts: list[str]) -> None:
        """Fit Word2Vec model."""
        try:
            from gensim.models import Word2Vec
            
            # Tokenize all texts
            tokenized_texts = [self._tokenize(text) for text in texts]
            
            self._model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=self.embedding_dim,
                window=5,
                min_count=2,
                workers=4,
            )
            self._is_fitted = True
            
            log.info(f"Word2Vec model fitted with {len(self._model.wv)} words")
            
        except ImportError:
            log.warning("gensim not available - using fallback embeddings")
            self.embedding_dim = self.embedding_dim
            self._is_fitted = True
    
    def _load_pretrained(self) -> None:
        """Load pre-trained transformer model."""
        if self.method == "bert":
            try:
                from transformers import BertModel, BertTokenizer
                
                model_name = "bert-base-chinese" if self.language == "zh" else "bert-base-uncased"
                self._tokenizer = BertTokenizer.from_pretrained(model_name)
                self._model = BertModel.from_pretrained(model_name)
                self.embedding_dim = self._model.config.hidden_size
                
                log.info(f"Loaded BERT model: {model_name}")
                
            except ImportError:
                log.warning("transformers not available - falling back to TF-IDF")
                self.method = "tfidf"
                self._load_pretrained = lambda: None
        
        elif self.method == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer
                
                model_name = (
                    "paraphrase-multilingual-MiniLM-L12-v2"
                    if self.language == "zh"
                    else "all-MiniLM-L6-v2"
                )
                self._model = SentenceTransformer(model_name)
                self.embedding_dim = self._model.get_sentence_embedding_dimension()
                
                log.info(f"Loaded SentenceTransformer: {model_name}")
                
            except ImportError:
                log.warning("sentence-transformers not available - falling back to TF-IDF")
                self.method = "tfidf"
    
    def transform(self, text: str) -> TextEmbeddingResult:
        """Transform text to embedding.
        
        Args:
            text: Input text
            
        Returns:
            TextEmbeddingResult object
        """
        if not self._is_fitted and self.method == "tfidf":
            raise ValueError("Model not fitted. Call fit() first.")
        
        result = TextEmbeddingResult(
            text_length=len(text),
            language=self.language,
            method=self.method,
        )
        
        if self.method == "tfidf":
            result.embedding = self._transform_tfidf(text)
        elif self.method == "word2vec":
            result.embedding = self._transform_word2vec(text)
        elif self.method == "bert":
            result.embedding = self._transform_bert(text)
        elif self.method == "sentence_transformers":
            result.embedding = self._transform_sentence_transformers(text)
        else:
            # Fallback: simple bag-of-words
            result.embedding = self._transform_bow(text)
        
        result.embedding_dim = len(result.embedding)
        
        # Calculate sentiment score
        if self.use_sentiment:
            result.sentiment_score = self._calculate_sentiment(text)
        
        # Extract keywords
        result.keywords = self._extract_keywords(text)
        
        return result
    
    def _transform_tfidf(self, text: str) -> np.ndarray:
        """Transform text using TF-IDF."""
        tokens = self._tokenize(text)
        processed_text = " ".join(tokens)
        
        try:
            embedding = self._model.transform([processed_text]).toarray()[0]
            return embedding.astype(np.float32)
        except (AttributeError, ValueError):
            # Model not fitted or error
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def _transform_word2vec(self, text: str) -> np.ndarray:
        """Transform text using Word2Vec."""
        tokens = self._tokenize(text)
        
        try:
            embeddings = []
            for token in tokens:
                if token in self._model.wv:
                    embeddings.append(self._model.wv[token])
            
            if embeddings:
                return np.mean(embeddings, axis=0).astype(np.float32)
            else:
                return np.zeros(self.embedding_dim, dtype=np.float32)
        except (AttributeError, ValueError):
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def _transform_bert(self, text: str) -> np.ndarray:
        """Transform text using BERT."""
        import torch
        
        try:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
            
            return embedding.astype(np.float32)
            
        except (AttributeError, ValueError, ImportError):
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def _transform_sentence_transformers(self, text: str) -> np.ndarray:
        """Transform text using SentenceTransformers."""
        try:
            embedding = self._model.encode([text])[0]
            return embedding.astype(np.float32)
        except (AttributeError, ValueError, ImportError):
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def _transform_bow(self, text: str) -> np.ndarray:
        """Fallback bag-of-words embedding."""
        tokens = self._tokenize(text)
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        
        if self._vocab:
            for i, token in enumerate(tokens[:self.embedding_dim]):
                if token in self._vocab:
                    embedding[self._vocab[token]] = 1.0
        else:
            # Hash-based
            for i, token in enumerate(tokens[:self.embedding_dim]):
                idx = hash(token) % self.embedding_dim
                embedding[idx] = 1.0
        
        return embedding
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score using lexicon."""
        tokens = self._tokenize(text)
        
        scores = []
        for token in tokens:
            if token in self._sentiment_lexicon:
                scores.append(self._sentiment_lexicon[token])
        
        if scores:
            return float(np.mean(scores))
        return 0.0
    
    def _extract_keywords(self, text: str, top_k: int = 10) -> list[str]:
        """Extract keywords from text."""
        tokens = self._tokenize(text)
        
        # Filter stopwords and short tokens
        stopwords = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "就", "这", "也", "但", "而"}
        keywords = [
            token for token in tokens
            if token not in stopwords and len(token) > 1
        ]
        
        # Return top frequent keywords
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [word for word, _ in keyword_counts.most_common(top_k)]
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        import pickle
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "method": self.method,
            "embedding_dim": self.embedding_dim,
            "vocab": self._vocab,
            "model": self._model,
            "sentiment_lexicon": self._sentiment_lexicon,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        log.info(f"Embedding model saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        import pickle
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.method = state.get("method", "tfidf")
        self.embedding_dim = state.get("embedding_dim", 256)
        self._vocab = state.get("vocab")
        self._model = state.get("model")
        self._sentiment_lexicon = state.get("sentiment_lexicon", {})
        self._is_fitted = True
        
        log.info(f"Embedding model loaded from {path}")


def create_news_features(
    texts: list[str],
    method: str = "tfidf",
    embedding_dim: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create news features from texts.
    
    Args:
        texts: List of news texts
        method: Embedding method
        embedding_dim: Embedding dimension
        
    Returns:
        Tuple of (features array, metadata dict)
    """
    model = EmbeddingModel(
        method=method,
        embedding_dim=embedding_dim,
    )
    
    # Fit and transform
    model.fit(texts)
    
    embeddings = []
    sentiment_scores = []
    keyword_lists = []
    
    for text in texts:
        result = model.transform(text)
        embeddings.append(result.embedding)
        sentiment_scores.append(result.sentiment_score)
        keyword_lists.append(result.keywords)
    
    features = np.array(embeddings, dtype=np.float32)
    
    metadata = {
        "method": method,
        "embedding_dim": embedding_dim,
        "n_texts": len(texts),
        "sentiment_scores": sentiment_scores,
        "keywords": keyword_lists,
    }
    
    return features, metadata
