# models/news_trainer.py
"""News and Policy-Based Model Training.

This module trains models to understand and predict market movements based on:
- News sentiment analysis
- Policy impact assessment
- Historical price patterns correlated with news
- Multi-modal learning (text + numerical data)

Architecture:
1. News Encoder: Transformer-based text encoder for news/policy content
2. Sentiment Fusion: Combines sentiment scores with encoded text
3. Price Encoder: LSTM/GRU for historical price patterns
4. Fusion Layer: Combines news and price features
5. Prediction Head: Outputs trading signals and confidence
"""

import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from config.settings import CONFIG
from data.news_collector import NewsArticle, get_collector
from data.sentiment_analyzer import get_analyzer
from utils.logger import get_logger

log = get_logger(__name__)

# FIX: Define ConvergenceWarning for compatibility
try:
    from sklearn.exceptions import ConvergenceWarning
except ImportError:
    class ConvergenceWarning(Exception):
        """Dummy ConvergenceWarning for sklearn compatibility."""
        pass


@dataclass
class NewsTrainingSample:
    """Single training sample combining news and price data."""
    news_embeddings: np.ndarray  # [seq_len, embed_dim]
    sentiment_features: np.ndarray  # [sentiment_dim]
    price_features: np.ndarray  # [lookback, price_features]
    label: int  # 0: sell, 1: hold, 2: buy
    label_return: float  # Actual return over prediction horizon
    symbol: str
    timestamp: datetime


class NewsDataset(Dataset):
    """PyTorch Dataset for news-based training."""

    def __init__(
        self,
        samples: list[NewsTrainingSample],
        max_news_len: int = 512,
    ) -> None:
        self.samples = samples
        self.max_news_len = max_news_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Truncate or pad news embeddings
        news_emb = sample.news_embeddings[:self.max_news_len]
        if len(news_emb) < self.max_news_len:
            pad_len = self.max_news_len - len(news_emb)
            news_emb = np.vstack([
                news_emb,
                np.zeros((pad_len, news_emb.shape[1] if news_emb.ndim > 1 else 1))
            ])

        return {
            "news_embeddings": torch.FloatTensor(news_emb),
            "sentiment_features": torch.FloatTensor(sample.sentiment_features),
            "price_features": torch.FloatTensor(sample.price_features),
            "label": torch.LongTensor([sample.label])[0],
            "label_return": torch.FloatTensor([sample.label_return])[0],
        }


class NewsEncoder(nn.Module):
    """Transformer-based encoder for news text."""

    def __init__(
        self,
        vocab_size: int = 30000,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.max_len = max_len

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode news text to embeddings.

        Args:
            token_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            [batch, embed_dim] - pooled representation
        """
        batch_size, seq_len = token_ids.shape

        # Add position embeddings
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)

        # Token embeddings
        token_emb = self.token_embedding(token_ids)
        emb = token_emb + pos_emb

        # Transformer encoding
        if attention_mask is None:
            attention_mask = (token_ids != 0).float()

        # Transformer expects src_key_padding_mask with True for padding
        src_key_padding_mask = (1 - attention_mask).bool()

        encoded = self.transformer(
            emb,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Pooling (mean over non-padded positions)
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoded)
        sum_embeddings = (encoded * mask_expanded).sum(1)
        sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask

        # Output projection
        output = self.output_proj(pooled)
        output = self.layer_norm(output)

        return output


class NewsPriceFusionModel(nn.Module):
    """Multi-modal model fusing news sentiment and price data."""

    def __init__(
        self,
        news_embed_dim: int = 256,
        sentiment_dim: int = 10,
        price_dim: int = 20,
        price_hidden: int = 64,
        num_classes: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # News encoder (pretrained or trained end-to-end)
        self.news_encoder = NewsEncoder(embed_dim=news_embed_dim)

        # Sentiment MLP
        self.sentiment_mlp = nn.Sequential(
            nn.Linear(sentiment_dim, sentiment_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(sentiment_dim * 2, news_embed_dim),
        )

        # Price encoder (LSTM)
        self.price_lstm = nn.LSTM(
            input_size=price_dim,
            hidden_size=price_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        # Fusion layer
        fusion_dim = news_embed_dim + news_embed_dim + price_hidden * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
        )

        # Prediction heads
        self.signal_head = nn.Linear(fusion_dim // 4, num_classes)
        self.return_head = nn.Linear(fusion_dim // 4, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        news_embeddings: torch.Tensor,
        sentiment_features: torch.Tensor,
        price_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            news_embeddings: [batch, seq_len, embed_dim] or [batch, seq_len] (if pre-computed)
            sentiment_features: [batch, sentiment_dim]
            price_features: [batch, lookback, price_dim]

        Returns:
            Dictionary with 'signal_logits' and 'return_pred'
        """
        # Handle pre-computed news embeddings vs token IDs
        if news_embeddings.dim() == 3:
            # Pre-computed embeddings - pool directly
            news_pooled = news_embeddings.mean(dim=1)
        else:
            # Token IDs - encode
            news_pooled = self.news_encoder(news_embeddings)

        # Process sentiment
        sentiment_encoded = self.sentiment_mlp(sentiment_features)

        # Process prices
        price_encoded, (h_n, c_n) = self.price_lstm(price_features)
        price_pooled = price_encoded[:, -1, :]  # Last timestep

        # Fuse
        combined = torch.cat([news_pooled, sentiment_encoded, price_pooled], dim=1)
        fused = self.fusion(combined)

        # Predictions
        signal_logits = self.signal_head(fused)
        return_pred = self.return_head(fused)

        return {
            "signal_logits": signal_logits,
            "return_pred": return_pred.squeeze(-1),
        }


class NewsTrainer:
    """Trainer for news-based prediction models."""

    def __init__(
        self,
        model_dir: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.model_dir = model_dir or CONFIG.model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # Model
        self.model: NewsPriceFusionModel | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.criterion = nn.CrossEntropyLoss()

        # Data
        self.collector = get_collector()
        self.analyzer = get_analyzer()

        # Training state
        self.training_history: list[dict] = []

    def prepare_training_data(
        self,
        symbols: list[str],
        days_back: int = 90,
        min_articles_per_day: int = 5,
    ) -> list[NewsTrainingSample]:
        """Prepare training data from news and price history."""
        log.info(f"Preparing training data for {len(symbols)} symbols...")

        samples = []
        cutoff_date = datetime.now() - timedelta(days=days_back)

        for symbol in symbols:
            try:
                # Collect news
                articles = self.collector.collect_news(
                    keywords=[symbol],
                    hours_back=days_back * 24,
                    limit=1000,
                )

                if len(articles) < min_articles_per_day * days_back // 7:
                    log.warning(f"Insufficient news for {symbol}, skipping")
                    continue

                # Analyze sentiment
                for article in articles:
                    score = self.analyzer.analyze_article(article)
                    article.sentiment_score = score.overall

                # Get price data
                try:
                    from data.fetcher import get_fetcher
                    fetcher = get_fetcher()

                    prices = fetcher.get_history(
                        symbol,
                        interval="1d",
                        bars=days_back + 20,  # Extra for labels
                    )

                    if prices is None or len(prices) < days_back:
                        log.warning(f"Insufficient price data for {symbol}")
                        continue

                except Exception as e:
                    log.error(f"Failed to get price data for {symbol}: {e}")
                    continue

                # Create samples
                symbol_samples = self._create_samples(
                    symbol=symbol,
                    articles=articles,
                    prices=prices,
                )
                samples.extend(symbol_samples)

                log.info(f"Created {len(symbol_samples)} samples for {symbol}")

            except Exception as e:
                log.error(f"Error preparing data for {symbol}: {e}")
                continue

        log.info(f"Total training samples: {len(samples)}")
        return samples

    def _create_samples(
        self,
        symbol: str,
        articles: list[NewsArticle],
        prices: Any,  # DataFrame
    ) -> list[NewsTrainingSample]:
        """Create training samples from news and prices."""
        samples = []
        lookback = 5  # Days of price history
        horizon = 3  # Prediction horizon (days)

        # Group articles by date
        articles_by_date: dict[str, list[NewsArticle]] = {}
        for article in articles:
            date_str = article.published_at.strftime("%Y-%m-%d")
            if date_str not in articles_by_date:
                articles_by_date[date_str] = []
            articles_by_date[date_str].append(article)

        # Iterate through price data
        for i in range(lookback, len(prices) - horizon):
            try:
                current_date = prices.index[i]
                if hasattr(current_date, 'strftime'):
                    date_str = current_date.strftime("%Y-%m-%d")
                else:
                    # Handle non-datetime index
                    continue

                # Get news for this day
                day_articles = articles_by_date.get(date_str, [])
                if not day_articles:
                    continue

                # Aggregate sentiment
                sentiment_scores = [a.sentiment_score for a in day_articles]
                avg_sentiment = float(np.mean(sentiment_scores)) if sentiment_scores else 0.0
                sentiment_std = float(np.std(sentiment_scores)) if len(sentiment_scores) > 1 else 0.0

                # Sentiment features
                sentiment_features = np.array([
                    avg_sentiment,
                    sentiment_std,
                    float(len(day_articles)),
                    float(max(sentiment_scores)) if sentiment_scores else 0.0,
                    float(min(sentiment_scores)) if sentiment_scores else 0.0,
                    float(sum(1 for s in sentiment_scores if s > 0.3)),  # Positive count
                    float(sum(1 for s in sentiment_scores if s < -0.3)),  # Negative count
                    float(sum(1 for a in day_articles if a.category == "policy")),  # Policy count
                    float(np.mean([a.relevance_score for a in day_articles])) if day_articles else 0.0,
                    0.0,  # Placeholder for future features
                ])

                # Price features
                price_data = prices.iloc[i-lookback:i]
                price_features = self._extract_price_features(price_data)

                # Label: price movement over horizon
                # FIX: Add bounds checking for iloc access
                if i >= len(prices) or i + horizon >= len(prices):
                    log.debug(f"Index out of bounds at i={i}, horizon={horizon}, len(prices)={len(prices)}")
                    continue

                try:
                    current_price = float(prices["close"].iloc[i])
                    future_price = float(prices["close"].iloc[i + horizon])
                    future_return = (future_price - current_price) / current_price if current_price != 0 else 0.0
                except (IndexError, KeyError, TypeError, ValueError) as e:
                    log.debug(f"Failed to get price data at index {i}: {e}")
                    continue

                # Convert to class
                if future_return > 0.03:
                    label = 2  # Buy
                elif future_return < -0.03:
                    label = 0  # Sell
                else:
                    label = 1  # Hold

                # News embeddings (simplified - use sentiment as proxy)
                # In production, would use actual text embeddings
                news_embeddings = np.array([
                    [avg_sentiment] * 256  # Repeat for embedding dim
                ])

                samples.append(NewsTrainingSample(
                    news_embeddings=news_embeddings,
                    sentiment_features=sentiment_features,
                    price_features=price_features,
                    label=label,
                    label_return=future_return,
                    symbol=symbol,
                    timestamp=current_date if isinstance(current_date, datetime) else datetime.now(),
                ))

            except Exception as e:
                log.debug(f"Error creating sample at index {i}: {e}")
                continue

        return samples

    def _extract_price_features(self, price_data: Any) -> np.ndarray:
        """Extract features from price DataFrame."""
        features = []

        for _, row in price_data.iterrows():
            try:
                close_price = float(row.get("close", 1.0))
                if close_price <= 0:
                    close_price = 1.0  # Prevent division by zero
                
                # OHLCV + returns
                row_features = [
                    float(row.get("open", close_price)) / close_price - 1.0,  # Intraday return
                    float(row.get("high", close_price)) / max(float(row.get("low", close_price)), 1e-8) - 1.0,  # Daily range
                    float(row.get("volume", 0)) / 1e6,  # Volume (millions)
                    close_price,  # Close price (will be normalized)
                ]
                features.append(row_features)
            except (TypeError, ValueError, ZeroDivisionError):
                # Keep fixed lookback width for batch collation.
                features.append([0.0, 0.0, 0.0, 1.0])

        if not features:
            # Return default features if no valid data
            return np.zeros((len(price_data), 4), dtype=np.float32)

        result = np.array(features, dtype=np.float32)

        # Normalize
        if len(result) > 0 and result[:, 3].max() > 0:
            result[:, 3] = result[:, 3] / result[:, 3].max()  # Normalize price

        return result

    def train(
        self,
        symbols: list[str],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_split: float = 0.2,
    ) -> dict[str, Any]:
        """Train the news-based prediction model."""
        log.info(f"Starting news-based training for {len(symbols)} symbols...")
        start_time = time.time()

        # Prepare data
        samples = self.prepare_training_data(symbols, days_back=90)

        if not samples:
            log.error("No training samples generated")
            return {"error": "No training samples"}

        # Split train/validation
        np.random.shuffle(samples)
        split_idx = int(len(samples) * (1 - validation_split))
        if len(samples) < 2:
            log.error("Need at least 2 samples for train/validation split")
            return {"error": "Insufficient samples for split"}
        split_idx = max(1, min(len(samples) - 1, split_idx))
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        log.info(f"Train: {len(train_samples)}, Validation: {len(val_samples)}")

        # Create datasets
        train_dataset = NewsDataset(train_samples)
        val_dataset = NewsDataset(val_samples)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            sampler=None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(val_dataset),
        )

        # Initialize model
        self.model = NewsPriceFusionModel(
            news_embed_dim=256,
            sentiment_dim=10,
            price_dim=4,
            price_hidden=64,
            num_classes=3,
            dropout=0.2,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-5,
        )

        # Training loop
        best_val_acc = 0.0
        training_history = []

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._validate(val_loader)

            scheduler.step()

            history_entry = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            training_history.append(history_entry)

            log.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model("news_model_best.pt")

        # Save final model
        self.training_history = list(training_history)
        self.save_model("news_model.pt")

        elapsed = time.time() - start_time
        log.info(f"Training completed in {elapsed:.1f}s, best val acc: {best_val_acc:.2%}")

        return {
            "status": "success",
            "epochs": epochs,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "best_val_acc": best_val_acc,
            "training_time_seconds": elapsed,
            "history": training_history,
        }

    def _train_epoch(
        self,
        loader: DataLoader,
    ) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            news_emb = batch["news_embeddings"].to(self.device)
            sentiment = batch["sentiment_features"].to(self.device)
            prices = batch["price_features"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(news_emb, sentiment, prices)
            loss = self.criterion(outputs["signal_logits"], labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            predictions = outputs["signal_logits"].argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        if len(loader) == 0 or total <= 0:
            return 0.0, 0.0
        return total_loss / len(loader), correct / total

    def _validate(self, loader: DataLoader) -> tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                news_emb = batch["news_embeddings"].to(self.device)
                sentiment = batch["sentiment_features"].to(self.device)
                prices = batch["price_features"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(news_emb, sentiment, prices)
                loss = self.criterion(outputs["signal_logits"], labels)

                total_loss += loss.item()
                predictions = outputs["signal_logits"].argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        if len(loader) == 0 or total <= 0:
            return 0.0, 0.0
        return total_loss / len(loader), correct / total

    def save_model(self, filename: str = "news_model.pt") -> Path:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        path = self.model_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "training_history": self.training_history,
        }, path)

        log.info(f"Model saved to {path}")
        return path

    def load_model(self, filename: str = "news_model.pt") -> None:
        """Load model from disk."""
        path = self.model_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        # FIX: Use weights_only=True for security (PyTorch 2.6+)
        # This prevents arbitrary code execution from malicious model files
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions that don't support weights_only
            log.warning("PyTorch version doesn't support weights_only, using legacy load")
            checkpoint = torch.load(path, map_location=self.device)

        if self.model is None:
            self.model = NewsPriceFusionModel()

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        if checkpoint.get("optimizer_state_dict"):
            if self.optimizer is None:
                self.optimizer = torch.optim.AdamW(self.model.parameters())
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.training_history = checkpoint.get("training_history", [])

        log.info(f"Model loaded from {path}")

    def predict(
        self,
        symbol: str,
        articles: list[NewsArticle],
        price_data: Any,
    ) -> dict[str, Any]:
        """Make prediction using trained model."""
        if self.model is None:
            raise ValueError("No model loaded")

        # Prepare input
        samples = self._create_samples(symbol, articles, price_data)

        if not samples:
            return {"error": "Could not prepare input"}

        sample = samples[-1]  # Most recent

        # Create tensors
        news_emb = torch.FloatTensor(sample.news_embeddings).unsqueeze(0).to(self.device)
        sentiment = torch.FloatTensor(sample.sentiment_features).unsqueeze(0).to(self.device)
        prices = torch.FloatTensor(sample.price_features).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(news_emb, sentiment, prices)
            probs = torch.softmax(outputs["signal_logits"], dim=1)[0]
            return_pred = outputs["return_pred"][0].item()

        # Convert to signal
        signal_idx = probs.argmax().item()
        signals = ["SELL", "HOLD", "BUY"]
        signal = signals[signal_idx]
        confidence = probs[signal_idx].item()

        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "expected_return": return_pred,
            "probabilities": {
                "sell": probs[0].item(),
                "hold": probs[1].item(),
                "buy": probs[2].item(),
            },
        }


# Singleton
_trainer: NewsTrainer | None = None


def get_trainer() -> NewsTrainer:
    """Get or create news trainer instance."""
    global _trainer
    if _trainer is None:
        _trainer = NewsTrainer()
    return _trainer


def reset_trainer() -> None:
    """Reset trainer instance."""
    global _trainer
    _trainer = None
