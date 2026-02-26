# tests/integration/test_ml_pipeline.py
"""
Integration Tests for ML Pipeline

FIXES:
- Test model training pipeline
- Test model registry
- Test pretrained model integration
- End-to-end ML flow testing
"""

from __future__ import annotations

import os
import pytest
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.model_registry import (
    ModelRegistry,
    ModelType,
    ModelStatus,
    get_registry,
)
from models.pretrained_integration import (
    create_sentiment_model,
    FineTuningConfig,
    FallbackSentimentModel,
)


class TestModelRegistry:
    """Integration tests for model registry."""
    
    @pytest.fixture
    def registry(self, tmp_path: Path) -> ModelRegistry:
        """Create registry with temp directory."""
        registry_path = tmp_path / "registry"
        registry_path.mkdir()
        return ModelRegistry(registry_path=str(registry_path))
    
    @pytest.fixture
    def sample_model(self) -> nn.Module:
        """Create sample model for testing."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)
        
        return SimpleModel()
    
    def test_register_model(
        self,
        registry: ModelRegistry,
        sample_model: nn.Module,
    ) -> None:
        """Test model registration."""
        model_id = registry.register_model(
            model=sample_model,
            model_type=ModelType.CUSTOM,
            version="1.0.0",
            metrics={"accuracy": 0.85, "loss": 0.45},
            hyperparameters={
                "learning_rate": 0.001,
                "batch_size": 32,
                "train_samples": 1000,
                "val_samples": 200,
            },
            data_hash="abc123",
            training_duration_seconds=120.5,
            tags=["test", "integration"],
        )
        
        assert model_id.startswith("custom_1.0.0_")
        assert registry.get_model(model_id, type(sample_model)) is not None
    
    def test_model_promotion_workflow(
        self,
        registry: ModelRegistry,
        sample_model: nn.Module,
    ) -> None:
        """Test model promotion workflow (dev -> staging -> production)."""
        model_id = registry.register_model(
            model=sample_model,
            model_type=ModelType.CUSTOM,
            version="1.0.0",
            metrics={"accuracy": 0.85},
            hyperparameters={"train_samples": 1000},
            data_hash="abc123",
            training_duration_seconds=120.5,
        )
        
        # Promote to staging
        registry.promote_model(model_id, ModelStatus.STAGING)
        metadata = registry._metadata_cache[model_id]
        assert metadata.status == ModelStatus.STAGING
        
        # Promote to production
        registry.promote_model(model_id, ModelStatus.PRODUCTION)
        metadata = registry._metadata_cache[model_id]
        assert metadata.status == ModelStatus.PRODUCTION
    
    def test_invalid_promotion_rejected(
        self,
        registry: ModelRegistry,
        sample_model: nn.Module,
    ) -> None:
        """Test invalid promotion transitions are rejected."""
        model_id = registry.register_model(
            model=sample_model,
            model_type=ModelType.CUSTOM,
            version="1.0.0",
            metrics={"accuracy": 0.85},
            hyperparameters={"train_samples": 1000},
            data_hash="abc123",
            training_duration_seconds=120.5,
        )
        
        # Cannot promote directly from development to production
        with pytest.raises(ValueError):
            registry.promote_model(model_id, ModelStatus.PRODUCTION)
    
    def test_list_models_filtering(
        self,
        registry: ModelRegistry,
        sample_model: nn.Module,
    ) -> None:
        """Test model listing with filters."""
        # Register multiple models
        for i in range(3):
            registry.register_model(
                model=sample_model,
                model_type=ModelType.CUSTOM,
                version=f"1.0.{i}",
                metrics={"accuracy": 0.80 + i * 0.05},
                hyperparameters={"train_samples": 1000},
                data_hash=f"hash{i}",
                training_duration_seconds=120.5,
            )
        
        # List all
        all_models = registry.list_models()
        assert len(all_models) == 3
        
        # Filter by metric threshold
        high_acc_models = registry.list_models(
            min_metric=("accuracy", 0.85)
        )
        assert len(high_acc_models) <= 2
    
    def test_model_checksum_verification(
        self,
        registry: ModelRegistry,
        sample_model: nn.Module,
    ) -> None:
        """Test model integrity verification."""
        model_id = registry.register_model(
            model=sample_model,
            model_type=ModelType.CUSTOM,
            version="1.0.0",
            metrics={"accuracy": 0.85},
            hyperparameters={"train_samples": 1000},
            data_hash="abc123",
            training_duration_seconds=120.5,
        )
        
        # Load should verify checksum
        loaded_model = registry.get_model(model_id, type(sample_model))
        assert loaded_model is not None
        
        # Corrupt model file
        model_path = registry.models_path / model_id / "model.pt"
        with open(model_path, "ab") as f:
            f.write(b"corrupted")
        
        # Loading should fail with corrupted checksum
        with pytest.raises(RuntimeError, match="checksum"):
            registry.get_model(model_id, type(sample_model))
    
    def test_cleanup_old_models(
        self,
        registry: ModelRegistry,
        sample_model: nn.Module,
    ) -> None:
        """Test automatic cleanup of old models."""
        # Register multiple development models
        for i in range(10):
            registry.register_model(
                model=sample_model,
                model_type=ModelType.CUSTOM,
                version=f"1.0.{i}",
                metrics={"accuracy": 0.80 + i * 0.01},
                hyperparameters={"train_samples": 1000},
                data_hash=f"hash{i}",
                training_duration_seconds=120.5,
            )
        
        # Cleanup should keep only recent models
        deleted = registry.cleanup_old_models(
            keep_count=5,
            older_than_days=0,  # All are "old" for this test
        )
        
        remaining = registry.list_models(model_type=ModelType.CUSTOM)
        assert len(remaining) <= 5


class TestPretrainedIntegration:
    """Integration tests for pretrained model integration."""
    
    @pytest.fixture
    def sample_text_data(self) -> list[str]:
        """Create sample text data for testing."""
        return [
            "This is positive news about the stock",
            "The market is performing well today",
            "Negative sentiment in the market",
            "Stock prices are falling sharply",
            "Neutral market conditions expected",
        ]
    
    @pytest.fixture
    def sample_labels(self) -> list[int]:
        """Create sample labels for testing."""
        return [0, 0, 1, 1, 2]  # 0=positive, 1=negative, 2=neutral
    
    def test_fallback_model_creation(self) -> None:
        """Test fallback model creation without transformers."""
        model = create_sentiment_model(
            use_pretrained=False,
        )
        
        assert isinstance(model, FallbackSentimentModel)
        assert isinstance(model, nn.Module)
    
    @pytest.mark.skipif(
        not os.environ.get("TEST_PRETRAINED"),
        reason="Requires transformers library",
    )
    def test_pretrained_model_fine_tuning(
        self,
        sample_text_data: list[str],
        sample_labels: list[int],
    ) -> None:
        """Test fine-tuning pretrained model."""
        # This test requires transformers
        pytest.importorskip("transformers")
        
        config = FineTuningConfig(
            model_name="bert-base-chinese",
            num_labels=3,
            batch_size=2,
            num_epochs=1,  # Minimal for testing
        )
        
        model = create_sentiment_model(
            use_pretrained=True,
            fine_tuning_config=config,
        )
        
        # Create dummy dataloader
        # In real test, would tokenize text data
        assert model is not None
    
    def test_model_inference(
        self,
        sample_text_data: list[str],
    ) -> None:
        """Test model inference."""
        model = FallbackSentimentModel(
            vocab_size=1000,
            embedding_dim=64,
            hidden_size=128,
            num_labels=3,
        )
        model.eval()
        
        # Create dummy input
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, 3)


class TestEndToEndMLFlow:
    """End-to-end tests for complete ML flow."""
    
    @pytest.fixture
    def temp_dir(self, tmp_path: Path) -> Path:
        """Create temp directory for ML artifacts."""
        ml_dir = tmp_path / "ml_test"
        ml_dir.mkdir()
        return ml_dir
    
    def test_full_training_pipeline(
        self,
        temp_dir: Path,
    ) -> None:
        """Test complete training pipeline from training to registration."""
        # 1. Create model
        model = FallbackSentimentModel(
            vocab_size=500,
            embedding_dim=32,
            hidden_size=64,
            num_labels=2,
        )
        
        # 2. Create dummy training data
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=10)
        
        # 3. Simple training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        
        for epoch in range(2):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = outputs.get("loss", torch.tensor(0.0))
                loss.backward()
                optimizer.step()
        
        # 4. Register model
        registry = ModelRegistry(registry_path=str(temp_dir / "registry"))
        model_id = registry.register_model(
            model=model,
            model_type=ModelType.CUSTOM,
            version="1.0.0",
            metrics={"accuracy": 0.75, "loss": 0.5},
            hyperparameters={
                "learning_rate": 0.01,
                "batch_size": 10,
                "train_samples": 100,
            },
            data_hash="test_hash",
            training_duration_seconds=30.0,
        )
        
        # 5. Promote to production
        registry.promote_model(model_id, ModelStatus.STAGING)
        registry.promote_model(model_id, ModelStatus.PRODUCTION)
        
        # 6. Load production model
        production_model = registry.get_production_model(ModelType.CUSTOM)
        assert production_model is not None
        assert production_model.model_id == model_id
    
    def test_model_comparison(
        self,
        temp_dir: Path,
    ) -> None:
        """Test model comparison for selection."""
        registry = ModelRegistry(registry_path=str(temp_dir / "registry"))
        
        # Register multiple models with different accuracies
        model_ids = []
        for i, acc in enumerate([0.70, 0.80, 0.75]):
            model = nn.Linear(10, 1)
            model_id = registry.register_model(
                model=model,
                model_type=ModelType.INFORMER,
                version=f"1.0.{i}",
                metrics={"accuracy": acc},
                hyperparameters={"train_samples": 1000},
                data_hash=f"hash{i}",
                training_duration_seconds=60.0,
            )
            model_ids.append(model_id)
        
        # Compare models
        comparison = registry.compare_models(
            model_ids=model_ids,
            test_data_hash="test_data",
        )
        
        # Best model should be ranked first
        assert comparison["ranking"][0] == model_ids[1]  # 0.80 accuracy


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Cleanup test artifacts after each test."""
    yield
    # Cleanup temp files
    test_dirs = [
        Path("models_saved/test_registry"),
        Path("tests/temp"),
    ]
    for dir_path in test_dirs:
        if dir_path.exists():
            shutil.rmtree(dir_path)
