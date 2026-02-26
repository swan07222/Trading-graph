# models/pretrained_integration.py
"""
Pretrained Model Integration with Fine-tuning

FIXES:
- Re-add pretrained transformer support (optional)
- Fine-tuning on domain-specific data
- Transfer learning for limited data scenarios
- Model distillation for deployment
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils.logger import get_logger

log = get_logger(__name__)

# Optional imports - gracefully degrade if unavailable
try:
    from transformers import (
        BertTokenizer,
        BertForSequenceClassification,
        BertConfig,
        AutoTokenizer,
        AutoModel,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    log.info("Transformers not available - using fallback models")


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning pretrained models."""
    model_name: str = "bert-base-chinese"
    num_labels: int = 3  # positive, neutral, negative
    learning_rate: float = 2e-5
    batch_size: int = 16
    max_length: int = 128
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    freeze_layers: int = 6  # Number of layers to freeze
    use_fp16: bool = False
    gradient_accumulation_steps: int = 1


class PretrainedSentimentModel(nn.Module):
    """
    Pretrained transformer for sentiment analysis with fine-tuning.
    
    FIXES:
    1. Optional pretrained model support (BERT, FinBERT)
    2. Fine-tuning on China A-share domain data
    3. Knowledge distillation for smaller deployment models
    """
    
    def __init__(self, config: FineTuningConfig):
        super().__init__()
        self.config = config
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library required. Install with: "
                "pip install transformers sentence-transformers"
            )
        
        # Load pretrained model
        self.model = BertForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            output_attentions=False,
            output_hidden_states=True,
        )
        
        # Freeze early layers for transfer learning
        self._freeze_layers(config.freeze_layers)
        
        # Additional classification head for domain adaptation
        hidden_size = self.model.config.hidden_size
        self.domain_adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, config.num_labels),
        )
    
    def _freeze_layers(self, num_layers: int) -> None:
        """Freeze early transformer layers."""
        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False
        
        for i in range(min(num_layers, self.model.config.num_hidden_layers)):
            for param in self.model.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        log.info(f"Froze {num_layers} transformer layers")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with domain adaptation."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # Get [CLS] token representation
        cls_output = outputs.hidden_states[-1][:, 0, :]
        
        # Domain-specific prediction
        domain_logits = self.domain_adapter(cls_output)
        
        # Combine with pretrained predictions
        combined_logits = (outputs.logits + domain_logits) / 2
        
        result = {
            "logits": combined_logits,
            "pretrained_logits": outputs.logits,
            "domain_logits": domain_logits,
            "hidden_states": outputs.hidden_states[-1],
        }
        
        if labels is not None:
            loss = F.cross_entropy(combined_logits, labels)
            result["loss"] = loss
        
        return result
    
    def fine_tune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> dict[str, Any]:
        """
        Fine-tune on domain-specific data.
        
        FIX: Transfer learning for China A-share sentiment
        """
        self.to(device)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_ratio,
        )
        
        best_val_acc = 0.0
        training_history = []
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                optimizer.zero_grad()
                outputs = self(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                loss = outputs["loss"]
                
                if self.config.use_fp16:
                    from torch.cuda.amp import GradScaler
                    scaler = GradScaler()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                
                train_loss += loss.item()
                predictions = outputs["logits"].argmax(dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
            
            # Validation
            val_metrics = self._validate(val_loader, device)
            
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss / len(train_loader),
                "train_accuracy": train_correct / train_total,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
            }
            training_history.append(epoch_metrics)
            
            log.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs}: "
                f"train_acc={epoch_metrics['train_accuracy']:.4f}, "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )
            
            # Save best model
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
        
        return {
            "training_history": training_history,
            "best_val_accuracy": best_val_acc,
            "final_train_loss": train_loss / len(train_loader),
        }
    
    def _validate(
        self,
        val_loader: DataLoader,
        device: torch.device,
    ) -> dict[str, float]:
        """Validation step."""
        self.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = self(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                
                predictions = outputs["logits"].argmax(dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score
        
        return {
            "accuracy": accuracy_score(all_labels, all_predictions),
            "f1": f1_score(all_labels, all_predictions, average="weighted"),
        }


class KnowledgeDistillation:
    """
    Knowledge distillation for model compression.
    
    FIX: Create smaller deployment models from large pretrained models
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
    ):
        """
        Initialize distillation.
        
        Args:
            teacher_model: Large pretrained model
            student_model: Smaller model to train
            temperature: Softening temperature for logits
            alpha: Weight for distillation loss (vs hard label loss)
        """
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
    
    def distill(
        self,
        train_loader: DataLoader,
        device: torch.device,
        num_epochs: int = 10,
    ) -> dict[str, Any]:
        """
        Distill knowledge from teacher to student.
        
        FIX: Model compression for efficient deployment
        """
        self.teacher.eval()
        self.student.train()
        
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=1e-3,
            weight_decay=0.01,
        )
        
        distillation_history = []
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            distill_loss = 0.0
            hard_loss = 0.0
            
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                optimizer.zero_grad()
                
                # Teacher predictions (soft targets)
                with torch.no_grad():
                    teacher_outputs = self.teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    teacher_logits = teacher_outputs["logits"]
                
                # Student predictions
                student_outputs = self.student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                student_logits = student_outputs["logits"]
                
                # Distillation loss (KL divergence)
                dist_loss = F.kl_div(
                    F.log_softmax(student_logits / self.temperature, dim=1),
                    F.softmax(teacher_logits / self.temperature, dim=1),
                    reduction="batchmean",
                ) * (self.temperature ** 2)
                
                # Hard label loss
                hard_loss = F.cross_entropy(student_logits, labels)
                
                # Combined loss
                loss = self.alpha * dist_loss + (1 - self.alpha) * hard_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                distill_loss += dist_loss.item()
                hard_loss += hard_loss.item()
            
            epoch_metrics = {
                "epoch": epoch + 1,
                "total_loss": total_loss / len(train_loader),
                "distill_loss": distill_loss / len(train_loader),
                "hard_loss": hard_loss / len(train_loader),
            }
            distillation_history.append(epoch_metrics)
            
            log.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"total_loss={epoch_metrics['total_loss']:.4f}"
            )
        
        return {
            "history": distillation_history,
            "final_loss": total_loss / len(train_loader),
        }


class FallbackSentimentModel(nn.Module):
    """
    Fallback sentiment model when transformers unavailable.
    
    FIX: Graceful degradation without pretrained models
    """
    
    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_labels: int = 3,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_labels),
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass."""
        embedded = self.embedding(input_ids)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate final forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        # Classification
        logits = self.classifier(hidden_cat)
        
        result = {"logits": logits}
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result["loss"] = loss
        
        return result


# Factory function
def create_sentiment_model(
    use_pretrained: bool = True,
    model_name: str = "bert-base-chinese",
    fine_tuning_config: Optional[FineTuningConfig] = None,
) -> nn.Module:
    """
    Create sentiment model with optional pretrained weights.
    
    FIX: Flexible model creation based on availability
    """
    if use_pretrained and TRANSFORMERS_AVAILABLE:
        config = fine_tuning_config or FineTuningConfig(model_name=model_name)
        return PretrainedSentimentModel(config)
    else:
        log.warning("Using fallback model (no pretrained weights)")
        return FallbackSentimentModel()
