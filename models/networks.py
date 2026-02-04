"""
Neural Network Architectures - Built from scratch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .layers import (
    PositionalEncoding, MultiHeadAttention, TransformerBlock,
    LSTMBlock, TemporalConvBlock, AttentionPooling, ConfidenceHead
)
from config import CONFIG


class BaseModel(nn.Module):
    """Base class for all models"""
    
    def __init__(self):
        super().__init__()
        self.input_size = None
        self.hidden_size = None
        self.num_classes = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, features)
            
        Returns:
            logits: Class logits (batch, num_classes)
            confidence: Confidence score (batch, 1)
        """
        raise NotImplementedError


class LSTMModel(BaseModel):
    """
    Bidirectional LSTM with Multi-Head Attention
    
    Architecture:
    1. Input projection
    2. Bidirectional LSTM
    3. Multi-head self-attention
    4. Attention pooling
    5. Classification head
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 3,
                 num_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # LSTM block
        self.lstm = LSTMBlock(hidden_size, hidden_size, num_layers, dropout)
        
        # Self-attention
        self.attention = MultiHeadAttention(hidden_size * 2, num_heads=8, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size * 2)
        
        # Pooling
        self.pool = AttentionPooling(hidden_size * 2)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Confidence head
        self.confidence = ConfidenceHead(hidden_size * 2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input projection
        h = self.input_proj(x)
        
        # LSTM
        h = self.lstm(h)
        
        # Self-attention
        attn_out, _ = self.attention(h)
        h = self.norm(h + attn_out)
        
        # Pooling
        context = self.pool(h)
        
        # Output
        logits = self.classifier(context)
        conf = self.confidence(context)
        
        return logits, conf


class TransformerModel(BaseModel):
    """
    Transformer Encoder for Time Series Classification
    
    Architecture:
    1. Input projection
    2. Positional encoding
    3. Transformer encoder blocks
    4. Global average pooling
    5. Classification head
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 num_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_size, dropout=dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.confidence = ConfidenceHead(hidden_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input projection + positional encoding
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        
        # Transformer blocks
        for block in self.blocks:
            h = block(h)
        
        h = self.norm(h)
        
        # Global average pooling
        context = h.mean(dim=1)
        
        # Output
        logits = self.classifier(context)
        conf = self.confidence(context)
        
        return logits, conf


class GRUModel(BaseModel):
    """
    Bidirectional GRU with Attention
    
    Simpler and faster than LSTM, often works well for time series
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 num_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # GRU
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Attention pooling
        self.pool = AttentionPooling(hidden_size * 2)
        
        # Output
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.confidence = ConfidenceHead(hidden_size * 2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        h, _ = self.gru(h)
        
        context = self.pool(h)
        
        logits = self.classifier(context)
        conf = self.confidence(context)
        
        return logits, conf


class TCNModel(BaseModel):
    """
    Temporal Convolutional Network
    
    Uses dilated causal convolutions for capturing long-range dependencies
    More parallelizable than RNNs
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 4,
                 num_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # TCN blocks with increasing dilation
        self.tcn_blocks = nn.ModuleList([
            TemporalConvBlock(hidden_size, hidden_size, 
                            kernel_size=3, dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])
        
        # Output
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.confidence = ConfidenceHead(hidden_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # (batch, seq, features) -> (batch, features, seq)
        h = self.input_proj(x).transpose(1, 2)
        
        for block in self.tcn_blocks:
            h = block(h)
        
        # Global average pooling
        context = h.mean(dim=2)
        
        logits = self.classifier(context)
        conf = self.confidence(context)
        
        return logits, conf


class HybridModel(BaseModel):
    """
    Hybrid CNN-LSTM Model
    
    Combines:
    1. CNN for local pattern extraction
    2. LSTM for sequential modeling
    3. Attention for focusing on important parts
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 256,
                 num_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # CNN for local features
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, hidden_size // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
        )
        
        # LSTM for sequential modeling
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=2,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Attention
        self.attention = MultiHeadAttention(hidden_size * 2, num_heads=8, dropout=dropout)
        self.pool = AttentionPooling(hidden_size * 2)
        
        # Output
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.confidence = ConfidenceHead(hidden_size * 2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # CNN: (batch, seq, feat) -> (batch, feat, seq) -> (batch, hidden, seq)
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        
        # LSTM
        h, _ = self.lstm(h)
        
        # Attention
        attn_out, _ = self.attention(h)
        h = h + attn_out
        
        # Pooling
        context = self.pool(h)
        
        logits = self.classifier(context)
        conf = self.confidence(context)
        
        return logits, conf