"""
Neural Network Architectures for Stock Prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .layers import (
    PositionalEncoding,
    MultiHeadAttention,
    TransformerBlock,
    LSTMBlock,
    TemporalConvBlock,
    AttentionPooling,
)


class LSTMModel(nn.Module):
    """LSTM with Multi-Head Attention"""
    
    def __init__(self, input_size: int, hidden_size: int = 256, 
                 num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        self.lstm = LSTMBlock(hidden_size, hidden_size, num_layers=2, dropout=dropout)
        
        self.attention = MultiHeadAttention(hidden_size * 2, num_heads=8, dropout=dropout)
        
        self.pool = AttentionPooling(hidden_size * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.confidence = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        x = self.lstm(x)
        x, _ = self.attention(x)
        pooled = self.pool(x)
        logits = self.classifier(pooled)
        conf = self.confidence(pooled)
        return logits, conf


class TransformerModel(nn.Module):
    """Transformer Encoder for sequence classification"""
    
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_classes: int = 3, dropout: float = 0.3,
                 num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, dropout=dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        self.pool = AttentionPooling(hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self.confidence = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        pooled = self.pool(x)
        logits = self.classifier(pooled)
        conf = self.confidence(pooled)
        return logits, conf


class GRUModel(nn.Module):
    """GRU with attention"""
    
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.pool = AttentionPooling(hidden_size * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.confidence = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        x, _ = self.gru(x)
        x = self.norm(x)
        pooled = self.pool(x)
        logits = self.classifier(pooled)
        conf = self.confidence(pooled)
        return logits, conf


class TCNModel(nn.Module):
    """Temporal Convolutional Network"""
    
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        self.tcn_blocks = nn.ModuleList([
            TemporalConvBlock(hidden_size, hidden_size, dilation=1, dropout=dropout),
            TemporalConvBlock(hidden_size, hidden_size, dilation=2, dropout=dropout),
            TemporalConvBlock(hidden_size, hidden_size, dilation=4, dropout=dropout),
            TemporalConvBlock(hidden_size, hidden_size, dilation=8, dropout=dropout),
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self.confidence = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        x = x.transpose(1, 2)  # (batch, hidden, seq)
        
        for block in self.tcn_blocks:
            x = block(x)
        
        x = self.pool(x).squeeze(-1)  # (batch, hidden)
        logits = self.classifier(x)
        conf = self.confidence(x)
        return logits, conf


class HybridModel(nn.Module):
    """CNN-LSTM Hybrid Model"""
    
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # CNN for local patterns
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2)
        self.conv_norm = nn.BatchNorm1d(hidden_size)
        
        # LSTM for sequential patterns
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.norm = nn.LayerNorm(hidden_size)
        self.pool = AttentionPooling(hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self.confidence = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        
        # CNN path
        x_t = x.transpose(1, 2)
        conv_out = F.gelu(self.conv1(x_t)) + F.gelu(self.conv2(x_t))
        conv_out = self.conv_norm(conv_out)
        conv_out = conv_out.transpose(1, 2)
        
        # Residual
        x = x + conv_out
        
        # LSTM path
        x, _ = self.lstm(x)
        x = self.norm(x)
        
        pooled = self.pool(x)
        logits = self.classifier(pooled)
        conf = self.confidence(pooled)
        return logits, conf