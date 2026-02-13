"""
Models Module
"""
from .ensemble import EnsembleModel, EnsemblePrediction
from .layers import (
    AttentionPooling,
    LSTMBlock,
    MultiHeadAttention,
    PositionalEncoding,
    TemporalConvBlock,
    TransformerBlock,
)
from .networks import (
    GRUModel,
    HybridModel,
    LSTMModel,
    TCNModel,
    TransformerModel,
)
from .trainer import Trainer

try:
    from .predictor import Predictor
except ImportError:
    Predictor = None

try:
    from .auto_learner import AutoLearner, LearningProgress
except ImportError:
    AutoLearner = None
    LearningProgress = None

__all__ = [
    'MultiHeadAttention',
    'PositionalEncoding',
    'LSTMBlock',
    'TransformerBlock',
    'TemporalConvBlock',
    'AttentionPooling',
    'LSTMModel',
    'TransformerModel',
    'GRUModel',
    'TCNModel',
    'HybridModel',
    'EnsembleModel',
    'EnsemblePrediction',
    'Trainer',
    'Predictor',
    'AutoLearner',
    'LearningProgress',
]