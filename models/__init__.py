"""
Models Module
"""
from .layers import (
    MultiHeadAttention,
    PositionalEncoding,
    LSTMBlock,
    TransformerBlock,
    TemporalConvBlock,
    AttentionPooling,
)
from .networks import (
    LSTMModel,
    TransformerModel,
    GRUModel,
    TCNModel,
    HybridModel,
)
from .ensemble import EnsembleModel, EnsemblePrediction
from .trainer import Trainer


# Optional imports
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