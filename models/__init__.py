"""Models Module."""
try:
    from .layers import (
        AttentionPooling,
        LSTMBlock,
        MultiHeadAttention,
        PositionalEncoding,
        TemporalConvBlock,
        TransformerBlock,
    )
except Exception:
    MultiHeadAttention = None
    PositionalEncoding = None
    LSTMBlock = None
    TransformerBlock = None
    TemporalConvBlock = None
    AttentionPooling = None

try:
    from .networks import (
        GRUModel,
        HybridModel,
        LSTMModel,
        TCNModel,
        TransformerModel,
    )
except Exception:
    LSTMModel = None
    TransformerModel = None
    GRUModel = None
    TCNModel = None
    HybridModel = None

try:
    from .ensemble import EnsembleModel, EnsemblePrediction
except Exception:
    EnsembleModel = None
    EnsemblePrediction = None

try:
    from .trainer import Trainer
except Exception:
    Trainer = None

try:
    from .predictor import Predictor
except Exception:
    Predictor = None

try:
    from .auto_learner import AutoLearner, LearningProgress
except Exception:
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
