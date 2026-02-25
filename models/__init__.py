"""Models Module.

Available models:
    - Informer: Efficient Transformer for long sequences
    - TemporalFusionTransformer (TFT): Interpretable predictions
    - NBEATS: Neural basis expansion analysis
    - TSMixer: All-MLP architecture
"""
try:
    from .networks import (
        Informer,
        NBEATS,
        TSMixer,
        TemporalFusionTransformer,
    )
except Exception:
    Informer = None
    TemporalFusionTransformer = None
    NBEATS = None
    TSMixer = None

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
    'Informer',
    'TemporalFusionTransformer',
    'NBEATS',
    'TSMixer',
    'EnsembleModel',
    'EnsemblePrediction',
    'Trainer',
    'Predictor',
    'AutoLearner',
    'LearningProgress',
]
