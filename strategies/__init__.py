# strategies/__init__.py
"""Trading Strategy Library.

This module provides:
- 50+ pre-built trading strategies
- Genetic algorithm optimization
- Bayesian optimization
- Strategy registry and management
"""

from .strategy_library import (
    BaseStrategy,
    StrategyCategory,
    StrategyConfig,
    StrategySignal,
    StrategyRegistry,
    TimeFrame,
    get_strategy,
    list_all_strategies,
)

from .genetic_optimizer import (
    GeneticConfig,
    GeneticOptimizer,
    BayesianOptimizer,
    OptimizationResult,
    ParetoFront,
    optimize_strategy,
)

__all__ = [
    # Strategy Library
    "BaseStrategy",
    "StrategyCategory",
    "StrategyConfig",
    "StrategySignal",
    "StrategyRegistry",
    "TimeFrame",
    "get_strategy",
    "list_all_strategies",
    
    # Optimization
    "GeneticConfig",
    "GeneticOptimizer",
    "BayesianOptimizer",
    "OptimizationResult",
    "ParetoFront",
    "optimize_strategy",
]
