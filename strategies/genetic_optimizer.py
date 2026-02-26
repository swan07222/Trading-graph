# strategies/genetic_optimizer.py
"""Genetic Algorithm Strategy Optimization.

This module provides evolutionary optimization for trading strategies:
- Genetic algorithm for parameter optimization
- Multi-objective optimization (Sharpe, Drawdown, Returns)
- Pareto front analysis
- Population-based search
- Crossover and mutation operators
- Elitism and tournament selection
"""

import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from deap import base, creator, tools

from analysis.advanced_backtest import AdvancedBacktestEngine, BacktestMetrics
from utils.logger import get_logger

from .strategy_library import BaseStrategy, StrategyRegistry

log = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Optimization result."""
    parameters: dict[str, Any]
    metrics: dict[str, float]
    fitness: float
    generation: int
    rank: int


@dataclass
class ParetoFront:
    """Pareto optimal front."""
    solutions: list[OptimizationResult]
    objectives: list[str]

    def get_best(self, objective: str) -> OptimizationResult:
        """Get best solution for specific objective."""
        return max(self.solutions, key=lambda x: x.metrics.get(objective, 0))


@dataclass
class GeneticConfig:
    """Genetic algorithm configuration."""
    population_size: int = 50
    generations: int = 30
    crossover_rate: float = 0.7
    mutation_rate: float = 0.2
    elitism_rate: float = 0.1
    tournament_size: int = 3

    # Multi-objective weights
    sharpe_weight: float = 0.4
    return_weight: float = 0.3
    drawdown_weight: float = 0.3

    # Constraints
    min_trades: int = 30
    max_drawdown: float = 0.30


class GeneticOptimizer:
    """Genetic algorithm optimizer for trading strategies."""

    def __init__(
        self,
        strategy_name: str,
        config: GeneticConfig | None = None,
    ) -> None:
        self.strategy_name = strategy_name
        self.config = config or GeneticConfig()
        self.strategy = StrategyRegistry.get_strategy(strategy_name)

        # Define parameter ranges for optimization
        self.param_ranges = self._get_param_ranges()

        # Setup DEAP framework
        self._setup_deap()

    def _get_param_ranges(self) -> dict[str, tuple]:
        """Get parameter optimization ranges."""
        ranges = {
            # Moving averages
            "fast_period": (5, 50),
            "slow_period": (20, 200),
            "short_period": (5, 30),
            "medium_period": (20, 60),
            "long_period": (60, 200),

            # RSI
            "rsi_period": (7, 28),
            "oversold": (10, 40),
            "overbought": (60, 90),

            # Bollinger Bands
            "period": (10, 50),
            "std_dev": (1.0, 3.0),

            # MACD
            "fast": (8, 20),
            "slow": (20, 40),
            "signal": (5, 15),

            # ADX
            "adx_period": (10, 20),
            "adx_threshold": (15, 35),

            # SuperTrend
            "multiplier": (1.0, 5.0),

            # Breakout
            "volume_ma_period": (10, 50),
            "spike_threshold": (1.5, 5.0),

            # Z-Score
            "lookback": (10, 50),
            "entry_z": (1.0, 3.0),
            "exit_z": (0.0, 1.5),

            # Donchian
            "donchian_period": (10, 55),

            # Risk parameters
            "max_position_pct": (0.05, 0.25),
            "stop_loss_pct": (0.02, 0.10),
            "take_profit_pct": (0.05, 0.30),
        }
        return ranges

    def _setup_deap(self) -> None:
        """Setup DEAP optimization framework."""
        # Create fitness and individual classes
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0))  # Max Sharpe, Min DD, Max Return
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()

        # Register attribute generators
        for param_name, (min_val, max_val) in self.param_ranges.items():
            if isinstance(min_val, float):
                self.toolbox.register(
                    f"attr_{param_name}",
                    random.uniform,
                    min_val,
                    max_val,
                )
            else:
                self.toolbox.register(
                    f"attr_{param_name}",
                    random.randint,
                    min_val,
                    max_val,
                )

        # Register individual creation
        self._register_individual_creator()

        # Register operators
        self.toolbox.register("mate", tools.cxUniform, indpb=0.2)
        self.toolbox.register("mutate", self._mutate_individual, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=self.config.tournament_size)

    def _register_individual_creator(self) -> None:
        """Register individual creator based on strategy parameters."""
        attrs = []
        for param_name in self.strategy.parameters.keys():
            if param_name in self.param_ranges:
                attrs.append(getattr(self.toolbox, f"attr_{param_name}"))

        if attrs:
            creator.create("Individual", list, fitness=creator.FitnessMulti)
            self.toolbox.register(
                "individual",
                tools.initCycle,
                creator.Individual,
                attrs,
                n=1,
            )
        else:
            # Fallback: create individual with random parameters
            self.toolbox.register(
                "individual",
                tools.initIterate,
                creator.Individual,
                lambda: [random.uniform(0, 1) for _ in range(5)],
            )

        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual,
        )

    def _mutate_individual(self, individual: list, indpb: float) -> tuple:
        """Mutate individual with parameter constraints."""
        size = len(individual)
        param_names = list(self.param_ranges.keys())[:size]

        for i in range(size):
            if random.random() < indpb:
                param_name = param_names[i] if i < len(param_names) else f"param_{i}"
                min_val, max_val = self.param_ranges.get(param_name, (0, 1))

                if isinstance(min_val, float):
                    individual[i] = random.uniform(min_val, max_val)
                else:
                    individual[i] = random.randint(min_val, max_val)

        return (individual,)

    def _decode_individual(
        self,
        individual: list,
    ) -> dict[str, Any]:
        """Decode individual to strategy parameters."""
        params = {}
        param_names = list(self.strategy.parameters.keys())

        for i, param_name in enumerate(param_names):
            if i < len(individual):
                min_val, max_val = self.param_ranges.get(param_name, (0, 1))

                if isinstance(min_val, float):
                    params[param_name] = float(individual[i])
                else:
                    params[param_name] = int(individual[i])

        return params

    def _evaluate_individual(
        self,
        individual: list,
        data: pd.DataFrame,
        signal_generator: Callable,
    ) -> tuple[float, float, float]:
        """Evaluate individual fitness."""
        try:
            # Decode parameters
            params = self._decode_individual(individual)

            # Update strategy parameters
            for key, value in params.items():
                if key in self.strategy.parameters:
                    self.strategy.parameters[key] = value

            # Generate signals
            signals = signal_generator(data, **params)

            # Run backtest
            engine = AdvancedBacktestEngine()
            metrics = engine.run(data, signals)

            # Check constraints
            if metrics.total_trades < self.config.min_trades:
                return (-1000, 1.0, -1000)  # Penalty

            if metrics.max_drawdown > self.config.max_drawdown:
                return (-1000, 1.0, -1000)  # Penalty

            # Calculate fitness (Sharpe, -Drawdown, Return)
            fitness = (
                metrics.sharpe_ratio,
                metrics.max_drawdown,
                metrics.total_return / 100,
            )

            return fitness

        except Exception as e:
            log.debug(f"Evaluation failed: {e}")
            return (-1000, 1.0, -1000)  # Penalty for failed evaluations

    def optimize(
        self,
        data: pd.DataFrame,
        signal_generator: Callable,
        verbose: bool = True,
    ) -> tuple[list[OptimizationResult], ParetoFront]:
        """Run genetic optimization.

        Args:
            data: OHLCV data for optimization
            signal_generator: Function to generate signals
            verbose: Print progress

        Returns:
            Tuple of (all results, Pareto front)
        """
        # Create initial population
        population = self.toolbox.population(n=self.config.population_size)

        # Hall of Fame for best individuals
        hof = tools.HallOfFame(max(1, int(self.config.population_size * 0.1)))

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        # Create evaluation function with data
        def eval_func(ind):
            return self._evaluate_individual(ind, data, signal_generator)

        # Register evaluation
        self.toolbox.register("evaluate", eval_func)

        # Run optimization
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        all_results = []

        for gen in range(self.config.generations):
            # Evaluate individuals
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

                # Store result
                if fit[0] > -999:  # Valid evaluation
                    params = self._decode_individual(ind)
                    result = OptimizationResult(
                        parameters=params,
                        metrics={
                            "sharpe": fit[0],
                            "max_drawdown": fit[1],
                            "total_return": fit[2] * 100,
                        },
                        fitness=fit[0] * self.config.sharpe_weight -
                               fit[1] * self.config.drawdown_weight +
                               fit[2] * self.config.return_weight,
                        generation=gen,
                        rank=0,
                    )
                    all_results.append(result)

            # Update Hall of Fame
            hof.update(population)

            # Compile statistics
            record = stats.compile(population) if population else {}
            logbook.record(gen=gen, evals=len(invalid_ind), **record)

            if verbose:
                print(logbook.stream)

            # Selection
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.config.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.config.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Replace population
            population[:] = offspring

        # Extract Pareto front
        pareto_front = self._extract_pareto_front(all_results)

        return all_results, pareto_front

    def _extract_pareto_front(
        self,
        results: list[OptimizationResult],
    ) -> ParetoFront:
        """Extract Pareto optimal solutions."""
        if not results:
            return ParetoFront(solutions=[], objectives=["sharpe", "drawdown", "return"])

        # Non-dominated sorting
        pareto = []
        for result in results:
            is_dominated = False

            for other in results:
                if result == other:
                    continue

                # Check domination
                dominates = (
                    other.metrics["sharpe"] >= result.metrics["sharpe"] and
                    other.metrics["max_drawdown"] <= result.metrics["max_drawdown"] and
                    other.metrics["total_return"] >= result.metrics["total_return"] and
                    (
                        other.metrics["sharpe"] > result.metrics["sharpe"] or
                        other.metrics["max_drawdown"] < result.metrics["max_drawdown"] or
                        other.metrics["total_return"] > result.metrics["total_return"]
                    )
                )

                if dominates:
                    is_dominated = True
                    break

            if not is_dominated:
                result.rank = 1
                pareto.append(result)

        return ParetoFront(
            solutions=pareto,
            objectives=["sharpe", "max_drawdown", "total_return"],
        )


class BayesianOptimizer:
    """Bayesian Optimization for strategy parameters."""

    def __init__(
        self,
        strategy_name: str,
        n_iterations: int = 50,
    ) -> None:
        self.strategy_name = strategy_name
        self.n_iterations = n_iterations
        self.strategy = StrategyRegistry.get_strategy(strategy_name)

        try:
            from skopt import gp_minimize
            self.gp_minimize = gp_minimize
            self.available = True
        except ImportError:
            self.available = False
            log.warning("scikit-optimize not available for Bayesian optimization")

    def optimize(
        self,
        data: pd.DataFrame,
        signal_generator: Callable,
    ) -> OptimizationResult:
        """Run Bayesian optimization."""
        if not self.available:
            raise ImportError("scikit-optimize required for Bayesian optimization")

        # Define search space
        dimensions = []
        param_names = []

        for param_name, (min_val, max_val) in self._get_param_ranges().items():
            if param_name in self.strategy.parameters:
                dimensions.append((min_val, max_val))
                param_names.append(param_name)

        # Objective function
        def objective(params):
            params_dict = dict(zip(param_names, params))

            try:
                # Update strategy
                for key, value in params_dict.items():
                    self.strategy.parameters[key] = value

                # Generate signals and backtest
                signals = signal_generator(data, **params_dict)
                engine = AdvancedBacktestEngine()
                metrics = engine.run(data, signals)

                # Return negative Sharpe (for minimization)
                return -metrics.sharpe_ratio

            except Exception:
                return 1000  # Penalty

        # Run optimization
        result = self.gp_minimize(
            objective,
            dimensions,
            n_calls=self.n_iterations,
            random_state=42,
        )

        # Create optimization result
        best_params = dict(zip(param_names, result.x))
        best_sharpe = -result.fun

        return OptimizationResult(
            parameters=best_params,
            metrics={"sharpe": best_sharpe},
            fitness=best_sharpe,
            generation=self.n_iterations,
            rank=1,
        )

    def _get_param_ranges(self) -> dict[str, tuple]:
        """Get parameter ranges (same as genetic optimizer)."""
        return {
            "fast_period": (5, 50),
            "slow_period": (20, 200),
            "rsi_period": (7, 28),
            "oversold": (10, 40),
            "overbought": (60, 90),
            "period": (10, 50),
            "std_dev": (1.0, 3.0),
            "adx_period": (10, 20),
            "adx_threshold": (15, 35),
        }


def optimize_strategy(
    strategy_name: str,
    data: pd.DataFrame,
    signal_generator: Callable,
    method: str = "genetic",
    **kwargs: Any,
) -> dict[str, Any]:
    """Optimize strategy parameters.

    Args:
        strategy_name: Strategy to optimize
        data: OHLCV data
        signal_generator: Signal generation function
        method: "genetic" or "bayesian"
        **kwargs: Additional optimizer arguments

    Returns:
        Optimization results dict
    """
    if method == "genetic":
        config = GeneticConfig(**kwargs) if "config" not in kwargs else kwargs["config"]
        optimizer = GeneticOptimizer(strategy_name, config)
        results, pareto = optimizer.optimize(data, signal_generator)

        return {
            "method": "genetic",
            "all_results": results,
            "pareto_front": pareto,
            "best_solution": pareto.get_best("sharpe") if pareto.solutions else None,
        }

    elif method == "bayesian":
        n_iterations = kwargs.get("n_iterations", 50)
        optimizer = BayesianOptimizer(strategy_name, n_iterations)
        result = optimizer.optimize(data, signal_generator)

        return {
            "method": "bayesian",
            "best_solution": result,
        }

    else:
        raise ValueError(f"Unknown optimization method: {method}")
