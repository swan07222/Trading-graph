# trading/__init__.py
"""Trading module - Sentiment Analysis and News Processing only.

This module has been refactored to focus on:
- News and policy data collection
- Sentiment analysis for market prediction
- Model training on news/policy data

Removed components:
- Portfolio management
- Risk management
- Order Management System (OMS)
- Broker integration
- Auto trading execution
"""

from typing import Any

__all__: list[str] = []


def __getattr__(name: str) -> Any:
    """Lazy loading for backward compatibility during transition."""
    if name in {"Portfolio", "RiskManager", "OrderManagementSystem", "AutoTrader", "SimulatorBroker"}:
        raise ImportError(
            f"{name} has been removed from this build. "
            "Trading execution components are no longer available."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
