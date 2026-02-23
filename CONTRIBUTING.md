# Contributing to Trading Graph

Thank you for your interest in contributing to Trading Graph! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Architecture Decisions](#architecture-decisions)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Prioritize user security and data privacy
- Maintain production-grade quality standards

## Getting Started

### Prerequisites

- Python 3.10, 3.11, or 3.12
- pip or uv package manager
- Git for version control

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/trading-graph.git
cd trading-graph

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests to verify setup
pytest -q
```

## Development Setup

### Environment Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit with your settings
# TRADING_MODE=simulation
# TRADING_CAPITAL=100000
```

### IDE Setup

**VS Code Settings** (`.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.testing.pytestEnabled": true,
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "charliermarsh.ruff"
}
```

**PyCharm Settings**:
- Set project interpreter to venv
- Enable ruff as linter
- Configure pytest as test runner
- Set code style to match project (100 char line length)

## Coding Standards

### Type Hints

All public functions and methods MUST have type hints:

```python
# ✅ Good
def calculate_pnl(entry: float, exit: float, quantity: int) -> float:
    """Calculate profit and loss."""
    return (exit - entry) * quantity

# ❌ Bad
def calculate_pnl(entry, exit, quantity):
    return (exit - entry) * quantity
```

### Docstrings

Use Google-style docstrings for all public APIs:

```python
def submit_order(
    self,
    symbol: str,
    side: OrderSide,
    quantity: int,
    price: float | None = None,
) -> Order:
    """
    Submit an order to the market.

    Args:
        symbol: Stock symbol (e.g., "600519")
        side: Order side (BUY or SELL)
        quantity: Number of shares (must be multiple of lot size)
        price: Limit price (None for market orders)

    Returns:
        Order object with assigned order ID

    Raises:
        OrderError: If order validation fails
        RiskError: If order violates risk limits

    Example:
        >>> order = oms.submit_order("600519", OrderSide.BUY, 100, 50.0)
        >>> print(order.id)
        'ORD-20260223-001'
    """
```

### Code Organization

**File Size Limits**:
- Maximum 300 lines per file (excluding tests)
- Maximum 50 lines per function
- Maximum 5 parameters per function (use dataclasses for more)

**Module Structure**:
```python
# 1. Future imports
from __future__ import annotations

# 2. Standard library imports
import threading
from datetime import datetime
from typing import Any

# 3. Third-party imports
import numpy as np
import pandas as pd

# 4. Local imports
from core.types import Order, OrderSide
from utils.logger import get_logger

# 5. Module constants
_DEFAULT_TIMEOUT = 30.0

# 6. Classes and functions
```

### Error Handling

```python
# ✅ Good - Specific exception handling
try:
    order = self._create_order(symbol, quantity)
except OrderValidationError as e:
    log.warning("Order validation failed: %s", e)
    return None
except DatabaseError as e:
    log.error("Database error: %s", e)
    raise

# ❌ Bad - Bare except
try:
    order = self._create_order(symbol, quantity)
except:
    log.error("Error creating order")
    return None
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Variables | snake_case | `order_quantity` |
| Functions | snake_case | `calculate_pnl()` |
| Classes | PascalCase | `OrderManagementSystem` |
| Constants | UPPER_SNAKE_CASE | `MAX_POSITION_PCT` |
| Private | Leading underscore | `_internal_method()` |
| Type aliases | PascalCase | `OrderDict = dict[str, Order]` |

## Testing Guidelines

### Test Structure

```python
"""Tests for order management system."""
import pytest
from unittest.mock import Mock, patch

from core.types import Order, OrderSide, OrderType
from trading.oms import OrderManagementSystem

class TestOrderManagementSystem:
    """Test suite for OMS."""

    @pytest.fixture
    def oms(self) -> OrderManagementSystem:
        """Create OMS instance for testing."""
        return OrderManagementSystem(initial_capital=100000)

    def test_submit_order_success(self, oms: OrderManagementSystem) -> None:
        """Test successful order submission."""
        order = Order(
            symbol="600519",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=50.0,
        )
        result = oms.submit_order(order)
        assert result.status == OrderStatus.SUBMITTED
        assert result.id is not None

    def test_submit_order_insufficient_funds(self, oms: OrderManagementSystem) -> None:
        """Test order rejection due to insufficient funds."""
        order = Order(
            symbol="600519",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10000,  # Too large
            price=50.0,
        )
        result = oms.submit_order(order)
        assert result.status == OrderStatus.REJECTED
        assert "Insufficient funds" in result.message
```

### Coverage Requirements

- **Overall**: 85% minimum
- **Critical modules** (oms, risk, security): 95% minimum
- **UI components**: 70% minimum

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=trading_graph --cov-report=html

# Run specific test file
pytest tests/test_oms.py

# Run specific test function
pytest tests/test_oms.py::TestOrderManagementSystem::test_submit_order_success

# Run with verbose output
pytest -v
```

## Documentation

### Code Comments

```python
# ✅ Good - Explains WHY, not WHAT
# Use WAL mode for concurrent reads without blocking writes
# See: https://sqlite.org/wal.html
conn.execute("PRAGMA journal_mode=WAL")

# ❌ Bad - Restates code
# Set journal mode to WAL
conn.execute("PRAGMA journal_mode=WAL")
```

### README Updates

Update README.md when:
- Adding new features
- Changing installation requirements
- Modifying CLI arguments
- Adding configuration options

### API Documentation

Generate API docs:
```bash
# Install sphinx
pip install sphinx sphinx-autodoc-typehints

# Generate docs
sphinx-apidoc -o docs/ trading_graph
make -C docs html
```

## Pull Request Process

### Before Submitting

1. **Run all tests**: `pytest --cov-fail-under=85`
2. **Run linter**: `ruff check .`
3. **Run type checker**: `mypy trading_graph/`
4. **Update documentation**: README, docstrings, etc.
5. **Add changelog entry**: See below

### Changelog Format

```markdown
## [Unreleased]

### Added
- New feature description (#123)

### Changed
- Modified behavior description (#456)

### Fixed
- Bug fix description (#789)

### Removed
- Deprecated feature removal (#321)
```

### PR Template

```markdown
## Description
Brief description of changes

## Related Issue
Fixes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass
- [ ] Coverage maintained
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

## Architecture Decisions

### Decision Record Format

Create `docs/adr/NNNN-decision-name.md`:

```markdown
# ADR NNNN: Decision Name

## Status
Proposed | Accepted | Deprecated | Superseded

## Context
What is the issue that we're seeing?

## Decision
What is the change that we're proposing?

## Consequences
What becomes easier or more difficult?
What are the trade-offs?
```

### Current Architecture Decisions

| ADR | Title | Status |
|-----|-------|--------|
| 0001 | Use SQLite for persistence | Accepted |
| 0002 | PyQt6 for desktop UI | Accepted |
| 0003 | PyTorch for ML models | Accepted |
| 0004 | Event-driven architecture | Accepted |
| 0005 | Single-node design | Accepted |

## Performance Guidelines

### Memory Management

```python
# ✅ Good - Use generators for large datasets
def load_bars(symbol: str) -> Iterator[Bar]:
    for row in database.query(symbol):
        yield Bar.from_row(row)

# ❌ Bad - Load all into memory
def load_bars(symbol: str) -> list[Bar]:
    return [Bar.from_row(row) for row in database.query(symbol)]
```

### Async Operations

```python
# ✅ Good - Offload blocking operations
async def fetch_multiple_symbols(symbols: list[str]) -> dict[str, Data]:
    tasks = [fetch_symbol(s) for s in symbols]
    return await asyncio.gather(*tasks)

# ❌ Bad - Sequential blocking
def fetch_multiple_symbols(symbols: list[str]) -> dict[str, Data]:
    return {s: fetch_symbol(s) for s in symbols}
```

## Security Guidelines

### Handling Sensitive Data

```python
# ✅ Good - Use secure storage
from utils.security import get_secure_storage

storage = get_secure_storage()
storage.set("api_key", secret_key)

# ❌ Bad - Plain text storage
with open("config.json", "w") as f:
    json.dump({"api_key": secret_key}, f)
```

### Audit Logging

```python
# ✅ Good - Log security-relevant events
from utils.security import get_audit_log

audit = get_audit_log()
audit.log(
    event="ORDER_SUBMIT",
    user=user_id,
    details={"order_id": order.id, "symbol": symbol},
)

# ❌ Bad - No audit trail
log.info("Order submitted")
```

## Getting Help

- **Documentation**: See `ARCHITECTURE.md`, `README.md`
- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions for questions

## Recognition

Contributors will be recognized in:
- CHANGELOG.md
- README.md contributors section
- Annual release notes

Thank you for contributing to Trading Graph!
