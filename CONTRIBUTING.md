# Contributing Guide

Welcome! This guide covers how to contribute to Trading Graph.

---

## Quick Links

- [Code of Conduct](#code-of-conduct)
- [Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing-guidelines)
- [Pull Requests](#pull-request-process)

---

## Code of Conduct

- Be respectful and inclusive
- Give constructive feedback
- Prioritize user security and privacy
- Maintain quality standards

---

## Development Setup

### Prerequisites

- Python 3.11+
- Git
- pip or uv

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/trading-graph.git
cd trading-graph

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements-dev.txt

# Verify setup
pytest -q
```

### Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with your settings
```

### IDE Configuration

**VS Code** (`.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.testing.pytestEnabled": true,
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "charliermarsh.ruff"
}
```

**PyCharm**:
- Set interpreter to `venv`
- Enable ruff linter
- Configure pytest
- Set line length to 100

---

## Coding Standards

### Type Hints

All public functions require type hints:

```python
# ✅ Correct
def calculate_pnl(entry: float, exit: float, quantity: int) -> float:
    """Calculate profit and loss."""
    return (exit - entry) * quantity

# ❌ Incorrect
def calculate_pnl(entry, exit, quantity):
    return (exit - entry) * quantity
```

### Docstrings

Use Google-style docstrings:

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
        quantity: Number of shares
        price: Limit price (None for market orders)

    Returns:
        Order object with assigned ID

    Raises:
        OrderError: If validation fails
        RiskError: If risk limits violated

    Example:
        >>> order = oms.submit_order("600519", OrderSide.BUY, 100, 50.0)
    """
```

### File Organization

**Import order:**
```python
# 1. Future imports
from __future__ import annotations

# 2. Standard library
import threading
from datetime import datetime

# 3. Third-party
import numpy as np
import pandas as pd

# 4. Local
from core.types import Order, OrderSide
from utils.logger import get_logger
```

**Limits:**
- Max 300 lines per file (excluding tests)
- Max 50 lines per function
- Max 5 parameters (use dataclasses for more)

### Error Handling

```python
# ✅ Correct - Specific exceptions
try:
    order = self._create_order(symbol, quantity)
except OrderValidationError as e:
    log.warning("Validation failed: %s", e)
    return None
except DatabaseError as e:
    log.error("Database error: %s", e)
    raise

# ❌ Incorrect - Bare except
try:
    order = self._create_order(symbol, quantity)
except:
    log.error("Error creating order")
    return None
```

### Naming Conventions

| Type | Style | Example |
|------|-------|---------|
| Variables | snake_case | `order_quantity` |
| Functions | snake_case | `calculate_pnl()` |
| Classes | PascalCase | `OrderManagement` |
| Constants | UPPER_SNAKE_CASE | `MAX_POSITION_PCT` |
| Private | `_` prefix | `_internal_method()` |

---

## Testing Guidelines

### Test Structure

```python
"""Tests for order management."""
import pytest
from unittest.mock import Mock

from core.types import Order, OrderSide
from trading.oms import OrderManagementSystem


class TestOrderManagement:
    """Test suite for OMS."""

    @pytest.fixture
    def oms(self) -> OrderManagementSystem:
        """Create OMS for testing."""
        return OrderManagementSystem(initial_capital=100000)

    def test_submit_order_success(self, oms: OrderManagementSystem) -> None:
        """Test successful order submission."""
        order = Order(symbol="600519", side=OrderSide.BUY, quantity=100, price=50.0)
        result = oms.submit_order(order)
        
        assert result.status == OrderStatus.SUBMITTED
        assert result.id is not None
```

### Coverage Requirements

| Module Type | Minimum Coverage |
|-------------|------------------|
| Overall | 85% |
| Critical (OMS, Risk, Security) | 95% |
| UI Components | 70% |

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=trading_graph --cov-report=html

# Specific file
pytest tests/test_oms.py

# Specific function
pytest tests/test_oms.py::TestOrderManagement::test_submit_order

# Verbose output
pytest -v
```

---

## Documentation

### Code Comments

```python
# ✅ Good - Explains WHY
# Use WAL mode for concurrent reads without blocking writes
# See: https://sqlite.org/wal.html
conn.execute("PRAGMA journal_mode=WAL")

# ❌ Bad - Restates code
# Set journal mode to WAL
conn.execute("PRAGMA journal_mode=WAL")
```

### When to Update README

- Adding new features
- Changing installation requirements
- Modifying CLI arguments
- Adding configuration options

---

## Pull Request Process

### Before Submitting

```bash
# 1. Run all tests
pytest --cov-fail-under=85

# 2. Run linter
ruff check .

# 3. Run type checker
mypy trading_graph/

# 4. Update documentation
# README, docstrings, etc.
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

### Changelog Format

```markdown
## [Unreleased]

### Added
- Feature description (#123)

### Changed
- Behavior description (#456)

### Fixed
- Bug description (#789)

### Removed
- Deprecated feature (#321)
```

---

## Performance Guidelines

### Memory Management

```python
# ✅ Good - Generator for large datasets
def load_bars(symbol: str) -> Iterator[Bar]:
    for row in database.query(symbol):
        yield Bar.from_row(row)

# ❌ Bad - Load all to memory
def load_bars(symbol: str) -> list[Bar]:
    return [Bar.from_row(row) for row in database.query(symbol)]
```

### Async Operations

```python
# ✅ Good - Concurrent async
async def fetch_multiple(symbols: list[str]) -> dict[str, Data]:
    tasks = [fetch_symbol(s) for s in symbols]
    return await asyncio.gather(*tasks)

# ❌ Bad - Sequential blocking
def fetch_multiple(symbols: list[str]) -> dict[str, Data]:
    return {s: fetch_symbol(s) for s in symbols}
```

---

## Security Guidelines

### Sensitive Data

```python
# ✅ Good - Secure storage
from utils.security import get_secure_storage

storage = get_secure_storage()
storage.set("api_key", secret_key)

# ❌ Bad - Plain text
with open("config.json", "w") as f:
    json.dump({"api_key": secret_key}, f)
```

### Audit Logging

```python
# ✅ Good - Security events logged
from utils.security import get_audit_log

audit = get_audit_log()
audit.log(
    event="ORDER_SUBMIT",
    user=user_id,
    details={"order_id": order.id, "symbol": symbol},
)
```

---

## Architecture Decisions

Document major decisions in `docs/adr/NNNN-decision-name.md`:

```markdown
# ADR NNNN: Decision Name

## Status
Proposed | Accepted | Deprecated | Superseded

## Context
What problem are we solving?

## Decision
What change are we making?

## Consequences
Trade-offs and implications?
```

### Current Decisions

| ADR | Title | Status |
|-----|-------|--------|
| 0001 | SQLite for persistence | Accepted |
| 0002 | PyQt6 for desktop UI | Accepted |
| 0003 | PyTorch for ML models | Accepted |
| 0004 | Event-driven architecture | Accepted |
| 0005 | Single-node design | Accepted |

---

## Getting Help

- **Docs**: `README.md`, `ARCHITECTURE.md`
- **Issues**: Open a GitHub issue
- **Discussions**: GitHub Discussions for questions

---

## Recognition

Contributors are recognized in:
- CHANGELOG.md
- README.md contributors section
- Annual release notes

Thank you for contributing!
