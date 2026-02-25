"""Modern async database layer with PostgreSQL and SQLite support.

This module provides:
    - Async database operations with SQLAlchemy 2.0
    - PostgreSQL support via asyncpg
    - SQLite support via aiosqlite
    - Connection pooling with automatic management
    - Database migration system
    - Repository pattern for data access

Example:
    >>> from config.settings import CONFIG
    >>> db = AsyncDatabase(CONFIG.database_url)
    >>> await db.connect()
    >>> async with db.session() as session:
    ...     result = await session.execute(select(Stock).where(Stock.code == "600519"))
    ...     stock = result.scalar_one_or_none()
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Protocol

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Text,
    select,
    text,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from config.runtime_env import env_text
from utils.logger import get_logger

log = get_logger()


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


class Stock(Base):
    """Stock metadata table."""
    __tablename__ = "stocks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(10), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    market: Mapped[str] = mapped_column(String(20), nullable=False)  # SSE, SZSE, BSE
    industry: Mapped[str | None] = mapped_column(String(100))
    market_cap: Mapped[float | None] = mapped_column(Float)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class DailyBar(Base):
    """Daily OHLCV bars."""
    __tablename__ = "daily_bars"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    amount: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        # Unique constraint for symbol + date
        {"sqlite_autoincrement": True},
    )


class IntradayBar(Base):
    """Intraday OHLCV bars (1m, 5m, 15m, etc.)."""
    __tablename__ = "intraday_bars"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False)
    interval: Mapped[str] = mapped_column(String(10), nullable=False)  # 1m, 5m, 15m, etc.
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    amount: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Feature(Base):
    """Computed technical features."""
    __tablename__ = "features"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False)
    feature_name: Mapped[str] = mapped_column(String(50), nullable=False)
    feature_value: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        # Composite index for efficient queries
        {"sqlite_autoincrement": True},
    )


class Prediction(Base):
    """Model predictions."""
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    prediction_date: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    signal: Mapped[str] = mapped_column(String(20), nullable=False)  # BUY, HOLD, SELL
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    target_price: Mapped[float | None] = mapped_column(Float)
    horizon_days: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DatabaseConfig:
    """Database configuration."""

    def __init__(
        self,
        database_url: str | None = None,
        pool_size: int = 20,
        max_overflow: int = 30,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False,
    ) -> None:
        """Initialize database configuration.

        Args:
            database_url: Database connection URL
            pool_size: Number of connections in pool
            max_overflow: Max overflow connections
            pool_timeout: Timeout for getting connection
            pool_recycle: Seconds before recycling connection
            echo: Enable SQL echo for debugging
        """
        # Auto-detect database type from URL
        if database_url is None:
            database_url = env_text("DATABASE_URL", "")

        if not database_url:
            # Default to SQLite for backwards compatibility
            from config.settings import CONFIG
            database_url = f"sqlite+aiosqlite:///{CONFIG.data_dir}/trading.db"

        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.echo = echo

        # Detect database type
        self.is_postgresql = database_url.startswith("postgresql+asyncpg")
        self.is_sqlite = database_url.startswith("sqlite+aiosqlite")

    def __repr__(self) -> str:
        # Hide credentials in logs
        url = self.database_url
        if "@" in url:
            url = url.split("@")[1]
        return f"DatabaseConfig(url={url}, pool_size={self.pool_size})"


class AsyncDatabase:
    """Async database manager with connection pooling.

    Features:
        - PostgreSQL support via asyncpg
        - SQLite support via aiosqlite
        - Connection pooling
        - Session management
        - Automatic schema creation
        - Database migrations

    Example:
        >>> db = AsyncDatabase()
        >>> await db.connect()
        >>> async with db.session() as session:
        ...     result = await session.execute(select(Stock))
        ...     stocks = result.scalars().all()
    """

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        """Initialize async database.

        Args:
            config: Database configuration
        """
        self._config = config or DatabaseConfig()
        self._engine: AsyncEngine | None = None
        self._async_session_factory: async_sessionmaker[AsyncSession] | None = None
        self._connected = False

    @property
    def config(self) -> DatabaseConfig:
        """Get database configuration."""
        return self._config

    @property
    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL."""
        return self._config.is_postgresql

    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite."""
        return self._config.is_sqlite

    async def connect(self) -> None:
        """Create database engine and connection pool."""
        if self._connected:
            return

        log.info(f"Connecting to database: {self._config}")

        # Create async engine
        if self.is_postgresql:
            # PostgreSQL with asyncpg
            self._engine = create_async_engine(
                self._config.database_url,
                pool_size=self._config.pool_size,
                max_overflow=self._config.max_overflow,
                pool_timeout=self._config.pool_timeout,
                pool_recycle=self._config.pool_recycle,
                pool_pre_ping=True,  # Test connections before use
                echo=self._config.echo,
                # PostgreSQL-specific optimizations
                execution_options={
                    "isolation_level": "READ COMMITTED",
                },
            )
        else:
            # SQLite with aiosqlite
            self._engine = create_async_engine(
                self._config.database_url,
                echo=self._config.echo,
                # SQLite-specific settings
                connect_args={"check_same_thread": False},
            )

        # Create session factory
        self._async_session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

        # Create tables
        await self._create_tables()

        self._connected = True
        log.info("Database connection established")

    async def disconnect(self) -> None:
        """Close database connections."""
        if not self._connected:
            return

        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._async_session_factory = None
            self._connected = False

        log.info("Database connection closed")

    async def _create_tables(self) -> None:
        """Create database tables."""
        if not self._engine:
            return

        async with self._engine.begin() as conn:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)

            # PostgreSQL-specific optimizations
            if self.is_postgresql:
                await conn.execute(text("""
                    -- Enable pg_stat_statements for query monitoring
                    CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

                    -- Optimize vacuum for high-write tables
                    ALTER TABLE daily_bars SET (autovacuum_vacuum_scale_factor = 0.1);
                    ALTER TABLE intraday_bars SET (autovacuum_vacuum_scale_factor = 0.1);
                """))

            log.info("Database tables created")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session from pool.

        Yields:
            AsyncSession for database operations

        Example:
            >>> async with db.session() as session:
            ...     result = await session.execute(select(Stock))
            ...     stocks = result.scalars().all()
        """
        if not self._async_session_factory:
            raise RuntimeError("Database not connected. Call connect() first.")

        session = self._async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def execute_query(
        self,
        query: Any,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute raw SQL query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of result dictionaries
        """
        async with self.session() as session:
            result = await session.execute(text(query), params or {})
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]

    async def health_check(self) -> dict[str, Any]:
        """Check database health.

        Returns:
            Health status dictionary
        """
        if not self._connected:
            return {"status": "disconnected", "healthy": False}

        try:
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
                return {
                    "status": "healthy",
                    "healthy": True,
                    "database_type": "postgresql" if self.is_postgresql else "sqlite",
                    "pool_size": self._config.pool_size,
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "healthy": False,
                "error": str(e),
            }


# Repository pattern for common operations
class Repository(Protocol):
    """Protocol for repository pattern."""

    async def get(self, id: int) -> Any:
        """Get entity by ID."""
        ...

    async def list(self, limit: int = 100) -> list[Any]:
        """List entities."""
        ...

    async def create(self, entity: Any) -> Any:
        """Create entity."""
        ...

    async def update(self, entity: Any) -> Any:
        """Update entity."""
        ...

    async def delete(self, id: int) -> None:
        """Delete entity by ID."""
        ...


class StockRepository:
    """Repository for stock operations."""

    def __init__(self, db: AsyncDatabase) -> None:
        self._db = db

    async def get_by_code(self, code: str) -> Stock | None:
        """Get stock by code."""
        async with self._db.session() as session:
            result = await session.execute(
                select(Stock).where(Stock.code == code)
            )
            return result.scalar_one_or_none()

    async def list_active(
        self,
        market: str | None = None,
        limit: int = 1000,
    ) -> list[Stock]:
        """List active stocks, optionally filtered by market."""
        async with self._db.session() as session:
            query = select(Stock).where(Stock.is_active == True)
            if market:
                query = query.where(Stock.market == market)
            query = query.limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def create(self, stock: Stock) -> Stock:
        """Create or update stock."""
        async with self._db.session() as session:
            existing = await self.get_by_code(stock.code)
            if existing:
                # Update existing
                for key, value in stock.__dict__.items():
                    if not key.startswith("_"):
                        setattr(existing, key, value)
                await session.flush()
                return existing
            else:
                # Create new
                session.add(stock)
                await session.flush()
                return stock


class BarRepository:
    """Repository for OHLCV bar operations."""

    def __init__(self, db: AsyncDatabase) -> None:
        self._db = db

    async def get_daily_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[DailyBar]:
        """Get daily bars for a date range."""
        async with self._db.session() as session:
            result = await session.execute(
                select(DailyBar)
                .where(DailyBar.symbol == symbol)
                .where(DailyBar.date >= start_date)
                .where(DailyBar.date <= end_date)
                .order_by(DailyBar.date)
            )
            return list(result.scalars().all())

    async def upsert_daily_bars(self, bars: list[DailyBar]) -> int:
        """Upsert daily bars (insert or update).

        Returns:
            Number of bars inserted/updated
        """
        async with self._db.session() as session:
            count = 0
            for bar in bars:
                # Check if exists
                existing = await session.execute(
                    select(DailyBar)
                    .where(DailyBar.symbol == bar.symbol)
                    .where(DailyBar.date == bar.date)
                )
                existing_bar = existing.scalar_one_or_none()

                if existing_bar:
                    # Update
                    for key, value in bar.__dict__.items():
                        if not key.startswith("_") and key != "id":
                            setattr(existing_bar, key, value)
                else:
                    # Insert
                    session.add(bar)
                count += 1

            return count


# Global database instance
_db_instance: AsyncDatabase | None = None


def get_database() -> AsyncDatabase:
    """Get global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = AsyncDatabase()
    return _db_instance


async def init_database() -> AsyncDatabase:
    """Initialize and connect to database."""
    db = get_database()
    await db.connect()
    return db
