"""Modern Redis caching layer with async support and distributed features.

This module provides:
    - Async Redis client with connection pooling
    - Multiple cache strategies (LRU, TTL, write-through)
    - Distributed locking for coordination
    - Pub/sub for real-time messaging
    - Cache invalidation patterns
    - Metrics and monitoring

Example:
    >>> cache = RedisCache()
    >>> await cache.connect()
    >>> await cache.set("stock:600519:price", 1850.5, ttl=60)
    >>> price = await cache.get("stock:600519:price")
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

import redis.asyncio as redis
from redis.asyncio.lock import Lock as RedisLock
from redis.exceptions import LockError, RedisError

from config.runtime_env import env_int, env_text
from utils.logger import get_logger

log = get_logger()

T = TypeVar("T")


@dataclass
class CacheConfig:
    """Redis cache configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    ssl: bool = False
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    decode_responses: bool = True
    default_ttl: int = 3600  # 1 hour
    key_prefix: str = "trading:"


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    errors: int = 0
    keys_count: int = 0
    memory_used: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "keys_count": self.keys_count,
            "memory_used_mb": self.memory_used / (1024 * 1024),
            "hit_rate": f"{self.hit_rate:.2%}",
        }


class CacheKey:
    """Helper for building cache keys with consistent naming."""

    def __init__(self, prefix: str = "trading") -> None:
        self.prefix = prefix

    def stock(self, symbol: str, suffix: str = "") -> str:
        """Build stock-related cache key."""
        key = f"{self.prefix}:stock:{symbol}"
        return f"{key}:{suffix}" if suffix else key

    def price(self, symbol: str) -> str:
        """Build price cache key."""
        return self.stock(symbol, "price")

    def bars(self, symbol: str, interval: str = "1d") -> str:
        """Build bars cache key."""
        return self.stock(symbol, f"bars:{interval}")

    def features(self, symbol: str, date: str) -> str:
        """Build features cache key."""
        return self.stock(symbol, f"features:{date}")

    def prediction(self, symbol: str, model: str) -> str:
        """Build prediction cache key."""
        return self.stock(symbol, f"pred:{model}")

    def sentiment(self, symbol: str) -> str:
        """Build sentiment cache key."""
        return self.stock(symbol, "sentiment")

    def news(self, category: str = "all") -> str:
        """Build news cache key."""
        return f"{self.prefix}:news:{category}"

    def model(self, name: str) -> str:
        """Build model cache key."""
        return f"{self.prefix}:model:{name}"

    def user(self, user_id: str, suffix: str = "") -> str:
        """Build user-related cache key."""
        key = f"{self.prefix}:user:{user_id}"
        return f"{key}:{suffix}" if suffix else key

    def lock(self, resource: str) -> str:
        """Build lock key."""
        return f"{self.prefix}:lock:{resource}"

    def pubsub(self, channel: str) -> str:
        """Build pub/sub channel key."""
        return f"{self.prefix}:channel:{channel}"


class RedisCache:
    """Async Redis cache with enterprise features.

    Features:
        - Connection pooling
        - Automatic reconnection
        - Multiple data type support
        - Distributed locking
        - Pub/sub messaging
        - Cache statistics
        - Key expiration management

    Example:
        >>> cache = RedisCache()
        >>> await cache.connect()
        >>> await cache.set("key", "value", ttl=3600)
        >>> value = await cache.get("key")
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        """Initialize Redis cache.

        Args:
            config: Cache configuration
        """
        self._config = config or self._default_config()
        self._client: redis.Redis | None = None
        self._stats = CacheStats()
        self._connected = False
        self._key_builder = CacheKey(self._config.key_prefix)

    def _default_config(self) -> CacheConfig:
        """Create default configuration from environment."""
        return CacheConfig(
            host=env_text("REDIS_HOST", "localhost"),
            port=env_int("REDIS_PORT", 6379),
            password=env_text("REDIS_PASSWORD", None),
            db=env_int("REDIS_DB", 0),
            ssl=env_text("REDIS_SSL", "0") == "1",
            max_connections=env_int("REDIS_MAX_CONNECTIONS", 50),
        )

    @property
    def key(self) -> CacheKey:
        """Get key builder."""
        return self._key_builder

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    async def connect(self) -> None:
        """Connect to Redis with connection pooling."""
        if self._connected:
            return

        log.info(f"Connecting to Redis at {self._config.host}:{self._config.port}")

        try:
            # Create connection pool
            pool = redis.ConnectionPool(
                host=self._config.host,
                port=self._config.port,
                db=self._config.db,
                password=self._config.password,
                ssl=self._config.ssl,
                max_connections=self._config.max_connections,
                socket_timeout=self._config.socket_timeout,
                socket_connect_timeout=self._config.socket_connect_timeout,
                retry_on_timeout=self._config.retry_on_timeout,
                decode_responses=self._config.decode_responses,
            )

            self._client = redis.Redis(connection_pool=pool)

            # Test connection
            await self._client.ping()

            self._connected = True
            log.info("Redis connection established")

        except RedisError as e:
            log.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if not self._connected:
            return

        if self._client:
            await self._client.close()
            self._client = None
            self._connected = False

        log.info("Redis connection closed")

    async def _ensure_connected(self) -> None:
        """Ensure connected to Redis."""
        if not self._connected:
            await self.connect()

    async def get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        await self._ensure_connected()

        try:
            value = await self._client.get(key)
            if value is None:
                self._stats.misses += 1
                return None
            self._stats.hits += 1

            # Try to parse as JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        except RedisError as e:
            log.error(f"Cache get error: {e}")
            self._stats.errors += 1
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (auto-serialized to JSON)
            ttl: Time to live in seconds
            nx: Only set if key does not exist
            xx: Only set if key exists

        Returns:
            True if successful
        """
        await self._ensure_connected()

        try:
            # Serialize to JSON if not string
            if not isinstance(value, str):
                value = json.dumps(value)

            # Set with optional TTL
            if ttl is None:
                ttl = self._config.default_ttl

            if nx:
                result = await self._client.set(key, value, ex=ttl, nx=True)
            elif xx:
                result = await self._client.set(key, value, ex=ttl, xx=True)
            else:
                result = await self._client.set(key, value, ex=ttl)

            return bool(result)

        except RedisError as e:
            log.error(f"Cache set error: {e}")
            self._stats.errors += 1
            return False

    async def delete(self, *keys: str) -> int:
        """Delete one or more keys.

        Args:
            keys: Keys to delete

        Returns:
            Number of keys deleted
        """
        await self._ensure_connected()

        try:
            return await self._client.delete(*keys)
        except RedisError as e:
            log.error(f"Cache delete error: {e}")
            self._stats.errors += 1
            return 0

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        await self._ensure_connected()

        try:
            return bool(await self._client.exists(key))
        except RedisError as e:
            log.error(f"Cache exists error: {e}")
            self._stats.errors += 1
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration on key.

        Args:
            key: Cache key
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        await self._ensure_connected()

        try:
            return await self._client.expire(key, ttl)
        except RedisError as e:
            log.error(f"Cache expire error: {e}")
            self._stats.errors += 1
            return False

    async def incr(self, key: str, amount: int = 1) -> int | None:
        """Increment integer value.

        Args:
            key: Cache key
            amount: Amount to increment by

        Returns:
            New value or None if error
        """
        await self._ensure_connected()

        try:
            return await self._client.incr(key, amount)
        except RedisError as e:
            log.error(f"Cache incr error: {e}")
            self._stats.errors += 1
            return None

    async def get_multi(self, *keys: str) -> dict[str, Any]:
        """Get multiple values.

        Args:
            keys: Keys to retrieve

        Returns:
            Dictionary of key-value pairs
        """
        await self._ensure_connected()

        try:
            values = await self._client.mget(*keys)
            result = {}
            for key, value in zip(keys, values, strict=False):
                if value is not None:
                    self._stats.hits += 1
                    try:
                        result[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        result[key] = value
                else:
                    self._stats.misses += 1
            return result

        except RedisError as e:
            log.error(f"Cache get_multi error: {e}")
            self._stats.errors += 1
            return {}

    async def set_multi(
        self,
        mapping: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Set multiple values.

        Args:
            mapping: Dictionary of key-value pairs
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        await self._ensure_connected()

        try:
            if ttl:
                # Use pipeline for atomic operation with TTL
                pipe = self._client.pipeline()
                for key, value in mapping.items():
                    if not isinstance(value, str):
                        value = json.dumps(value)
                    pipe.setex(key, ttl, value)
                await pipe.execute()
            else:
                # Simple mset without TTL
                serialized = {
                    k: json.dumps(v) if not isinstance(v, str) else v
                    for k, v in mapping.items()
                }
                await self._client.mset(serialized)
            return True

        except RedisError as e:
            log.error(f"Cache set_multi error: {e}")
            self._stats.errors += 1
            return False

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: int | None = None,
    ) -> Any:
        """Get value or compute and cache it.

        Args:
            key: Cache key
            factory: Function to compute value if not cached
            ttl: Time to live in seconds

        Returns:
            Cached or computed value
        """
        value = await self.get(key)
        if value is not None:
            return value

        # Compute value
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        # Cache it
        await self.set(key, value, ttl)
        return value

    # Distributed locking
    async def acquire_lock(
        self,
        name: str,
        timeout: float = 10.0,
        blocking: bool = True,
        blocking_timeout: float = 10.0,
    ) -> RedisLock | None:
        """Acquire distributed lock.

        Args:
            name: Lock name
            timeout: Lock timeout in seconds
            blocking: Whether to block waiting for lock
            blocking_timeout: How long to wait for lock

        Returns:
            Lock object or None if not acquired
        """
        await self._ensure_connected()

        try:
            lock = self._client.lock(
                self._key_builder.lock(name),
                timeout=timeout,
                blocking=blocking,
                blocking_timeout=blocking_timeout,
            )
            await lock.acquire()
            log.debug(f"Acquired lock: {name}")
            return lock

        except LockError as e:
            log.warning(f"Failed to acquire lock {name}: {e}")
            return None
        except RedisError as e:
            log.error(f"Lock error: {e}")
            return None

    async def release_lock(self, lock: RedisLock) -> bool:
        """Release distributed lock.

        Args:
            lock: Lock to release

        Returns:
            True if successful
        """
        try:
            await lock.release()
            log.debug("Lock released")
            return True
        except LockError as e:
            log.error(f"Failed to release lock: {e}")
            return False

    @asynccontextmanager
    async def distributed_lock(
        self,
        name: str,
        timeout: float = 10.0,
    ) -> AsyncGenerator[RedisLock | None, None]:
        """Context manager for distributed locking.

        Args:
            name: Lock name
            timeout: Lock timeout

        Yields:
            Lock object or None

        Example:
            >>> async with cache.distributed_lock("resource") as lock:
            ...     if lock:
            ...         # Critical section
            ...         await process_resource()
        """
        lock = await self.acquire_lock(name, timeout)
        try:
            yield lock
        finally:
            if lock:
                await self.release_lock(lock)

    # Pub/Sub
    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel.

        Args:
            channel: Channel name
            message: Message to publish

        Returns:
            Number of subscribers that received the message
        """
        await self._ensure_connected()

        try:
            if not isinstance(message, str):
                message = json.dumps(message)
            return await self._client.publish(
                self._key_builder.pubsub(channel),
                message,
            )
        except RedisError as e:
            log.error(f"Pub publish error: {e}")
            return 0

    async def subscribe(
        self,
        *channels: str,
        message_handler: Callable[[str, Any], None] | None = None,
    ) -> redis.client.PubSub:
        """Subscribe to channels.

        Args:
            channels: Channels to subscribe to
            message_handler: Optional message handler

        Returns:
            PubSub object
        """
        await self._ensure_connected()

        try:
            pubsub = self._client.pubsub()
            redis_channels = [
                self._key_builder.pubsub(c) for c in channels
            ]
            await pubsub.subscribe(*redis_channels)

            if message_handler:
                # Start background listener
                asyncio.create_task(
                    self._listen_loop(pubsub, message_handler)
                )

            return pubsub

        except RedisError as e:
            log.error(f"Pub subscribe error: {e}")
            raise

    async def _listen_loop(
        self,
        pubsub: redis.client.PubSub,
        handler: Callable[[str, Any], None],
    ) -> None:
        """Listen for messages in background."""
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    channel = message["channel"]
                    data = message["data"]
                    try:
                        data = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        log.debug("Failed to parse Redis message data: %s", data)
                        continue
                    try:
                        handler(channel, data)
                    except Exception as e:
                        log.error("Handler error for channel %s: %s", channel, e)
                else:
                    log.debug("Received non-message event from Redis: %s", message.get("type"))
        except asyncio.CancelledError:
            log.debug("Redis listen loop cancelled")
            raise
        except RedisError as e:
            log.error("Listen loop error: %s", e)
            raise
        except Exception as e:
            log.error("Unexpected error in Redis listen loop: %s", e)
            raise

    # Cache management
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "trading:stock:*")

        Returns:
            Number of keys deleted
        """
        await self._ensure_connected()

        try:
            keys = []
            async for key in self._client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                return await self._client.delete(*keys)
            return 0

        except RedisError as e:
            log.error(f"Clear pattern error: {e}")
            return 0

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        await self._ensure_connected()

        try:
            info = await self._client.info("memory")
            self._stats.memory_used = info.get("used_memory", 0)

            keys_count = await self._client.dbsize()
            self._stats.keys_count = keys_count

            return self._stats

        except RedisError as e:
            log.error(f"Get stats error: {e}")
            return self._stats

    async def health_check(self) -> dict[str, Any]:
        """Check Redis health."""
        if not self._connected:
            return {"status": "disconnected", "healthy": False}

        try:
            await self._client.ping()
            stats = await self.get_stats()
            return {
                "status": "healthy",
                "healthy": True,
                "stats": stats.to_dict(),
            }
        except RedisError as e:
            return {
                "status": "unhealthy",
                "healthy": False,
                "error": str(e),
            }


# Global cache instance
_cache_instance: RedisCache | None = None


def get_cache() -> RedisCache:
    """Get global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance


async def init_cache() -> RedisCache:
    """Initialize and connect to Redis."""
    cache = get_cache()
    await cache.connect()
    return cache


# Decorator for caching function results
def async_cache(
    key_prefix: str = "func",
    ttl: int = 3600,
    key_builder: Callable[..., str] | None = None,
) -> Callable:
    """Decorator for caching async function results.

    Args:
        key_prefix: Prefix for cache keys
        ttl: Time to live in seconds
        key_builder: Optional function to build cache key

    Returns:
        Decorated function

    Example:
        @async_cache(key_prefix="stock_price", ttl=60)
        async def get_price(symbol: str) -> float:
            return await fetch_price(symbol)
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = get_cache()

            # Build cache key
            if key_builder:
                key = key_builder(*args, **kwargs)
            else:
                # Auto-generate key from function name and args
                key_parts = [key_prefix, func.__name__]
                key_parts.extend(str(a) for a in args)
                key_parts.extend(f"{k}={v}" for k, v in kwargs.items())
                key_hash = hashlib.md5(
                    ":".join(key_parts).encode()
                ).hexdigest()
                key = f"{cache.key.prefix}:{key_prefix}:{key_hash}"

            # Try cache
            cached = await cache.get(key)
            if cached is not None:
                return cached

            # Call function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Cache result
            await cache.set(key, result, ttl)
            return result

        return wrapper
    return decorator
