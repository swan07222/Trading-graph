"""Tests for Recovery System.

Comprehensive tests for recovery mechanisms including:
- Recoverable exceptions and retry logic
- Recovery manager checkpoint operations
- Model guardian versioning and rollback
- Training checkpoint resume
- Fetch state recovery
- Recovery metrics

Run with: pytest tests/test_recovery.py -v
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import recovery modules
from utils.recoverable import (
    COMMON_RECOVERABLE_EXCEPTIONS,
    NETWORK_RECOVERABLE_EXCEPTIONS,
    RecoveryContext,
    RecoveryResult,
    RecoveryStrategy,
    retry_with_recovery,
)
from utils.recovery_manager import (
    CheckpointMetadata,
    RecoveryManager,
    RecoveryState,
    get_recovery_manager,
    reset_recovery_manager,
)
from utils.recovery_metrics import (
    RecoveryMetrics,
    get_recovery_metrics,
    record_recovery,
    reset_recovery_metrics,
)
from utils.fetch_recovery import (
    FetchSession,
    FetchStateRecovery,
    get_fetch_recovery,
    reset_fetch_recovery,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def recovery_manager(temp_dir):
    """Create a RecoveryManager instance."""
    manager = RecoveryManager(checkpoint_dir=temp_dir / "checkpoints")
    yield manager
    reset_recovery_manager()


@pytest.fixture
def recovery_metrics(temp_dir):
    """Create a RecoveryMetrics instance."""
    metrics = RecoveryMetrics(metrics_dir=temp_dir / "metrics")
    yield metrics
    reset_recovery_metrics()


@pytest.fixture
def fetch_recovery(temp_dir):
    """Create a FetchStateRecovery instance."""
    recovery = FetchStateRecovery(state_dir=temp_dir / "fetch_state")
    yield recovery
    reset_fetch_recovery()


# ============================================================================
# Tests: Recoverable Exceptions
# ============================================================================

class TestRecoverableExceptions:
    """Tests for recoverable exception handling."""

    def test_common_recoverable_exceptions_defined(self):
        """Test that common recoverable exceptions are defined."""
        assert len(COMMON_RECOVERABLE_EXCEPTIONS) > 0
        assert KeyError in COMMON_RECOVERABLE_EXCEPTIONS
        assert ValueError in COMMON_RECOVERABLE_EXCEPTIONS
        assert TimeoutError in COMMON_RECOVERABLE_EXCEPTIONS

    def test_network_recoverable_exceptions_defined(self):
        """Test that network recoverable exceptions are defined."""
        assert ConnectionError in NETWORK_RECOVERABLE_EXCEPTIONS
        assert TimeoutError in NETWORK_RECOVERABLE_EXCEPTIONS

    def test_retry_with_recovery_success(self):
        """Test successful operation with retry."""
        call_count = 0

        def succeed():
            nonlocal call_count
            call_count += 1
            return "success"

        result = retry_with_recovery(
            func=succeed,
            operation="test_success",
            max_attempts=3,
        )

        assert result.success is True
        assert result.result == "success"
        assert result.context.attempt == 1
        assert call_count == 1

    def test_retry_with_recovery_eventual_success(self):
        """Test operation succeeds after retries."""
        call_count = 0

        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"

        result = retry_with_recovery(
            func=fail_then_succeed,
            operation="test_eventual_success",
            max_attempts=5,
            base_delay=0.01,
        )

        assert result.success is True
        assert result.result == "success"
        assert result.context.attempt == 3
        assert call_count == 3

    def test_retry_with_recovery_failure(self):
        """Test operation fails after all retries."""
        call_count = 0

        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent error")

        result = retry_with_recovery(
            func=always_fail,
            operation="test_failure",
            max_attempts=3,
            base_delay=0.01,
        )

        assert result.success is False
        assert result.context.attempt == 3
        assert call_count == 3
        assert "failed after 3 attempts" in result.message

    def test_retry_with_recovery_fallback(self):
        """Test fallback function on failure."""
        def primary_fail():
            raise ValueError("Primary failed")

        def fallback():
            return "fallback_result"

        result = retry_with_recovery(
            func=primary_fail,
            operation="test_fallback",
            max_attempts=2,
            fallback=fallback,
            base_delay=0.01,
        )

        assert result.success is True
        assert result.result == "fallback_result"
        assert result.fallback_used is True

    def test_retry_with_recovery_callbacks(self):
        """Test retry with success/failure callbacks."""
        on_retry_called = []
        on_success_called = []

        def fail_once_then_succeed():
            if not on_retry_called:
                on_retry_called.append(True)
                raise ValueError("First fail")
            return "success"

        def on_retry(ctx):
            pass

        def on_success(result):
            on_success_called.append(result)

        result = retry_with_recovery(
            func=fail_once_then_succeed,
            operation="test_callbacks",
            max_attempts=3,
            base_delay=0.01,
            on_retry=on_retry,
            on_success=on_success,
        )

        assert result.success is True
        assert len(on_retry_called) == 1
        assert len(on_success_called) == 1

    def test_recovery_context_properties(self):
        """Test RecoveryContext properties."""
        ctx = RecoveryContext(
            operation="test",
            max_attempts=5,
            attempt=3,
        )

        assert ctx.can_retry is True
        assert ctx.should_abort is False

        ctx.attempt = 5
        assert ctx.can_retry is False

        ctx.strategy = RecoveryStrategy.ABORT
        assert ctx.should_abort is True

    def test_recovery_result_to_dict(self):
        """Test RecoveryResult serialization."""
        ctx = RecoveryContext(operation="test")
        result = RecoveryResult(
            success=True,
            context=ctx,
            result={"data": "value"},
            message="Success",
        )

        data = result.to_dict()
        assert data["success"] is True
        assert data["operation"] == "test"
        assert data["message"] == "Success"


# ============================================================================
# Tests: Recovery Manager
# ============================================================================

class TestRecoveryManager:
    """Tests for RecoveryManager."""

    def test_init_creates_directories(self, temp_dir):
        """Test initialization creates directories."""
        manager = RecoveryManager(checkpoint_dir=temp_dir / "checkpoints")

        assert (temp_dir / "checkpoints" / "training").exists()
        assert (temp_dir / "checkpoints" / "inference").exists()
        assert (temp_dir / "checkpoints" / "data").exists()

    def test_save_checkpoint(self, recovery_manager):
        """Test saving a checkpoint."""
        checkpoint_id = recovery_manager.save_checkpoint(
            key="test_checkpoint",
            state={"epoch": 10, "data": [1, 2, 3]},
            checkpoint_type="training",
            metrics={"accuracy": 0.95},
        )

        assert checkpoint_id is not None
        
        checkpoint_path = recovery_manager.checkpoint_dir / "training" / "test_checkpoint.pt"
        assert checkpoint_path.exists()

        metadata_path = recovery_manager.checkpoint_dir / "training" / "test_checkpoint.meta.json"
        assert metadata_path.exists()

    def test_load_checkpoint(self, recovery_manager):
        """Test loading a checkpoint."""
        # Save first
        recovery_manager.save_checkpoint(
            key="test_load",
            state={"key": "value", "number": 42},
            checkpoint_type="training",
        )

        # Load
        loaded = recovery_manager.load_checkpoint("test_load", "training")

        assert loaded is not None
        assert loaded["key"] == "value"
        assert loaded["number"] == 42

    def test_load_nonexistent_checkpoint(self, recovery_manager):
        """Test loading a nonexistent checkpoint."""
        loaded = recovery_manager.load_checkpoint("nonexistent", "training")
        assert loaded is None

    def test_load_latest_checkpoint(self, recovery_manager):
        """Test loading the latest checkpoint."""
        # Save multiple checkpoints
        recovery_manager.save_checkpoint(
            key="checkpoint_1",
            state={"version": 1},
            checkpoint_type="training",
        )
        time.sleep(0.01)
        recovery_manager.save_checkpoint(
            key="checkpoint_2",
            state={"version": 2},
            checkpoint_type="training",
        )

        # Load latest
        result = recovery_manager.load_latest_checkpoint("training")

        assert result is not None
        key, state = result
        assert key == "checkpoint_2"
        assert state["version"] == 2

    def test_list_checkpoints(self, recovery_manager):
        """Test listing checkpoints."""
        # Save checkpoints
        for i in range(3):
            recovery_manager.save_checkpoint(
                key=f"checkpoint_{i}",
                state={"index": i},
                checkpoint_type="training",
            )

        checkpoints = recovery_manager.list_checkpoints("training")

        assert len(checkpoints) == 3

    def test_delete_checkpoint(self, recovery_manager):
        """Test deleting a checkpoint."""
        # Save checkpoint
        recovery_manager.save_checkpoint(
            key="to_delete",
            state={"data": "value"},
            checkpoint_type="training",
        )

        # Delete
        deleted = recovery_manager.delete_checkpoint("to_delete", "training")

        assert deleted is True

        checkpoint_path = recovery_manager.checkpoint_dir / "training" / "to_delete.pt"
        assert not checkpoint_path.exists()

    def test_run_with_recovery(self, recovery_manager):
        """Test run_with_recovery method."""
        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary error")
            return {"result": "success"}

        result = recovery_manager.run_with_recovery(
            operation="test_run",
            func=operation,
            checkpoint_key="test_checkpoint",
            checkpoint_type="training",
            max_attempts=3,
            base_delay=0.01,
        )

        assert result.success is True
        assert result.result == {"result": "success"}
        assert call_count == 2

        # Check state updated
        state = recovery_manager.get_state()
        assert state.successful_operations >= 1

    def test_recovery_state_persistence(self, recovery_manager):
        """Test recovery state is persisted."""
        # Perform some operations
        recovery_manager.save_checkpoint(
            key="test",
            state={"data": "value"},
            checkpoint_type="training",
        )

        # Create new instance (should load state)
        new_manager = RecoveryManager(checkpoint_dir=recovery_manager.checkpoint_dir)
        state = new_manager.get_state()

        assert state.last_checkpoint == "test"

    def test_get_metrics(self, recovery_manager):
        """Test getting metrics."""
        metrics = recovery_manager.get_metrics()

        assert "state" in metrics
        assert "checkpoint_count" in metrics
        assert "checkpoints_by_type" in metrics

    def test_export_import_state(self, recovery_manager, temp_dir):
        """Test exporting and importing state."""
        # Save checkpoint
        recovery_manager.save_checkpoint(
            key="export_test",
            state={"data": "value"},
            checkpoint_type="training",
        )

        # Export
        export_path = temp_dir / "export.json"
        success = recovery_manager.export_state(export_path)
        assert success is True
        assert export_path.exists()

        # Import to new manager
        new_manager = RecoveryManager(checkpoint_dir=temp_dir / "new_checkpoints")
        success = new_manager.import_state(export_path)
        assert success is True

        # Verify state loaded
        state = new_manager.get_state()
        assert state.last_checkpoint == "export_test"


# ============================================================================
# Tests: Recovery Metrics
# ============================================================================

class TestRecoveryMetrics:
    """Tests for RecoveryMetrics."""

    def test_record_operation(self, recovery_metrics):
        """Test recording an operation."""
        recovery_metrics.record_operation(
            operation="test_op",
            success=True,
            duration_seconds=1.5,
            attempts=2,
        )

        metrics = recovery_metrics.get_operation_metrics("test_op")
        assert metrics is not None
        assert metrics.total_count == 1
        assert metrics.success_count == 1
        assert metrics.avg_attempts == 2.0

    def test_record_failure(self, recovery_metrics):
        """Test recording a failure."""
        recovery_metrics.record_operation(
            operation="test_fail",
            success=False,
            duration_seconds=2.0,
            attempts=3,
            error_type="ValueError",
        )

        metrics = recovery_metrics.get_operation_metrics("test_fail")
        assert metrics.failure_count == 1
        assert metrics.success_rate == 0.0

    def test_get_health_healthy(self, recovery_metrics):
        """Test health status when healthy."""
        # Record successful operations
        for i in range(10):
            recovery_metrics.record_operation(
                operation="success_op",
                success=True,
                duration_seconds=0.5,
            )

        health = recovery_metrics.get_health()

        assert health.status == "healthy"
        assert health.success_rate_24h == 1.0
        assert health.consecutive_failures == 0

    def test_get_health_unhealthy(self, recovery_metrics):
        """Test health status when unhealthy."""
        # Record consecutive failures
        for i in range(5):
            recovery_metrics.record_operation(
                operation="fail_op",
                success=False,
                duration_seconds=1.0,
            )

        health = recovery_metrics.get_health()

        assert health.status == "unhealthy"
        assert health.consecutive_failures == 5
        assert len(health.alerts) > 0

    def test_get_health_degraded(self, recovery_metrics):
        """Test health status when degraded."""
        # Record some failures
        for i in range(3):
            recovery_metrics.record_operation(
                operation="mixed_op",
                success=False,
                duration_seconds=1.0,
            )
        for i in range(7):
            recovery_metrics.record_operation(
                operation="mixed_op",
                success=True,
                duration_seconds=0.5,
            )

        health = recovery_metrics.get_health()

        assert health.status in ["healthy", "degraded"]
        assert health.success_rate_24h > 0.5

    def test_get_trends(self, recovery_metrics):
        """Test getting trends."""
        # Record operations
        for i in range(10):
            recovery_metrics.record_operation(
                operation="trend_op",
                success=True,
                duration_seconds=0.5,
            )

        trends = recovery_metrics.get_trends(hours=24)

        assert "trends" in trends
        assert len(trends["trends"]) > 0

    def test_get_top_failures(self, recovery_metrics):
        """Test getting top failures."""
        # Record failures
        for i in range(5):
            recovery_metrics.record_operation(
                operation="failing_op",
                success=False,
                duration_seconds=1.0,
            )

        failures = recovery_metrics.get_top_failures(limit=10)

        assert len(failures) > 0
        assert failures[0]["operation"] == "failing_op"

    def test_export_metrics(self, recovery_metrics, temp_dir):
        """Test exporting metrics."""
        recovery_metrics.record_operation(
            operation="export_test",
            success=True,
            duration_seconds=0.5,
        )

        export_path = temp_dir / "metrics_export.json"
        data = recovery_metrics.export_metrics(export_path)

        assert export_path.exists()
        assert "summary" in data
        assert "health" in data

    def test_record_recovery_convenience_function(self, recovery_metrics):
        """Test the record_recovery convenience function."""
        # Reset global instance
        reset_recovery_metrics()
        
        # Use convenience function
        record_recovery(
            operation="convenience_test",
            success=True,
            duration_seconds=0.5,
        )

        metrics = get_recovery_metrics()
        op_metrics = metrics.get_operation_metrics("convenience_test")
        assert op_metrics is not None
        assert op_metrics.total_count == 1


# ============================================================================
# Tests: Fetch Recovery
# ============================================================================

class TestFetchRecovery:
    """Tests for FetchStateRecovery."""

    def test_start_session(self, fetch_recovery):
        """Test starting a fetch session."""
        session_id = fetch_recovery.start_session(
            operation="fetch_history",
            symbols=["A", "B", "C"],
            interval="1m",
        )

        assert session_id is not None
        
        session = fetch_recovery.get_session(session_id)
        assert session is not None
        assert session.total_symbols == 3
        assert session.pending_count == 3
        assert session.status == "running"

    def test_mark_complete(self, fetch_recovery):
        """Test marking symbols complete."""
        session_id = fetch_recovery.start_session(
            operation="fetch_test",
            symbols=["A", "B", "C"],
        )

        fetch_recovery.mark_complete(session_id, "A")
        fetch_recovery.mark_complete(session_id, "B")

        session = fetch_recovery.get_session(session_id)
        assert session.completed_count == 2
        assert session.pending_count == 1
        assert "A" in session.completed_symbols
        assert "B" in session.completed_symbols

    def test_mark_failed(self, fetch_recovery):
        """Test marking symbols failed."""
        session_id = fetch_recovery.start_session(
            operation="fetch_test",
            symbols=["A", "B", "C"],
        )

        fetch_recovery.mark_failed(session_id, "B", "Error message")

        session = fetch_recovery.get_session(session_id)
        assert session.failed_count == 1
        assert "B" in session.failed_symbols
        assert session.last_error == "Error message"

    def test_retry_symbol(self, fetch_recovery):
        """Test retrying a failed symbol."""
        session_id = fetch_recovery.start_session(
            operation="fetch_test",
            symbols=["A", "B"],
        )

        fetch_recovery.mark_failed(session_id, "A", "Failed")
        
        # Retry
        retried = fetch_recovery.retry_symbol(session_id, "A")
        assert retried is True

        session = fetch_recovery.get_session(session_id)
        assert "A" in session.pending_symbols
        assert session.retry_counts.get("A", 0) == 1

    def test_pause_resume_session(self, fetch_recovery):
        """Test pausing and resuming a session."""
        session_id = fetch_recovery.start_session(
            operation="fetch_test",
            symbols=["A", "B", "C"],
        )

        # Pause
        paused = fetch_recovery.pause_session(session_id)
        assert paused is True
        
        session = fetch_recovery.get_session(session_id)
        assert session.status == "paused"

        # Resume
        resumed = fetch_recovery.resume_session(session_id)
        assert resumed is True
        
        session = fetch_recovery.get_session(session_id)
        assert session.status == "running"

    def test_complete_session(self, fetch_recovery):
        """Test completing a session."""
        session_id = fetch_recovery.start_session(
            operation="fetch_test",
            symbols=["A", "B"],
        )

        fetch_recovery.mark_complete(session_id, "A")
        fetch_recovery.mark_complete(session_id, "B")

        completed = fetch_recovery.complete_session(session_id)
        assert completed is True

        session = fetch_recovery.get_session(session_id)
        assert session.status == "completed"
        assert session.completed_at is not None

    def test_has_pending_session(self, fetch_recovery):
        """Test checking for pending session."""
        assert fetch_recovery.has_pending_session() is False

        session_id = fetch_recovery.start_session(
            operation="fetch_test",
            symbols=["A", "B"],
        )

        assert fetch_recovery.has_pending_session() is True

        # Complete all
        fetch_recovery.mark_complete(session_id, "A")
        fetch_recovery.mark_complete(session_id, "B")
        fetch_recovery.complete_session(session_id)

        assert fetch_recovery.has_pending_session() is False

    def test_load_pending_session(self, fetch_recovery):
        """Test loading pending session."""
        session_id = fetch_recovery.start_session(
            operation="fetch_test",
            symbols=["A", "B", "C"],
        )

        fetch_recovery.mark_complete(session_id, "A")

        session = fetch_recovery.load_pending_session()
        assert session is not None
        assert session.pending_count == 2

    def test_get_progress(self, fetch_recovery):
        """Test getting progress."""
        session_id = fetch_recovery.start_session(
            operation="fetch_test",
            symbols=["A", "B", "C", "D", "E"],
        )

        fetch_recovery.mark_complete(session_id, "A")
        fetch_recovery.mark_complete(session_id, "B")

        progress = fetch_recovery.get_progress(session_id)

        assert progress["completed"] == 2
        assert progress["pending"] == 3
        assert progress["progress_percent"] == 40.0

    def test_fetch_session_to_dict(self):
        """Test FetchSession serialization."""
        session = FetchSession(
            session_id="test_123",
            operation="fetch_test",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
            total_symbols=3,
            completed_symbols=["A"],
            pending_symbols=["B", "C"],
        )

        data = session.to_dict()
        assert data["session_id"] == "test_123"
        assert data["progress_percent"] > 0

        # Deserialize
        restored = FetchSession.from_dict(data)
        assert restored.session_id == session.session_id
        assert restored.completed_count == session.completed_count


# ============================================================================
# Tests: Checkpoint Metadata
# ============================================================================

class TestCheckpointMetadata:
    """Tests for CheckpointMetadata."""

    def test_create_metadata(self):
        """Test creating metadata."""
        metadata = CheckpointMetadata.create(
            checkpoint_type="training",
            labels={"epoch": "10"},
            metrics={"accuracy": 0.95},
        )

        assert metadata.checkpoint_type == "training"
        assert metadata.labels["epoch"] == "10"
        assert metadata.metrics["accuracy"] == 0.95
        assert metadata.checkpoint_id is not None

    def test_metadata_serialization(self):
        """Test metadata serialization."""
        metadata = CheckpointMetadata(
            checkpoint_id="test_123",
            checkpoint_type="training",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
            metrics={"loss": 0.05},
        )

        data = metadata.to_dict()
        restored = CheckpointMetadata.from_dict(data)

        assert restored.checkpoint_id == metadata.checkpoint_id
        assert restored.metrics == metadata.metrics


# ============================================================================
# Tests: Recovery State
# ============================================================================

class TestRecoveryState:
    """Tests for RecoveryState."""

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        state = RecoveryState(
            total_operations=10,
            successful_operations=8,
        )

        assert state.success_rate == 0.8

    def test_is_healthy(self):
        """Test health check."""
        healthy_state = RecoveryState(
            consecutive_failures=0,
            total_operations=10,
            successful_operations=9,
        )
        assert healthy_state.is_healthy is True

        unhealthy_state = RecoveryState(
            consecutive_failures=5,
            total_operations=10,
            successful_operations=3,
        )
        assert unhealthy_state.is_healthy is False

    def test_state_serialization(self):
        """Test state serialization."""
        state = RecoveryState(
            last_checkpoint="test_ckpt",
            recovery_count=5,
            consecutive_failures=2,
        )

        data = state.to_dict()
        assert data["last_checkpoint"] == "test_ckpt"
        assert data["consecutive_failures"] == 2
        assert "success_rate" in data


# ============================================================================
# Integration Tests
# ============================================================================

class TestRecoveryIntegration:
    """Integration tests for recovery system."""

    def test_full_training_checkpoint_flow(self, temp_dir):
        """Test complete training checkpoint workflow."""
        from models.training_checkpoint import TrainingCheckpoint

        checkpoint = TrainingCheckpoint(
            session_name="integration_test",
            checkpoint_dir=temp_dir / "checkpoints",
        )

        # Initialize session
        checkpoint.initialize_session(
            total_epochs=10,
            model_type="ensemble",
            interval="1m",
            horizon=30,
        )

        # Simulate training
        for epoch in range(5):
            checkpoint.save(
                {"epoch": epoch, "weights": [1.0, 2.0, 3.0]},
                epoch=epoch,
                metrics={"loss": 1.0 / (epoch + 1)},
            )

        # Verify checkpoints saved
        checkpoints = checkpoint.get_checkpoints()
        assert len(checkpoints) == 5

        # Load latest
        latest = checkpoint.load()
        assert latest is not None
        assert latest["epoch"] == 4

        # Get progress
        progress = checkpoint.get_progress()
        assert progress["current_epoch"] == 5
        assert progress["progress"] == 0.5

    def test_model_guardian_flow(self, temp_dir):
        """Test ModelGuardian workflow."""
        from models.auto_learner_components import ModelGuardian

        guardian = ModelGuardian(
            model_dir=temp_dir / "models",
            max_backups=3,
        )

        # Create mock model files
        model_dir = guardian.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)

        for filename in guardian._model_files("1m", 30):
            (model_dir / filename).write_text(f"model content for {filename}")

        # Create backup
        version_id = guardian.backup_current(
            "1m", 30,
            version_label="test_backup",
            metrics={"accuracy": 0.90},
        )

        assert version_id is not None

        # List versions
        versions = guardian.list_versions("1m", 30)
        assert len(versions) == 1
        assert versions[0]["label"] == "test_backup"

    def test_recovery_with_metrics_integration(self, temp_dir):
        """Test recovery operations with metrics tracking."""
        # Setup
        manager = RecoveryManager(checkpoint_dir=temp_dir / "checkpoints")
        metrics = RecoveryMetrics(metrics_dir=temp_dir / "metrics")

        # Run operations with recovery
        def operation():
            return {"result": "success"}

        result = manager.run_with_recovery(
            operation="integrated_test",
            func=operation,
            max_attempts=2,
            base_delay=0.01,
        )

        # Record in metrics
        metrics.record_operation(
            operation="integrated_test",
            success=result.success,
            duration_seconds=result.context.elapsed_seconds,
            attempts=result.context.attempt,
        )

        # Verify both systems updated
        assert manager.get_state().successful_operations >= 1
        
        health = metrics.get_health()
        assert health.success_rate_24h == 1.0


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
