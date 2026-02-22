from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace


def _load_auto_learn_workers_module():
    path = Path("ui/auto_learn_workers.py").resolve()
    spec = spec_from_file_location("auto_learn_workers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load ui/auto_learn_workers.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_training_interval_alias_and_fallback():
    workers = _load_auto_learn_workers_module()
    assert workers.normalize_training_interval("15m") == "15m"
    assert workers.normalize_training_interval("h1") == "1h"
    assert workers.normalize_training_interval("bad-token") == "1m"


def test_targeted_worker_passes_requested_interval(monkeypatch):
    workers = _load_auto_learn_workers_module()
    captured: dict[str, object] = {}

    class _FakeLearner:
        def __init__(self) -> None:
            self.progress = SimpleNamespace(is_running=False)
            self._thread = None
            self._cb = None

        def add_callback(self, cb):
            self._cb = cb

        def start_targeted(self, **kwargs):
            captured.update(kwargs)
            if callable(self._cb):
                self._cb(
                    SimpleNamespace(
                        progress=35,
                        message="running",
                        stage="cycle_start",
                        stocks_processed=2,
                        processed_count=2,
                        validation_accuracy=0.52,
                        stocks_found=2,
                    )
                )

        def stop(self, join_timeout: float = 6.0):  # noqa: ARG002
            return None

    monkeypatch.setattr(workers, "_get_auto_learner", lambda: _FakeLearner)

    worker = workers.TargetedLearnWorker(
        {
            "stock_codes": ["600519", "000001"],
            "epochs": 12,
            "interval": "15m",
            "horizon": 40,
            "incremental": True,
            "continuous": False,
        }
    )
    results: list[dict] = []
    worker.finished_result.connect(lambda payload: results.append(payload))
    worker.run()

    assert captured["interval"] == "15m"
    assert int(captured["prediction_horizon"]) == 40
    assert results and results[-1]["status"] == "ok"


def test_auto_worker_passes_requested_interval(monkeypatch):
    workers = _load_auto_learn_workers_module()
    captured: dict[str, object] = {}

    class _FakeLearner:
        def __init__(self) -> None:
            self.progress = SimpleNamespace(is_running=False)
            self._thread = None
            self._cb = None

        def add_callback(self, cb):
            self._cb = cb

        def start(self, **kwargs):
            captured.update(kwargs)
            if callable(self._cb):
                self._cb(
                    SimpleNamespace(
                        progress=15,
                        message="started",
                        stage="discovering",
                        stocks_processed=1,
                        processed_count=1,
                        validation_accuracy=0.41,
                        stocks_found=1,
                    )
                )

        def stop(self, join_timeout: float = 6.0):  # noqa: ARG002
            return None

    monkeypatch.setattr(workers, "_get_auto_learner", lambda: _FakeLearner)

    worker = workers.AutoLearnWorker(
        {
            "mode": "full",
            "max_stocks": 100,
            "epochs": 10,
            "interval": "30m",
            "horizon": 25,
            "incremental": True,
        }
    )
    results: list[dict] = []
    worker.finished_result.connect(lambda payload: results.append(payload))
    worker.run()

    assert captured["interval"] == "30m"
    assert int(captured["prediction_horizon"]) == 25
    assert results and results[-1]["status"] == "ok"
