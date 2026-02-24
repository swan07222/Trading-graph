from __future__ import annotations

from types import SimpleNamespace

from ui import app_analysis_ops


class _ActionStub:
    def __init__(self) -> None:
        self.enabled_calls: list[bool] = []

    def setEnabled(self, value: bool) -> None:  # noqa: N802
        self.enabled_calls.append(bool(value))


class _ProgressStub:
    def __init__(self) -> None:
        self.hide_calls = 0

    def hide(self) -> None:
        self.hide_calls += 1


class _StatusStub:
    def __init__(self) -> None:
        self.values: list[str] = []

    def setText(self, value: str) -> None:  # noqa: N802
        self.values.append(str(value))


def _mk_self(*, current_seq: int, worker_seq: int) -> SimpleNamespace:
    action = _ActionStub()
    progress = _ProgressStub()
    status = _StatusStub()
    stock_input = SimpleNamespace(text=lambda: "600519")
    worker = SimpleNamespace(_request_seq=worker_seq)
    logs: list[tuple[str, str]] = []
    dbg: list[str] = []
    return SimpleNamespace(
        _analyze_request_seq=current_seq,
        workers={"analyze": worker},
        _ui_norm=lambda x: str(x or "").strip(),
        _debug_console=lambda *_args, **_kwargs: dbg.append("debug"),
        stock_input=stock_input,
        analyze_action=action,
        progress=progress,
        status_label=status,
        log=lambda msg, level="info": logs.append((str(msg), str(level))),
        _dbg=dbg,
        _logs=logs,
    )


def test_on_analysis_done_ignores_stale_result_sequence() -> None:
    stub = _mk_self(current_seq=3, worker_seq=3)
    pred = SimpleNamespace(stock_code="600519")

    app_analysis_ops._on_analysis_done(stub, pred, request_seq=2)

    assert "analyze" in stub.workers
    assert stub.analyze_action.enabled_calls == []
    assert stub.progress.hide_calls == 0
    assert stub.status_label.values == []
    assert stub._dbg


def test_on_analysis_error_ignores_stale_error_sequence() -> None:
    stub = _mk_self(current_seq=5, worker_seq=5)

    app_analysis_ops._on_analysis_error(stub, "boom", request_seq=4)

    assert "analyze" in stub.workers
    assert stub.analyze_action.enabled_calls == []
    assert stub.progress.hide_calls == 0
    assert stub.status_label.values == []
    assert stub._logs == []
    assert stub._dbg
