from __future__ import annotations

import pandas as pd

from models.auto_learner_components import ParallelFetcher


class _DummyFetcher:
    def get_history(self, code, **kwargs):  # noqa: ARG002
        return pd.DataFrame({"close": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]})


def test_parallel_fetcher_batch_does_not_require_cancel_token(monkeypatch) -> None:
    fetcher = ParallelFetcher(max_workers=2)
    assert not hasattr(fetcher, "_cancel_token")

    monkeypatch.setattr(
        "models.auto_learner_components.get_fetcher",
        lambda: _DummyFetcher(),
    )

    ok_codes, failed_codes = fetcher.fetch_batch(
        codes=["000001", "000002"],
        interval="1d",
        lookback=5,
        min_bars=5,
        stop_check=lambda: False,
        progress_cb=lambda *_a, **_k: None,
    )

    assert set(ok_codes) == {"000001", "000002"}
    assert failed_codes == []
