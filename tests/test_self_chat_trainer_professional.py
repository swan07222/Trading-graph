from __future__ import annotations

import json
from pathlib import Path

from ai import self_chat_trainer as trainer


def test_iter_message_lists_filters_operational_noise() -> None:
    payload = {
        "messages": [
            {"role": "system", "text": "System initialized - Ready for analysis"},
            {"role": "assistant", "text": "AI message feed ready."},
            {"role": "user", "text": "Analyze 600519 trend and risk."},
            {
                "role": "assistant",
                "text": (
                    "Use trend, volume, and catalyst persistence. "
                    "Define invalidation and position size before entry."
                ),
            },
        ]
    }

    convos = trainer._iter_message_lists(payload)
    assert len(convos) == 1
    convo = convos[0]
    assert len(convo) == 2
    assert convo[0]["role"] == "user"
    assert convo[1]["role"] == "assistant"

    samples = trainer._conversation_to_samples(convo)
    assert len(samples) == 1
    text = samples[0].lower()
    assert "analyze 600519 trend and risk" in text
    assert "ai message feed ready" not in text
    assert "system initialized - ready for analysis" not in text


def test_split_text_samples_returns_disjoint_train_val() -> None:
    texts = [f"sample-{i}-market-context" for i in range(10)]
    train_texts, val_texts = trainer._split_text_samples(
        texts,
        validation_split=0.2,
        seed=42,
    )
    assert train_texts
    assert val_texts
    assert len(train_texts) + len(val_texts) == len(texts)
    assert set(train_texts).isdisjoint(set(val_texts))


def test_train_self_chat_model_normalizes_profile_and_uses_clean_corpus(
    monkeypatch,
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "history.json"
    history = {
        "messages": [
            {"role": "system", "text": "No trained GM model found. Please train GM."},
            {"role": "user", "text": "How should I manage downside risk?"},
            {
                "role": "assistant",
                "text": "Set max loss first, then size by stop distance and liquidity conditions.",
            },
            {"role": "user", "text": "How do I avoid overtrading?"},
            {
                "role": "assistant",
                "text": "Use strict entry filters and a capped daily trade count.",
            },
        ]
    }
    history_path.write_text(json.dumps(history), encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_run(cfg, texts):
        captured["cfg"] = cfg
        captured["texts"] = list(texts)
        return {"status": "trained", "steps": 7}

    monkeypatch.setattr(trainer, "_run_self_training", _fake_run)

    cfg = trainer.SelfChatTrainingConfig(
        output_dir=tmp_path / "out",
        chat_history_path=history_path,
        min_text_samples=1,
        training_profile="invalid-profile",
    )
    out = trainer.train_self_chat_model(cfg)

    assert out.get("status") == "trained"
    assert out.get("training_profile") == "balanced"
    cleaned = "\n".join(str(x) for x in captured.get("texts", []))
    assert "No trained GM model found" not in cleaned


def test_training_backend_auto_uses_professional_pretrained_path() -> None:
    assert trainer._normalize_training_backend("auto", profile="professional") == "pretrained_lora"
    assert trainer._normalize_training_backend("auto", profile="balanced") == "scratch"
    assert trainer._normalize_training_backend("invalid", profile="professional") == "pretrained_lora"


def test_train_self_chat_model_prefers_pretrained_backend_for_professional_profile(
    monkeypatch,
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "history.json"
    history = {
        "messages": [
            {"role": "user", "text": "How should I manage downside risk?"},
            {
                "role": "assistant",
                "text": "Define invalidation level and cap loss before entering any position.",
            },
            {"role": "user", "text": "How do I avoid overtrading in weak liquidity?"},
            {
                "role": "assistant",
                "text": "Use strict entry filters, smaller size, and a capped daily trade count.",
            },
        ]
    }
    history_path.write_text(json.dumps(history), encoding="utf-8")

    calls = {"professional": 0, "scratch": 0}

    def _fake_professional(cfg, texts):
        _ = (cfg, texts)
        calls["professional"] += 1
        return {
            "status": "trained",
            "training_backend": "pretrained_lora",
            "steps": 12,
        }

    def _fake_scratch(cfg, texts):
        _ = (cfg, texts)
        calls["scratch"] += 1
        return {"status": "trained", "training_backend": "scratch", "steps": 3}

    monkeypatch.setattr(trainer, "_run_professional_training", _fake_professional)
    monkeypatch.setattr(trainer, "_run_self_training", _fake_scratch)

    cfg = trainer.SelfChatTrainingConfig(
        output_dir=tmp_path / "out",
        chat_history_path=history_path,
        min_text_samples=1,
        training_profile="professional",
        training_backend="auto",
    )
    out = trainer.train_self_chat_model(cfg)

    assert out.get("status") == "trained"
    assert out.get("training_backend") == "pretrained_lora"
    assert int(calls["professional"]) == 1
    assert int(calls["scratch"]) == 0


def test_train_self_chat_model_falls_back_to_scratch_on_professional_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "history.json"
    history = {
        "messages": [
            {"role": "user", "text": "Build a trading plan for volatile markets."},
            {
                "role": "assistant",
                "text": (
                    "Control drawdown first, then scale entries by liquidity "
                    "and catalyst durability."
                ),
            },
            {"role": "user", "text": "How should position sizing adapt after a drawdown?"},
            {
                "role": "assistant",
                "text": "Reduce gross exposure first, then scale up only after process stability returns.",
            },
        ]
    }
    history_path.write_text(json.dumps(history), encoding="utf-8")

    def _fake_professional(cfg, texts):
        _ = (cfg, texts)
        return {
            "status": "error",
            "message": "peft missing",
            "error": "ImportError: peft",
        }

    def _fake_scratch(cfg, texts):
        _ = (cfg, texts)
        return {"status": "trained", "training_backend": "scratch", "steps": 5}

    monkeypatch.setattr(trainer, "_run_professional_training", _fake_professional)
    monkeypatch.setattr(trainer, "_run_self_training", _fake_scratch)

    cfg = trainer.SelfChatTrainingConfig(
        output_dir=tmp_path / "out",
        chat_history_path=history_path,
        min_text_samples=1,
        training_profile="professional",
        training_backend="pretrained_lora",
        allow_scratch_fallback=True,
    )
    out = trainer.train_self_chat_model(cfg)

    assert out.get("status") == "trained"
    assert out.get("training_backend") == "scratch"
    assert out.get("fallback_from_backend") == "pretrained_lora"
    assert "fallback_reason" in out
