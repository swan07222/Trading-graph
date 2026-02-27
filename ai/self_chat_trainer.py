"""Self-trained transformer chat model trainer.

This module trains a local causal Transformer model from your own corpus.
No external inference service is required.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SelfChatTrainingConfig:
    """Training configuration for self chat model."""

    output_dir: Path
    chat_history_path: Path
    training_corpus_path: Path | None = None
    min_text_samples: int = 32
    vocab_size: int = 16000
    context_length: int = 512
    hidden_size: int = 384
    num_layers: int = 6
    num_heads: int = 6
    dropout: float = 0.1
    batch_size: int = 4
    epochs: int = 2
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    grad_accum_steps: int = 8
    max_steps: int = 1200
    seed: int = 42
    save_every_steps: int = 0

    @classmethod
    def from_defaults(cls) -> "SelfChatTrainingConfig":
        llm_dir = Path(getattr(CONFIG, "llm_model_dir", CONFIG.model_dir))
        return cls(
            output_dir=llm_dir / "chat_transformer",
            chat_history_path=Path("data/chat_history/chat_history.json"),
            training_corpus_path=llm_dir / "llm_training_corpus.jsonl",
        )


class _PackedTokenDataset:
    """Packed token blocks for causal LM training."""

    def __init__(self, token_ids: list[int], block_size: int) -> None:
        self.block_size = max(32, int(block_size))
        self.blocks: list[list[int]] = []
        for i in range(0, len(token_ids) - self.block_size, self.block_size):
            block = token_ids[i:i + self.block_size]
            if len(block) == self.block_size:
                self.blocks.append(block)

    def __len__(self) -> int:
        return int(len(self.blocks))

    def __getitem__(self, idx: int) -> list[int]:
        return list(self.blocks[idx])


def _safe_read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("Failed to read JSON %s: %s", path, exc)
        return None


def _iter_message_lists(payload: Any) -> list[list[dict[str, str]]]:
    out: list[list[dict[str, str]]] = []
    if payload is None:
        return out

    def _walk(node: Any) -> None:
        if isinstance(node, list):
            if node and all(isinstance(x, dict) for x in node):
                if all(("role" in x or "sender" in x) for x in node):
                    convo: list[dict[str, str]] = []
                    for row in node:
                        role = str(row.get("role", row.get("sender", "user"))).strip().lower()
                        text = str(row.get("content", row.get("text", ""))).strip()
                        if not text:
                            continue
                        if role in {"you"}:
                            role = "user"
                        elif role in {"ai"}:
                            role = "assistant"
                        if role not in {"system", "user", "assistant"}:
                            role = "user"
                        convo.append({"role": role, "content": text})
                    if convo:
                        out.append(convo)
                    return
            for row in node:
                _walk(row)
            return

        if isinstance(node, dict):
            for key in ("messages", "history", "conversation", "conversations", "turns"):
                if key in node:
                    _walk(node.get(key))
            # Handle single message object.
            if "role" in node and ("content" in node or "text" in node):
                role = str(node.get("role", "user")).strip().lower()
                text = str(node.get("content", node.get("text", ""))).strip()
                if text:
                    out.append([{"role": role, "content": text}])

    _walk(payload)
    return out


def _conversation_to_samples(messages: list[dict[str, str]]) -> list[str]:
    samples: list[str] = []
    window: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "user")).strip().lower()
        text = str(msg.get("content", "")).strip()
        if not text:
            continue
        if role not in {"system", "user", "assistant"}:
            role = "user"
        window.append({"role": role, "content": text})
        if role == "assistant":
            tail = window[-8:]
            rows = [f"<|{r['role']}|>\n{r['content']}" for r in tail]
            rows.append("<|assistant|>")
            samples.append("\n".join(rows))
    return samples


def _load_corpus_texts(cfg: SelfChatTrainingConfig) -> list[str]:
    texts: list[str] = []

    payload = _safe_read_json(cfg.chat_history_path)
    for convo in _iter_message_lists(payload):
        texts.extend(_conversation_to_samples(convo))

    corpus_path = cfg.training_corpus_path
    if corpus_path is not None and corpus_path.exists():
        try:
            with corpus_path.open("r", encoding="utf-8") as f:
                for line in f:
                    raw = str(line or "").strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except Exception:
                        continue
                    title = str(row.get("title", "") or "").strip()
                    content = str(row.get("content", "") or "").strip()
                    if not (title or content):
                        continue
                    texts.append(
                        "<|system|>\nYou are a professional trading assistant.\n"
                        "<|user|>\nPlease analyze this market text.\n"
                        f"<|assistant|>\nTitle: {title}\n{content[:1800]}"
                    )
        except Exception as exc:
            log.warning("Failed to read training corpus %s: %s", corpus_path, exc)

    # Minimal bootstrap to avoid empty training.
    if not texts:
        texts.extend([
            "<|system|>\nYou are a professional stock analysis assistant.\n"
            "<|user|>\nWhat are key risks for a stock rally?\n"
            "<|assistant|>\nI evaluate liquidity, policy, valuation, and sentiment risks.",
            "<|system|>\nYou are a bilingual market analyst.\n"
            "<|user|>\n请给我一个稳健的交易分析框架。\n"
            "<|assistant|>\n先看趋势，再看成交量与政策，再设定止损与仓位上限。",
        ])

    # Normalize and dedupe.
    clean: list[str] = []
    seen: set[str] = set()
    for txt in texts:
        t = str(txt or "").strip()
        if len(t) < 16:
            continue
        if t in seen:
            continue
        seen.add(t)
        clean.append(t)
    return clean


def _run_self_training(cfg: SelfChatTrainingConfig, texts: list[str]) -> dict[str, Any]:
    try:
        import torch
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    except Exception as exc:
        return {
            "status": "error",
            "message": "Self training requires torch + transformers + tokenizers",
            "error": str(exc),
        }

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = [
        "<pad>",
        "<unk>",
        "<s>",
        "</s>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
    ]

    tokenizer_core = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer_core.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer_core.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=max(4096, int(cfg.vocab_size)),
        min_frequency=2,
        special_tokens=special_tokens,
    )
    tokenizer_core.train_from_iterator(texts, trainer=trainer)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_core,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=["<|system|>", "<|user|>", "<|assistant|>"],
    )

    eos_id = int(tokenizer.eos_token_id or 0)
    all_ids: list[int] = []
    for txt in texts:
        ids = tokenizer.encode(str(txt), add_special_tokens=True)
        if not ids:
            continue
        all_ids.extend(int(x) for x in ids)
        all_ids.append(eos_id)

    dataset = _PackedTokenDataset(all_ids, cfg.context_length)
    if len(dataset) < 2:
        return {
            "status": "error",
            "message": "Not enough packed samples for training",
            "packed_samples": len(dataset),
        }

    model_cfg = GPT2Config(
        vocab_size=int(tokenizer.vocab_size),
        n_positions=int(cfg.context_length),
        n_ctx=int(cfg.context_length),
        n_embd=int(cfg.hidden_size),
        n_layer=int(cfg.num_layers),
        n_head=int(cfg.num_heads),
        resid_pdrop=float(cfg.dropout),
        embd_pdrop=float(cfg.dropout),
        attn_pdrop=float(cfg.dropout),
        bos_token_id=int(tokenizer.bos_token_id or eos_id),
        eos_token_id=eos_id,
        pad_token_id=int(tokenizer.pad_token_id or eos_id),
    )
    model = GPT2LMHeadModel(model_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    steps = 0
    loss_sum = 0.0
    started = time.time()

    for epoch in range(max(1, int(cfg.epochs))):
        for batch in data_loader:
            input_ids = batch.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss / max(1, int(cfg.grad_accum_steps))

            scaler.scale(loss).backward()

            if (steps + 1) % max(1, int(cfg.grad_accum_steps)) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            loss_sum += float(loss.item()) * max(1, int(cfg.grad_accum_steps))
            steps += 1

            if cfg.max_steps > 0 and steps >= int(cfg.max_steps):
                break
        if cfg.max_steps > 0 and steps >= int(cfg.max_steps):
            break

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    avg_loss = float(loss_sum / max(1, steps))
    duration = float(time.time() - started)
    report = {
        "status": "trained",
        "model_dir": str(cfg.output_dir),
        "samples": int(len(texts)),
        "packed_samples": int(len(dataset)),
        "steps": int(steps),
        "epochs": int(cfg.epochs),
        "avg_loss": avg_loss,
        "duration_seconds": duration,
        "architecture": "self_trained_transformer",
        "backend": "transformers_local",
    }
    try:
        with (cfg.output_dir / "training_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return report


def train_self_chat_model(
    config: SelfChatTrainingConfig | None = None,
) -> dict[str, Any]:
    """Train a self-owned chat transformer model from local data."""
    cfg = config or SelfChatTrainingConfig.from_defaults()
    texts = _load_corpus_texts(cfg)
    if len(texts) < 2:
        return {
            "status": "error",
            "message": (
                f"Insufficient corpus size ({len(texts)}). "
                "Provide more chat history or local training corpus."
            ),
            "samples": len(texts),
            "chat_history_path": str(cfg.chat_history_path),
            "training_corpus_path": str(cfg.training_corpus_path) if cfg.training_corpus_path else "",
        }
    report = dict(_run_self_training(cfg, texts) or {})
    if len(texts) < int(cfg.min_text_samples):
        report["warning"] = (
            f"Low corpus size ({len(texts)} < {cfg.min_text_samples}); "
            "quality may be limited."
        )
    return report
