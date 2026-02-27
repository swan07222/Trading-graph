"""Local chat model trainer with scratch and professional fine-tuning backends.

Scratch backend:
- trains a compact GPT-style model from local corpus only.

Professional backend:
- fine-tunes a pretrained causal model with LoRA adapters and merges to a
  standalone local artifact for offline inference.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)

_NOISE_SUBSTRINGS = (
    "ai message feed ready",
    "system initialized - ready for analysis",
    "no trained gm model found",
    "gm training completed successfully",
    "gm model loaded successfully",
    "subscribed to feeds for",
    "monitoring started:",
    "auto train llm panel opened",
    "starting llm training",
    "starting optimized llm training",
    "llm training complete:",
)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _is_operational_noise(role: str, text: str) -> bool:
    clean = _normalize_text(text).lower()
    if not clean:
        return True
    if len(clean) < 3:
        return True
    if any(token in clean for token in _NOISE_SUBSTRINGS):
        return True
    if role == "system" and (
        clean.startswith("starting ")
        or clean.startswith("monitoring ")
        or clean.startswith("system ")
    ):
        return True
    return False


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
    save_every_steps: int = 200
    validation_split: float = 0.1
    training_profile: str = "balanced"  # fast, balanced, professional
    training_backend: str = "auto"  # auto, scratch, pretrained_lora
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    allow_scratch_fallback: bool = True
    local_files_only: bool = False
    trust_remote_code: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

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
            block = token_ids[i : i + self.block_size]
            if len(block) == self.block_size:
                self.blocks.append(block)

    def __len__(self) -> int:
        return int(len(self.blocks))

    def __getitem__(self, idx: int) -> list[int]:
        return list(self.blocks[idx])


class _TokenizedTextDataset:
    """Simple tokenized text dataset for supervised causal LM fine-tuning."""

    def __init__(self, tokenizer: Any, texts: list[str], max_length: int) -> None:
        self.rows: list[dict[str, list[int]]] = []
        max_len = max(128, int(max_length))
        for text in list(texts or []):
            raw = str(text or "").strip()
            if len(raw) < 8:
                continue
            encoded = tokenizer(
                raw,
                truncation=True,
                max_length=max_len,
                padding=False,
                return_attention_mask=True,
            )
            input_ids = [int(x) for x in list(encoded.get("input_ids") or [])]
            attention_mask = [int(x) for x in list(encoded.get("attention_mask") or [])]
            if len(input_ids) < 4:
                continue
            self.rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": list(input_ids),
                }
            )

    def __len__(self) -> int:
        return int(len(self.rows))

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return dict(self.rows[idx])


def _normalize_training_profile(value: str) -> str:
    profile = str(value or "balanced").strip().lower()
    if profile not in {"fast", "balanced", "professional"}:
        return "balanced"
    return profile


def _normalize_training_backend(value: str, *, profile: str) -> str:
    backend = str(value or "auto").strip().lower()
    if backend not in {"auto", "scratch", "pretrained_lora"}:
        backend = "auto"
    if backend == "auto":
        return "pretrained_lora" if profile == "professional" else "scratch"
    return backend


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
                        text = _normalize_text(str(row.get("content", row.get("text", ""))))
                        if not text:
                            continue
                        if role in {"you"}:
                            role = "user"
                        elif role in {"ai"}:
                            role = "assistant"
                        elif role in {"system", "sys"}:
                            role = "system"
                        if role not in {"system", "user", "assistant"}:
                            role = "user"
                        if _is_operational_noise(role, text):
                            continue
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
            if "role" in node and ("content" in node or "text" in node):
                role = str(node.get("role", "user")).strip().lower()
                text = _normalize_text(str(node.get("content", node.get("text", ""))))
                if text:
                    if role in {"you"}:
                        role = "user"
                    elif role in {"ai"}:
                        role = "assistant"
                    elif role in {"system", "sys"}:
                        role = "system"
                    if role not in {"system", "user", "assistant"}:
                        role = "user"
                    if _is_operational_noise(role, text):
                        return
                    out.append([{"role": role, "content": text}])

    _walk(payload)
    return out


def _conversation_to_samples(messages: list[dict[str, str]]) -> list[str]:
    samples: list[str] = []
    window: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "user")).strip().lower()
        text = _normalize_text(str(msg.get("content", "")))
        if not text:
            continue
        if role not in {"system", "user", "assistant"}:
            role = "user"
        if _is_operational_noise(role, text):
            continue
        window.append({"role": role, "content": text})
        if role == "assistant":
            if len(text) < 16:
                continue
            tail = window[-8:]
            if not any(r.get("role") == "user" for r in tail):
                continue
            rows = [f"<|{r['role']}|>\n{r['content']}" for r in tail]
            samples.append("\n".join(rows))
    return samples


def _bootstrap_seed_samples() -> list[str]:
    """Fallback seed corpus when no local history is available."""
    pairs: list[tuple[str, str]] = [
        (
            "How should I analyze a sudden breakout?",
            "Check volume quality, catalyst durability, valuation, and invalidation levels.",
        ),
        (
            "What are key risks for a rally?",
            "Liquidity shocks, policy reversal, earnings miss, and crowding risk.",
        ),
        (
            "How do I size a position conservatively?",
            "Define max loss first, then size by stop distance and portfolio risk budget.",
        ),
        (
            "How should I interpret weak sentiment but strong price?",
            "Treat it as divergence: reduce conviction, shorten horizon, and wait for confirmation.",
        ),
        (
            "Give me a checklist before entering a trade.",
            "Trend alignment, catalyst, liquidity, risk-reward, stop level, and exit plan.",
        ),
        (
            "How do I avoid overtrading?",
            "Use strict filters, capped daily trades, and a pre-trade written thesis.",
        ),
        (
            "How to react after a bad loss day?",
            "Cut size, pause discretionary trades, review mistakes, and reset rules.",
        ),
        (
            "What should I do when data is missing?",
            "State the gap, avoid guessing, and propose the next best verifiable step.",
        ),
        (
            "Provide a disciplined market analysis framework.",
            "Start with trend and liquidity, validate catalyst and valuation, then define stop-loss and size limits.",
        ),
        (
            "How can I tell if positive news is sustainable?",
            "Check policy follow-through, persistent capital flows, and confirmation in fundamentals.",
        ),
        (
            "What should I do first in a sharp market drawdown?",
            "Control risk and liquidity first, then reassess correlation and reduce exposures in tiers.",
        ),
        (
            "How do I place a rational stop loss?",
            "Anchor stops to invalidation levels, not emotions, and align with total portfolio risk.",
        ),
    ]
    out: list[str] = []
    for user_text, assistant_text in pairs:
        out.append(
            "<|system|>\nYou are a disciplined market assistant.\n"
            f"<|user|>\n{user_text}\n"
            f"<|assistant|>\n{assistant_text}"
        )
    return out

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

    if not texts:
        texts.extend(_bootstrap_seed_samples())

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


def _augment_low_resource_texts(
    texts: list[str],
    *,
    target_count: int,
    seed: int,
) -> list[str]:
    """Expand tiny corpora with conservative templated variations."""
    base = [str(t or "").strip() for t in list(texts or []) if str(t or "").strip()]
    if not base:
        return []

    seen: set[str] = set()
    out: list[str] = []
    for txt in base:
        if txt in seen:
            continue
        seen.add(txt)
        out.append(txt)

    goal = max(len(out), int(target_count))
    if len(out) >= goal:
        return out

    rng = random.Random(int(seed) + 101)
    templates = [
        (
            "{base}\n<|assistant|>\n"
            "Key checks: trend, catalyst quality, liquidity, and invalidation risk."
        ),
        "<|system|>\nYou are a prudent trading assistant.\n{base}",
        (
            "{base}\n<|user|>\nSummarize the risks and execution constraints.\n"
            "<|assistant|>\nFocus on liquidity, policy sensitivity, earnings durability, and stop-loss discipline."
        ),
    ]

    cursor = 0
    max_attempts = max(64, goal * 12)
    while len(out) < goal and cursor < max_attempts:
        src = out[cursor % len(out)]
        snippet = src
        if len(snippet) > 900:
            start_idx = rng.randint(0, max(0, len(snippet) - 700))
            snippet = snippet[start_idx : start_idx + 700]
        tpl = templates[cursor % len(templates)]
        candidate = str(tpl.format(base=snippet)).strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
        cursor += 1

    return out[:goal]


def _split_text_samples(
    texts: list[str],
    *,
    validation_split: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    clean = [str(t or "").strip() for t in list(texts or []) if str(t or "").strip()]
    if len(clean) < 2 or validation_split <= 0.0:
        return clean, []

    idx = list(range(len(clean)))
    rng = random.Random(int(seed) + 17)
    rng.shuffle(idx)

    val_frac = float(max(0.0, min(0.4, validation_split)))
    val_count = int(round(len(clean) * val_frac))
    val_count = max(1, min(val_count, len(clean) - 1))
    val_set = set(idx[:val_count])

    train_texts = [clean[i] for i in range(len(clean)) if i not in val_set]
    val_texts = [clean[i] for i in range(len(clean)) if i in val_set]
    if not train_texts:
        return clean, []
    return train_texts, val_texts


def _tokenize_texts(
    tokenizer: Any,
    texts: list[str],
    *,
    eos_id: int,
) -> list[int]:
    out: list[int] = []
    for txt in texts:
        ids = tokenizer.encode(str(txt), add_special_tokens=True)
        if not ids:
            continue
        out.extend(int(x) for x in ids)
        out.append(int(eos_id))
    return out


def _infer_lora_target_modules(model: Any) -> list[str] | None:
    """Infer common attention module names for LoRA injection."""
    candidates = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",
        "c_proj",
    }
    found: set[str] = set()
    try:
        for name, _module in model.named_modules():
            tail = str(name or "").split(".")[-1].strip()
            if tail in candidates:
                found.add(tail)
    except Exception:
        return None
    return sorted(found) if found else None


def _run_professional_training(cfg: SelfChatTrainingConfig, texts: list[str]) -> dict[str, Any]:
    """Fine-tune a pretrained causal model with LoRA and save merged artifact."""
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        return {
            "status": "error",
            "message": (
                "Professional training requires torch + transformers + accelerate "
                "(and optional sentencepiece for some tokenizers)."
            ),
            "error": str(exc),
            "training_backend": "pretrained_lora",
        }

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:
        return {
            "status": "error",
            "message": "Professional training requires peft (LoRA).",
            "error": str(exc),
            "training_backend": "pretrained_lora",
        }

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    profile = _normalize_training_profile(cfg.training_profile)
    sample_count = int(len(texts))
    requested_val_frac = float(max(0.0, min(0.4, float(cfg.validation_split))))

    base_model = str(cfg.base_model_name or "").strip()
    if not base_model:
        base_model = "Qwen/Qwen2.5-1.5B-Instruct"

    effective_context = max(256, int(cfg.context_length))
    effective_batch = max(1, int(cfg.batch_size))
    effective_epochs = max(1, int(cfg.epochs))
    effective_grad_accum = max(1, int(cfg.grad_accum_steps))
    effective_lr = float(cfg.learning_rate)
    adaptive_profile = f"{profile}_default"

    if profile == "fast":
        effective_context = min(effective_context, 512)
        effective_batch = min(effective_batch, 2)
        effective_epochs = min(effective_epochs, 2)
        effective_grad_accum = min(effective_grad_accum, 4)
        effective_lr = min(effective_lr, 2e-4)
    elif profile == "professional":
        effective_context = max(effective_context, 1024)
        effective_batch = min(max(1, effective_batch), 2)
        effective_epochs = max(effective_epochs, 3)
        effective_grad_accum = max(effective_grad_accum, 8)
        effective_lr = min(effective_lr, 1.5e-4)
    else:
        effective_context = min(max(512, effective_context), 1024)
        effective_batch = min(max(1, effective_batch), 2)
        effective_epochs = max(2, effective_epochs)
        effective_grad_accum = max(4, effective_grad_accum)
        effective_lr = min(effective_lr, 1.8e-4)

    if sample_count < 48:
        adaptive_profile = f"{profile}_low_data_tiny"
        effective_context = min(effective_context, 512)
        effective_batch = 1
        effective_epochs = max(effective_epochs, 3)
        effective_grad_accum = min(effective_grad_accum, 4)
        effective_lr = min(effective_lr, 1e-4)
    elif sample_count < 120:
        adaptive_profile = f"{profile}_low_data_small"
        effective_context = min(effective_context, 768)
        effective_batch = min(effective_batch, 2)
        effective_epochs = max(effective_epochs, 2)
        effective_grad_accum = min(effective_grad_accum, 6)
        effective_lr = min(effective_lr, 1.2e-4)

    train_texts, val_texts = _split_text_samples(
        texts,
        validation_split=requested_val_frac,
        seed=int(cfg.seed),
    )
    if len(train_texts) < 2:
        return {
            "status": "error",
            "message": "Not enough train samples for professional fine-tuning",
            "samples": int(len(texts)),
            "train_text_samples": int(len(train_texts)),
            "val_text_samples": int(len(val_texts)),
            "training_backend": "pretrained_lora",
            "training_profile": profile,
        }

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=bool(cfg.trust_remote_code),
            local_files_only=bool(cfg.local_files_only),
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": bool(cfg.trust_remote_code),
            "local_files_only": bool(cfg.local_files_only),
        }
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    except Exception as exc:
        return {
            "status": "error",
            "message": (
                "Failed to load pretrained base model. "
                "Set --base-model to an available local or Hub model."
            ),
            "error": str(exc),
            "base_model_name": base_model,
            "training_backend": "pretrained_lora",
            "training_profile": profile,
        }

    target_modules = _infer_lora_target_modules(model)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=max(4, int(cfg.lora_rank)),
        lora_alpha=max(8, int(cfg.lora_alpha)),
        lora_dropout=float(max(0.0, min(0.5, cfg.lora_dropout))),
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    if profile == "professional" and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    trainable_parameters = int(
        sum(p.numel() for p in model.parameters() if bool(getattr(p, "requires_grad", False)))
    )
    total_parameters = int(sum(p.numel() for p in model.parameters()))

    train_dataset = _TokenizedTextDataset(tokenizer, train_texts, max_length=effective_context)
    if len(train_dataset) < 1:
        return {
            "status": "error",
            "message": "Not enough tokenized train samples for professional fine-tuning",
            "samples": int(len(texts)),
            "training_backend": "pretrained_lora",
            "training_profile": profile,
        }

    val_dataset_obj = _TokenizedTextDataset(tokenizer, val_texts, max_length=effective_context)
    val_dataset = val_dataset_obj if len(val_dataset_obj) > 0 else None
    eval_enabled = val_dataset is not None

    updates_per_epoch = max(
        1,
        int(math.ceil(len(train_dataset) / float(max(1, effective_batch * effective_grad_accum)))),
    )
    planned_updates = max(1, updates_per_epoch * max(1, effective_epochs))
    if int(cfg.max_steps) > 0:
        max_updates = max(20, min(int(cfg.max_steps), max(30, planned_updates * 3)))
    else:
        max_updates = max(30, planned_updates)

    run_dir = cfg.output_dir / "trainer_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        args = TrainingArguments(
            output_dir=str(run_dir),
            overwrite_output_dir=True,
            num_train_epochs=float(effective_epochs),
            max_steps=int(max_updates),
            per_device_train_batch_size=int(effective_batch),
            per_device_eval_batch_size=int(effective_batch),
            gradient_accumulation_steps=int(effective_grad_accum),
            learning_rate=float(effective_lr),
            weight_decay=float(cfg.weight_decay),
            lr_scheduler_type="cosine",
            warmup_ratio=(0.06 if profile == "professional" else 0.03),
            logging_steps=max(5, min(50, int(cfg.save_every_steps // 2))),
            save_strategy="steps",
            save_steps=max(20, int(cfg.save_every_steps)),
            save_total_limit=2,
            evaluation_strategy=("steps" if eval_enabled else "no"),
            eval_steps=max(20, int(cfg.save_every_steps)),
            load_best_model_at_end=bool(eval_enabled),
            metric_for_best_model=("eval_loss" if eval_enabled else None),
            greater_is_better=False if eval_enabled else None,
            fp16=bool(torch.cuda.is_available()),
            bf16=False,
            report_to=[],
            dataloader_num_workers=0,
            remove_unused_columns=False,
            seed=int(cfg.seed),
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=(val_dataset if eval_enabled else None),
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
        train_result = trainer.train()
        train_metrics = dict(getattr(train_result, "metrics", {}) or {})
        eval_metrics = dict(trainer.evaluate() if eval_enabled else {})
    except Exception as exc:
        return {
            "status": "error",
            "message": "Professional fine-tuning failed during Trainer run.",
            "error": str(exc),
            "training_backend": "pretrained_lora",
            "training_profile": profile,
            "base_model_name": base_model,
        }

    merge_status = "skipped"
    final_model = trainer.model
    if hasattr(final_model, "merge_and_unload"):
        try:
            final_model = final_model.merge_and_unload()
            merge_status = "merged"
        except Exception as exc:
            merge_status = f"merge_failed: {exc}"

    final_model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    duration = float(time.time() - started)
    train_loss = float(train_metrics.get("train_loss", 0.0) or 0.0)
    eval_loss_raw = eval_metrics.get("eval_loss")
    eval_loss = float(eval_loss_raw) if eval_loss_raw is not None else None
    val_perplexity = (
        float(math.exp(min(20.0, eval_loss)))
        if eval_loss is not None and math.isfinite(eval_loss)
        else None
    )

    report = {
        "status": "trained",
        "model_dir": str(cfg.output_dir),
        "samples": int(len(texts)),
        "train_text_samples": int(len(train_texts)),
        "val_text_samples": int(len(val_texts)),
        "tokenized_train_samples": int(len(train_dataset)),
        "tokenized_val_samples": int(len(val_dataset) if val_dataset is not None else 0),
        "steps": int(train_metrics.get("global_step", max_updates)),
        "optimizer_steps": int(train_metrics.get("global_step", max_updates)),
        "epochs": int(effective_epochs),
        "epochs_requested": int(cfg.epochs),
        "avg_loss": train_loss,
        "train_loss_last_epoch": train_loss,
        "val_loss": eval_loss,
        "best_val_loss": eval_loss,
        "val_perplexity": val_perplexity,
        "save_every_steps": int(cfg.save_every_steps),
        "checkpoint_count": 0,
        "duration_seconds": duration,
        "architecture": "pretrained_lora_causal_lm",
        "backend": "transformers_local",
        "training_backend": "pretrained_lora",
        "base_model_name": base_model,
        "lora_merge_status": str(merge_status),
        "device": ("cuda" if torch.cuda.is_available() else "cpu"),
        "model_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "training_profile": str(profile),
        "early_stopped": False,
        "early_stop_patience": None,
        "adaptive_profile": str(adaptive_profile),
        "effective_config": {
            "context_length": int(effective_context),
            "batch_size": int(effective_batch),
            "grad_accum_steps": int(effective_grad_accum),
            "learning_rate": float(effective_lr),
            "max_updates": int(max_updates),
            "lora_rank": int(max(4, int(cfg.lora_rank))),
            "lora_alpha": int(max(8, int(cfg.lora_alpha))),
            "lora_dropout": float(max(0.0, min(0.5, cfg.lora_dropout))),
            "target_modules": list(target_modules or []),
        },
        "trainer_metrics": {
            "train": dict(train_metrics),
            "eval": dict(eval_metrics),
        },
    }

    try:
        with (cfg.output_dir / "model_card.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "architecture": "pretrained_lora_causal_lm",
                    "training_profile": str(profile),
                    "training_backend": "pretrained_lora",
                    "base_model_name": str(base_model),
                    "lora_merge_status": str(merge_status),
                    "samples": {
                        "total_text_samples": int(len(texts)),
                        "train_text_samples": int(len(train_texts)),
                        "val_text_samples": int(len(val_texts)),
                        "tokenized_train_samples": int(len(train_dataset)),
                        "tokenized_val_samples": int(
                            len(val_dataset) if val_dataset is not None else 0
                        ),
                    },
                    "parameters": {
                        "total": int(total_parameters),
                        "trainable": int(trainable_parameters),
                    },
                    "effective_config": dict(report.get("effective_config") or {}),
                    "metrics": {
                        "avg_loss": float(train_loss),
                        "val_loss": float(eval_loss) if eval_loss is not None else None,
                        "val_perplexity": float(val_perplexity)
                        if val_perplexity is not None
                        else None,
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass
    try:
        with (cfg.output_dir / "training_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return report


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
    sample_count = int(len(texts))
    requested_val_frac = float(max(0.0, min(0.4, float(cfg.validation_split))))

    effective_context = max(128, int(cfg.context_length))
    effective_hidden = max(128, int(cfg.hidden_size))
    effective_layers = max(2, int(cfg.num_layers))
    effective_heads = max(2, int(cfg.num_heads))
    effective_batch = max(1, int(cfg.batch_size))
    effective_epochs = max(1, int(cfg.epochs))
    effective_grad_accum = max(1, int(cfg.grad_accum_steps))
    effective_lr = float(cfg.learning_rate)
    profile = _normalize_training_profile(cfg.training_profile)
    adaptive_profile = f"{profile}_default"

    if profile == "fast":
        effective_context = min(effective_context, 384)
        effective_hidden = min(effective_hidden, 256)
        effective_layers = min(effective_layers, 4)
        effective_heads = min(effective_heads, 4)
        effective_batch = min(effective_batch, 2)
        effective_grad_accum = min(effective_grad_accum, 4)
        effective_lr = min(effective_lr, 5e-4)
    elif profile == "professional":
        effective_context = max(effective_context, 768)
        effective_hidden = max(effective_hidden, 512)
        effective_layers = max(effective_layers, 8)
        effective_heads = max(effective_heads, 8)
        effective_batch = min(max(1, effective_batch), 2)
        effective_epochs = max(effective_epochs, 4)
        effective_grad_accum = max(effective_grad_accum, 8)
        effective_lr = min(effective_lr, 4e-4)

    if sample_count < 48:
        adaptive_profile = f"{profile}_low_data_tiny"
        effective_context = min(effective_context, 256)
        effective_hidden = min(effective_hidden, 192)
        effective_layers = min(effective_layers, 3)
        effective_heads = min(effective_heads, 3)
        effective_batch = 1
        effective_grad_accum = min(effective_grad_accum, 2)
        effective_epochs = max(effective_epochs, 3)
        effective_lr = min(effective_lr, 3e-4)
    elif sample_count < 120:
        adaptive_profile = f"{profile}_low_data_small"
        effective_context = min(effective_context, 384)
        effective_hidden = min(effective_hidden, 256)
        effective_layers = min(effective_layers, 4)
        effective_heads = min(effective_heads, 4)
        effective_batch = min(effective_batch, 2)
        effective_grad_accum = min(effective_grad_accum, 4)
        effective_epochs = max(effective_epochs, 2)
        effective_lr = min(effective_lr, 4e-4)

    effective_hidden = max(effective_heads * 16, int(effective_hidden))
    if effective_hidden % effective_heads != 0:
        effective_hidden = int(math.ceil(effective_hidden / float(effective_heads)) * effective_heads)

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
    train_texts, val_texts = _split_text_samples(
        texts,
        validation_split=requested_val_frac,
        seed=int(cfg.seed),
    )
    tokenizer_core.train_from_iterator(train_texts, trainer=trainer)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_core,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=["<|system|>", "<|user|>", "<|assistant|>"],
    )

    eos_id = int(tokenizer.eos_token_id or 0)
    train_ids = _tokenize_texts(tokenizer, train_texts, eos_id=eos_id)
    val_ids = _tokenize_texts(tokenizer, val_texts, eos_id=eos_id)

    train_dataset = _PackedTokenDataset(train_ids, effective_context)
    if len(train_dataset) < 1:
        return {
            "status": "error",
            "message": "Not enough packed samples for training set",
            "packed_samples": len(train_dataset),
            "adaptive_profile": adaptive_profile,
            "training_profile": profile,
        }

    val_dataset_obj = _PackedTokenDataset(val_ids, effective_context) if val_ids else None
    val_dataset = val_dataset_obj if val_dataset_obj is not None and len(val_dataset_obj) > 0 else None

    model_cfg = GPT2Config(
        vocab_size=int(tokenizer.vocab_size),
        n_positions=int(effective_context),
        n_ctx=int(effective_context),
        n_embd=int(effective_hidden),
        n_layer=int(effective_layers),
        n_head=int(effective_heads),
        resid_pdrop=float(cfg.dropout),
        embd_pdrop=float(cfg.dropout),
        attn_pdrop=float(cfg.dropout),
        bos_token_id=int(tokenizer.bos_token_id or eos_id),
        eos_token_id=eos_id,
        pad_token_id=int(tokenizer.pad_token_id or eos_id),
    )
    model = GPT2LMHeadModel(model_cfg)
    total_parameters = int(sum(p.numel() for p in model.parameters()))
    trainable_parameters = int(
        sum(p.numel() for p in model.parameters() if bool(getattr(p, "requires_grad", False)))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=effective_batch,
        shuffle=True,
        drop_last=False,
    )
    val_loader = (
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=effective_batch,
            shuffle=False,
            drop_last=False,
        )
        if val_dataset is not None
        else None
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=effective_lr,
        weight_decay=float(cfg.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    updates_per_epoch = max(
        1,
        int(math.ceil(len(data_loader) / float(max(1, effective_grad_accum)))),
    )
    planned_updates = max(1, updates_per_epoch * max(1, effective_epochs))
    if int(cfg.max_steps) > 0:
        max_updates = max(10, min(int(cfg.max_steps), max(20, planned_updates * 3)))
    else:
        max_updates = max(20, planned_updates)

    warmup_steps = max(1, min(80, max_updates // 10))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_updates - warmup_steps))
        progress = max(0.0, min(1.0, progress))
        return max(0.10, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    steps = 0
    loss_sum = 0.0
    total_batches = 0
    started = time.time()
    epoch_train_losses: list[float] = []
    epoch_val_losses: list[float] = []
    checkpoint_count = 0
    checkpoints_dir = cfg.output_dir / "checkpoints"

    best_val_loss: float | None = None
    best_state: dict[str, Any] | None = None
    no_improve_epochs = 0
    early_stop_patience = 2 if len(train_dataset) >= 24 else 1
    early_stopped = False
    actual_epochs = 0

    def _evaluate(loader: Any) -> float | None:
        if loader is None:
            return None
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch.to(device=device, dtype=torch.long)
                outputs = model(input_ids=input_ids, labels=input_ids)
                losses.append(float(outputs.loss.item()))
        model.train()
        if not losses:
            return None
        return float(sum(losses) / max(1, len(losses)))

    def _save_checkpoint(step: int, epoch: int) -> None:
        nonlocal checkpoint_count
        try:
            ckpt_dir = checkpoints_dir / f"step_{step:05d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            with (ckpt_dir / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "step": int(step),
                        "epoch": int(epoch),
                        "saved_at": time.time(),
                        "adaptive_profile": adaptive_profile,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            checkpoint_count += 1
        except Exception as exc:
            log.debug("Checkpoint save failed at step %s: %s", step, exc)

    for epoch in range(max(1, effective_epochs)):
        epoch_loss_sum = 0.0
        epoch_batches = 0
        accum_counter = 0
        optimizer.zero_grad(set_to_none=True)

        for batch in data_loader:
            input_ids = batch.to(device=device, dtype=torch.long)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss / float(max(1, effective_grad_accum))

            scaler.scale(loss).backward()
            accum_counter += 1

            epoch_loss_sum += float(outputs.loss.item())
            loss_sum += float(outputs.loss.item())
            epoch_batches += 1
            total_batches += 1

            if accum_counter >= max(1, effective_grad_accum):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                accum_counter = 0
                steps += 1

                if int(cfg.save_every_steps) > 0 and (steps % int(cfg.save_every_steps) == 0):
                    _save_checkpoint(steps, epoch + 1)

                if steps >= max_updates:
                    break

        if accum_counter > 0 and steps < max_updates:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            steps += 1
            if int(cfg.save_every_steps) > 0 and (steps % int(cfg.save_every_steps) == 0):
                _save_checkpoint(steps, epoch + 1)

        if epoch_batches > 0:
            epoch_train_losses.append(float(epoch_loss_sum / max(1, epoch_batches)))

        val_loss = _evaluate(val_loader)
        if val_loss is not None:
            epoch_val_losses.append(float(val_loss))
            if best_val_loss is None or float(val_loss) < float(best_val_loss) - 1e-4:
                best_val_loss = float(val_loss)
                no_improve_epochs = 0
                try:
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                except Exception:
                    best_state = None
            else:
                no_improve_epochs += 1

        actual_epochs = epoch + 1

        if steps >= max_updates:
            break
        if (
            val_loader is not None
            and best_val_loss is not None
            and no_improve_epochs >= early_stop_patience
            and actual_epochs >= 2
        ):
            early_stopped = True
            break

    if best_state is not None:
        try:
            model.load_state_dict(best_state)
        except Exception:
            pass

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    avg_loss = float(loss_sum / max(1, total_batches))
    duration = float(time.time() - started)
    train_loss_last = float(epoch_train_losses[-1]) if epoch_train_losses else avg_loss
    val_loss_last = float(epoch_val_losses[-1]) if epoch_val_losses else None
    val_perplexity = (
        float(math.exp(min(20.0, val_loss_last)))
        if val_loss_last is not None and math.isfinite(val_loss_last)
        else None
    )

    report = {
        "status": "trained",
        "model_dir": str(cfg.output_dir),
        "samples": int(len(texts)),
        "train_text_samples": int(len(train_texts)),
        "val_text_samples": int(len(val_texts)),
        "packed_samples": int(len(train_dataset) + (len(val_dataset) if val_dataset is not None else 0)),
        "train_blocks": int(len(train_dataset)),
        "val_blocks": int(len(val_dataset) if val_dataset is not None else 0),
        "steps": int(steps),
        "optimizer_steps": int(steps),
        "epochs": int(actual_epochs),
        "epochs_requested": int(cfg.epochs),
        "avg_loss": avg_loss,
        "train_loss_last_epoch": train_loss_last,
        "val_loss": val_loss_last,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "val_perplexity": val_perplexity,
        "save_every_steps": int(cfg.save_every_steps),
        "checkpoint_count": int(checkpoint_count),
        "duration_seconds": duration,
        "architecture": "self_trained_transformer",
        "backend": "transformers_local",
        "training_backend": "scratch",
        "device": str(device),
        "model_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "training_profile": str(profile),
        "early_stopped": bool(early_stopped),
        "early_stop_patience": int(early_stop_patience),
        "adaptive_profile": str(adaptive_profile),
        "effective_config": {
            "context_length": int(effective_context),
            "hidden_size": int(effective_hidden),
            "num_layers": int(effective_layers),
            "num_heads": int(effective_heads),
            "batch_size": int(effective_batch),
            "grad_accum_steps": int(effective_grad_accum),
            "learning_rate": float(effective_lr),
            "max_updates": int(max_updates),
            "warmup_steps": int(warmup_steps),
        },
    }

    try:
        with (cfg.output_dir / "model_card.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "architecture": "gpt2_lm_head",
                    "training_profile": str(profile),
                    "samples": {
                        "total_text_samples": int(len(texts)),
                        "train_text_samples": int(len(train_texts)),
                        "val_text_samples": int(len(val_texts)),
                        "train_blocks": int(len(train_dataset)),
                        "val_blocks": int(len(val_dataset) if val_dataset is not None else 0),
                    },
                    "parameters": {
                        "total": int(total_parameters),
                        "trainable": int(trainable_parameters),
                    },
                    "effective_config": dict(report.get("effective_config") or {}),
                    "metrics": {
                        "avg_loss": float(avg_loss),
                        "train_loss_last_epoch": float(train_loss_last),
                        "val_loss": float(val_loss_last) if val_loss_last is not None else None,
                        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
                        "val_perplexity": float(val_perplexity) if val_perplexity is not None else None,
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass
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
    raw_text_count = int(len(texts))
    augmentation_applied = False
    profile = _normalize_training_profile(cfg.training_profile)
    requested_backend = _normalize_training_backend(cfg.training_backend, profile=profile)

    if 0 < raw_text_count < int(cfg.min_text_samples):
        target_count = max(int(cfg.min_text_samples), min(256, raw_text_count * 4))
        texts = _augment_low_resource_texts(
            texts,
            target_count=target_count,
            seed=int(cfg.seed),
        )
        augmentation_applied = int(len(texts)) > raw_text_count

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
            "training_profile": profile,
            "training_backend": requested_backend,
        }

    report: dict[str, Any]
    if requested_backend == "pretrained_lora":
        report = dict(_run_professional_training(cfg, texts) or {})
        status = str(report.get("status", "") or "").strip().lower()
        if status != "trained" and bool(cfg.allow_scratch_fallback):
            primary_msg = str(report.get("message", "") or "").strip()
            primary_err = str(report.get("error", "") or "").strip()
            fallback = dict(_run_self_training(cfg, texts) or {})
            fallback["status"] = str(fallback.get("status", "trained") or "trained")
            fallback["training_backend"] = "scratch"
            fallback["requested_training_backend"] = requested_backend
            fallback["fallback_from_backend"] = "pretrained_lora"
            if primary_msg:
                fallback["fallback_reason"] = primary_msg
            if primary_err:
                fallback["fallback_error"] = primary_err
            notes_prefix = (
                "Professional backend failed; automatically fell back to scratch backend."
            )
            old_notes = str(fallback.get("notes", "") or "").strip()
            fallback["notes"] = (
                f"{notes_prefix} {old_notes}".strip() if old_notes else notes_prefix
            )
            report = fallback
    else:
        report = dict(_run_self_training(cfg, texts) or {})
        report.setdefault("training_backend", "scratch")

    report.setdefault("training_profile", profile)
    report.setdefault("training_backend", requested_backend)
    report.setdefault("requested_training_backend", requested_backend)
    if augmentation_applied:
        report["original_samples"] = int(raw_text_count)
        report["augmented_samples"] = int(max(0, len(texts) - raw_text_count))
        report["samples"] = int(len(texts))
        msg = (
            f"Low-data augmentation expanded corpus from {raw_text_count} "
            f"to {len(texts)} samples."
        )
        prior_notes = str(report.get("notes", "")).strip()
        report["notes"] = f"{prior_notes} {msg}".strip() if prior_notes else msg

    if raw_text_count < int(cfg.min_text_samples):
        report["warning"] = (
            f"Low base corpus size ({raw_text_count} < {cfg.min_text_samples}); "
            "quality may be limited."
        )
    return report

