"""Local LLM runtime with a self-trained transformer-first design.

Primary backend:
- `transformers_local`: local model artifacts under your filesystem

Optional compatibility backends are available but not used by default:
- `ollama`
- `vllm`
- `llama_cpp`
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator

try:
    import httpx
except Exception:  # pragma: no cover - optional dependency in some test envs
    httpx = None  # type: ignore[assignment]

from config.runtime_env import env_flag, env_float, env_int, env_text
from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


class LLMBackend(str, Enum):
    """Supported backends."""

    TRANSFORMERS_LOCAL = "transformers_local"
    OLLAMA = "ollama"
    VLLM = "vllm"
    LLAMA_CPP = "llama_cpp"

    @classmethod
    def from_text(cls, value: str) -> "LLMBackend":
        text = str(value or "").strip().lower()
        aliases = {
            "transformers": cls.TRANSFORMERS_LOCAL,
            "hf": cls.TRANSFORMERS_LOCAL,
            "huggingface": cls.TRANSFORMERS_LOCAL,
            "llama.cpp": cls.LLAMA_CPP,
        }
        if text in aliases:
            return aliases[text]
        return cls(text)


@dataclass
class LocalLLMConfig:
    """Configuration for LocalLLM."""

    backend: LLMBackend = LLMBackend.TRANSFORMERS_LOCAL
    model_name: str = "self-chat-transformer"
    host: str = "127.0.0.1"
    port: int = 11434
    temperature: float = 0.4
    top_p: float = 0.9
    max_tokens: int = 512
    repetition_penalty: float = 1.05
    context_window: int = 4096
    request_timeout_seconds: float = 180.0
    seed: int | None = 42
    local_model_path: str = ""
    local_files_only: bool = True
    trust_remote_code: bool = True
    stream_chunk_chars: int = 64
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "LocalLLMConfig":
        backend = LLMBackend.from_text(
            env_text("TRADING_LLM_BACKEND", "transformers_local")
        )
        model_name = env_text("TRADING_LLM_MODEL", "self-chat-transformer")
        host = env_text("TRADING_LLM_HOST", "127.0.0.1")
        default_port = 11434 if backend == LLMBackend.OLLAMA else 8000
        return cls(
            backend=backend,
            model_name=model_name,
            host=host,
            port=env_int("TRADING_LLM_PORT", default_port),
            temperature=env_float("TRADING_CHAT_TEMPERATURE", "0.4"),
            top_p=env_float("TRADING_CHAT_TOP_P", "0.9"),
            max_tokens=env_int("TRADING_CHAT_MAX_TOKENS", 512),
            repetition_penalty=env_float("TRADING_CHAT_REPETITION_PENALTY", "1.05"),
            context_window=env_int("TRADING_CHAT_CONTEXT_WINDOW", 4096),
            request_timeout_seconds=env_float("TRADING_CHAT_TIMEOUT_SECONDS", "180"),
            seed=env_int("TRADING_CHAT_SEED", 42),
            local_model_path=env_text("TRADING_CHAT_MODEL_PATH", ""),
            local_files_only=env_flag("TRADING_CHAT_LOCAL_FILES_ONLY", "1"),
            trust_remote_code=env_flag("TRADING_CHAT_TRUST_REMOTE_CODE", "1"),
        )


@dataclass
class LLMResponse:
    """Normalized generation response."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    finish_reason: str = "stop"
    raw: dict[str, Any] = field(default_factory=dict)


class LocalLLM:
    """Unified local LLM client."""

    def __init__(self, config: LocalLLMConfig | None = None) -> None:
        self.config = config or LocalLLMConfig.from_env()
        self._initialized = False
        self._client: Any = None
        self._lock = threading.RLock()

        # Lazy transformer runtime.
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._resolved_model_ref = ""

    async def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return

        if self.config.backend in {
            LLMBackend.OLLAMA,
            LLMBackend.VLLM,
            LLMBackend.LLAMA_CPP,
        }:
            if httpx is None:
                raise RuntimeError("httpx is required for HTTP LLM backends")
            timeout = max(10.0, float(self.config.request_timeout_seconds))
            self._client = httpx.AsyncClient(timeout=timeout)

        with self._lock:
            self._initialized = True
        log.info(
            "LocalLLM initialized: backend=%s model=%s",
            self.config.backend.value,
            self.config.model_name,
        )

    async def shutdown(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass
        with self._lock:
            self._initialized = False

    def _base_url(self) -> str:
        host = str(self.config.host or "127.0.0.1").strip()
        if host.startswith("http://") or host.startswith("https://"):
            return host.rstrip("/")
        return f"http://{host}:{int(self.config.port)}"

    def _normalize_history(self, history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for row in list(history or []):
            role = str(row.get("role", "") or "").strip().lower()
            content = str(row.get("content", row.get("text", "")) or "").strip()
            if not content:
                continue
            if role not in {"system", "user", "assistant"}:
                sender = str(row.get("sender", "")).strip().lower()
                if sender in {"ai", "assistant"}:
                    role = "assistant"
                elif sender in {"system"}:
                    role = "system"
                else:
                    role = "user"
            rows.append({"role": role, "content": content})
        return rows

    def _build_messages(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        history: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        sys_text = str(system_prompt or "").strip()
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        messages.extend(self._normalize_history(history))
        messages.append({"role": "user", "content": str(prompt or "").strip()})
        return messages

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        history: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        if not self._initialized:
            await self.initialize()
        started = time.perf_counter()
        try:
            if self.config.backend == LLMBackend.TRANSFORMERS_LOCAL:
                return await self._generate_transformers(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history=history,
                    started=started,
                )
            if self.config.backend == LLMBackend.OLLAMA:
                return await self._generate_ollama(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history=history,
                    started=started,
                )
            if self.config.backend in {LLMBackend.VLLM, LLMBackend.LLAMA_CPP}:
                return await self._generate_openai_compatible(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history=history,
                    started=started,
                )
        except Exception as exc:
            log.warning("LLM generation failed: backend=%s error=%s", self.config.backend.value, exc)
            return self._error_response(started, f"LLM backend unavailable: {exc}")
        return self._error_response(started, "Unsupported LLM backend")

    async def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        if not self._initialized:
            await self.initialize()

        if self.config.backend == LLMBackend.OLLAMA:
            async for chunk in self._generate_stream_ollama(
                prompt=prompt,
                system_prompt=system_prompt,
                history=history,
            ):
                yield chunk
            return

        result = await self.generate(prompt, system_prompt=system_prompt, history=history)
        text = str(result.content or "")
        step = max(16, int(self.config.stream_chunk_chars))
        for i in range(0, len(text), step):
            yield text[i:i + step]

    async def _generate_ollama(
        self,
        *,
        prompt: str,
        system_prompt: str,
        history: list[dict[str, Any]] | None,
        started: float,
    ) -> LLMResponse:
        client = self._client
        if client is None:
            raise RuntimeError("HTTP client not initialized")
        payload = {
            "model": self.config.model_name,
            "messages": self._build_messages(
                prompt=prompt,
                system_prompt=system_prompt,
                history=history,
            ),
            "stream": False,
            "options": {
                "temperature": float(max(0.0, self.config.temperature)),
                "top_p": float(max(0.0, self.config.top_p)),
                "num_predict": int(max(1, self.config.max_tokens)),
                "repeat_penalty": float(max(1.0, self.config.repetition_penalty)),
                "seed": int(self.config.seed) if self.config.seed is not None else None,
            },
        }
        resp = await client.post(f"{self._base_url()}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        message = data.get("message", {}) if isinstance(data, dict) else {}
        content = str(message.get("content", "") or "").strip()
        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(data.get("eval_count", 0) or 0)
        if prompt_tokens <= 0:
            prompt_tokens = max(1, len(str(prompt).split()))
        if completion_tokens <= 0:
            completion_tokens = max(1, len(content.split()))
        return LLMResponse(
            content=content,
            model=str(data.get("model", self.config.model_name) or self.config.model_name),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=float((time.perf_counter() - started) * 1000.0),
            finish_reason=str(data.get("done_reason", "stop") or "stop"),
            raw=data if isinstance(data, dict) else {},
        )

    async def _generate_stream_ollama(
        self,
        *,
        prompt: str,
        system_prompt: str,
        history: list[dict[str, Any]] | None,
    ) -> AsyncIterator[str]:
        client = self._client
        if client is None:
            raise RuntimeError("HTTP client not initialized")
        payload = {
            "model": self.config.model_name,
            "messages": self._build_messages(
                prompt=prompt,
                system_prompt=system_prompt,
                history=history,
            ),
            "stream": True,
            "options": {
                "temperature": float(max(0.0, self.config.temperature)),
                "top_p": float(max(0.0, self.config.top_p)),
                "num_predict": int(max(1, self.config.max_tokens)),
                "repeat_penalty": float(max(1.0, self.config.repetition_penalty)),
            },
        }
        async with client.stream("POST", f"{self._base_url()}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                msg = row.get("message", {}) if isinstance(row, dict) else {}
                chunk = str(msg.get("content", "") or "")
                if chunk:
                    yield chunk

    async def _generate_openai_compatible(
        self,
        *,
        prompt: str,
        system_prompt: str,
        history: list[dict[str, Any]] | None,
        started: float,
    ) -> LLMResponse:
        client = self._client
        if client is None:
            raise RuntimeError("HTTP client not initialized")
        payload = {
            "model": self.config.model_name,
            "messages": self._build_messages(
                prompt=prompt,
                system_prompt=system_prompt,
                history=history,
            ),
            "temperature": float(max(0.0, self.config.temperature)),
            "top_p": float(max(0.0, self.config.top_p)),
            "max_tokens": int(max(1, self.config.max_tokens)),
            "stream": False,
        }
        resp = await client.post(f"{self._base_url()}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        choices = data.get("choices", []) if isinstance(data, dict) else []
        first = choices[0] if choices else {}
        msg = first.get("message", {}) if isinstance(first, dict) else {}
        content = str(msg.get("content", "") or "").strip()
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        if prompt_tokens <= 0:
            prompt_tokens = max(1, len(str(prompt).split()))
        if completion_tokens <= 0:
            completion_tokens = max(1, len(content.split()))
        return LLMResponse(
            content=content,
            model=str(data.get("model", self.config.model_name) or self.config.model_name),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=float((time.perf_counter() - started) * 1000.0),
            finish_reason=str(first.get("finish_reason", "stop") or "stop"),
            raw=data if isinstance(data, dict) else {},
        )

    def _resolve_local_model_ref(self) -> str:
        ref = str(self.config.local_model_path or "").strip()
        if not ref:
            ref = env_text("TRADING_CHAT_MODEL_PATH", "").strip()
        if not ref:
            default_dir = Path(getattr(CONFIG, "llm_model_dir", CONFIG.model_dir))
            local_dir = default_dir / "chat_transformer"
            if local_dir.exists():
                ref = str(local_dir)
        if not ref:
            ref = str(self.config.model_name or "").strip()
        return ref

    def _ensure_transformers_loaded_sync(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        mirror = env_text("TRADING_HF_ENDPOINT", "").strip()
        if mirror and not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = mirror

        model_ref = self._resolve_local_model_ref()
        local_only = bool(self.config.local_files_only)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "transformers_local requires torch + transformers installed"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            trust_remote_code=bool(self.config.trust_remote_code),
            local_files_only=local_only,
        )
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": bool(self.config.trust_remote_code),
            "local_files_only": local_only,
        }
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._resolved_model_ref = str(model_ref)

    def _build_transformers_prompt(self, messages: list[dict[str, str]]) -> str:
        tokenizer = self._tokenizer
        if tokenizer is None:
            return ""
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return str(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            except Exception:
                pass

        parts: list[str] = []
        for msg in messages:
            role = str(msg.get("role", "user")).strip().upper()
            text = str(msg.get("content", "")).strip()
            if text:
                parts.append(f"{role}: {text}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    def _generate_transformers_sync(
        self,
        *,
        prompt: str,
        system_prompt: str,
        history: list[dict[str, Any]] | None,
        started: float,
    ) -> LLMResponse:
        self._ensure_transformers_loaded_sync()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch

        messages = self._build_messages(prompt, system_prompt=system_prompt, history=history)
        compiled_prompt = self._build_transformers_prompt(messages)
        inputs = tokenizer(
            compiled_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max(256, int(self.config.context_window)),
        )
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        do_sample = bool(float(self.config.temperature) > 1e-6)
        if self.config.seed is not None:
            try:
                torch.manual_seed(int(self.config.seed))
            except Exception:
                pass

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(max(1, self.config.max_tokens)),
                do_sample=do_sample,
                temperature=(
                    float(max(0.0, self.config.temperature))
                    if do_sample
                    else 1.0
                ),
                top_p=float(max(0.0, self.config.top_p)),
                repetition_penalty=float(max(1.0, self.config.repetition_penalty)),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        generated_ids = output_ids[0][input_len:]
        content = str(tokenizer.decode(generated_ids, skip_special_tokens=True)).strip()
        if not content:
            content = "I could not generate a response for this prompt."

        prompt_tokens = max(1, int(input_len))
        if hasattr(generated_ids, "shape"):
            completion_tokens = int(generated_ids.shape[-1])
        else:
            completion_tokens = len(content.split())
        completion_tokens = max(1, int(completion_tokens))
        return LLMResponse(
            content=content,
            model=self._resolved_model_ref or self.config.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=float((time.perf_counter() - started) * 1000.0),
            finish_reason="stop",
            raw={"backend": "transformers_local"},
        )

    async def _generate_transformers(
        self,
        *,
        prompt: str,
        system_prompt: str,
        history: list[dict[str, Any]] | None,
        started: float,
    ) -> LLMResponse:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._generate_transformers_sync(
                prompt=prompt,
                system_prompt=system_prompt,
                history=history,
                started=started,
            ),
        )

    def _error_response(self, started: float, text: str) -> LLMResponse:
        content = str(text or "LLM backend unavailable").strip()
        completion = max(1, len(content.split()))
        return LLMResponse(
            content=content,
            model=self.config.model_name,
            prompt_tokens=1,
            completion_tokens=completion,
            total_tokens=1 + completion,
            latency_ms=float((time.perf_counter() - started) * 1000.0),
            finish_reason="error",
            raw={"error": content},
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "backend": self.config.backend.value,
            "model_name": self.config.model_name,
            "initialized": bool(self._initialized),
            "transformers_loaded": bool(self._model is not None and self._tokenizer is not None),
            "resolved_model_ref": str(self._resolved_model_ref or ""),
            "local_files_only": bool(self.config.local_files_only),
        }


_llm_instance: LocalLLM | None = None
_llm_lock = threading.Lock()


def get_llm(config: LocalLLMConfig | None = None) -> LocalLLM:
    """Get singleton LocalLLM instance."""
    global _llm_instance
    with _llm_lock:
        if _llm_instance is None:
            _llm_instance = LocalLLM(config)
            return _llm_instance
        if config is not None and _llm_instance.config != config:
            _llm_instance = LocalLLM(config)
        return _llm_instance


async def initialize_llm(config: LocalLLMConfig | None = None) -> LocalLLM:
    """Initialize singleton LocalLLM."""
    llm = get_llm(config)
    await llm.initialize()
    return llm
