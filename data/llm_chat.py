"""Local LLM chat assistant with internet retrieval (no API required).

This module provides ChatGPT-like chat behavior using:
- A local instruction model via transformers (CPU/GPU)
- Internet/news retrieval from existing China-compatible collectors
- Optional ACTION command extraction for app control routing
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any

from config.runtime_env import env_flag, env_text
from utils.logger import get_logger

from .news_aggregator import get_news_aggregator
from .news_collector import NewsArticle, get_collector

log = get_logger(__name__)

_TRANSFORMERS_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    import torch

    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False


class LLMChatAssistant:
    """Retrieval-augmented local LLM assistant."""

    _STOPWORDS_EN = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "what",
        "when",
        "where",
        "which",
        "about",
        "into",
        "stock",
        "market",
        "please",
    }
    _STOPWORDS_ZH = {
        "什么",
        "怎么",
        "如何",
        "为什么",
        "一下",
        "可以",
        "帮我",
        "请问",
        "今天",
        "现在",
        "最新",
        "市场",
        "股票",
    }

    def __init__(self) -> None:
        # Local model settings (no remote API required).
        self.model_id = str(
            env_text("TRADING_LOCAL_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
            or "Qwen/Qwen2.5-1.5B-Instruct"
        ).strip()
        self.model_path = str(env_text("TRADING_LOCAL_LLM_MODEL_PATH", "") or "").strip()
        self.device_pref = str(env_text("TRADING_LOCAL_LLM_DEVICE", "auto") or "auto").strip().lower()
        self.max_new_tokens = int(float(env_text("TRADING_LOCAL_LLM_MAX_NEW_TOKENS", "420") or 420))
        self.temperature = float(env_text("TRADING_LOCAL_LLM_TEMPERATURE", "0.25") or 0.25)
        self.top_p = float(env_text("TRADING_LOCAL_LLM_TOP_P", "0.92") or 0.92)
        self.trust_remote_code = bool(env_flag("TRADING_LOCAL_LLM_TRUST_REMOTE_CODE", "0"))

        self.max_news_items = int(float(env_text("TRADING_LLM_SEARCH_LIMIT", "20") or 20))
        self._search_cache: dict[str, tuple[dict[str, Any], float]] = {}
        self._search_ttl_seconds = 180.0

        self._tokenizer: Any = None
        self._model: Any = None
        self._device = "cpu"
        self._local_load_error: str = ""
        self._local_ready = False
        self._load_attempted = False

    # -----------------------------
    # Local model loading/generation
    # -----------------------------

    @property
    def configured(self) -> bool:
        return bool(_TRANSFORMERS_AVAILABLE)

    @property
    def local_ready(self) -> bool:
        return bool(self._local_ready)

    def _resolve_model_source(self) -> str:
        if self.model_path:
            return self.model_path
        return self.model_id

    def _ensure_local_model(self) -> bool:
        if self._local_ready and self._tokenizer is not None and self._model is not None:
            return True
        if self._load_attempted and not self._local_ready:
            return False

        self._load_attempted = True
        if not _TRANSFORMERS_AVAILABLE:
            self._local_load_error = "transformers is not installed."
            return False

        source = self._resolve_model_source()
        device = "cpu"
        if _TORCH_AVAILABLE:
            if self.device_pref == "cuda":
                device = "cuda"
            elif self.device_pref == "cpu":
                device = "cpu"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                source,
                trust_remote_code=bool(self.trust_remote_code),
            )

            model = AutoModelForCausalLM.from_pretrained(
                source,
                trust_remote_code=bool(self.trust_remote_code),
            )
            if _TORCH_AVAILABLE:
                if device == "cuda" and torch.cuda.is_available():
                    model = model.to("cuda")
                else:
                    model = model.to("cpu")
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            self._local_ready = True
            self._local_load_error = ""
            log.info("Local chat model loaded: source=%s device=%s", source, device)
            return True
        except Exception as exc:
            self._local_load_error = str(exc)
            self._local_ready = False
            self._tokenizer = None
            self._model = None
            log.warning("Local chat model load failed: %s", exc)
            return False

    @staticmethod
    def _extract_action_from_text(text: str) -> str:
        lines = [str(x).strip() for x in str(text or "").splitlines() if str(x).strip()]
        for line in reversed(lines):
            if line.lower().startswith("action:"):
                return str(line.split(":", 1)[1]).strip()
        return ""

    @staticmethod
    def _strip_action_line(text: str) -> str:
        lines = []
        for line in str(text or "").splitlines():
            if str(line).strip().lower().startswith("action:"):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def _run_local_llm(
        self,
        *,
        prompt: str,
        context: dict[str, Any],
        app_state: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> tuple[str, str]:
        if not self._ensure_local_model():
            raise RuntimeError(self._local_load_error or "Local model unavailable.")

        tok = self._tokenizer
        model = self._model
        if tok is None or model is None:
            raise RuntimeError("Local model/tokenizer not initialized.")

        hist_lines: list[str] = []
        for item in list(history or [])[-8:]:
            role = str(item.get("role", item.get("sender", "user")) or "user")
            text = str(item.get("text", "") or "").strip()
            if text:
                hist_lines.append(f"{role}: {text}")

        context_lines = "\n".join(list(context.get("headlines", []) or [])[:12])
        sent = dict(context.get("sentiment", {}) or {})
        sent_line = (
            f"sentiment overall={float(sent.get('overall', 0.0) or 0.0):+.2f}, "
            f"label={str(sent.get('label', 'neutral') or 'neutral')}, "
            f"confidence={float(sent.get('confidence', 0.0) or 0.0):.0%}, "
            f"total={int(sent.get('total_news', 0) or 0)}"
        )

        system_prompt = (
            "You are an on-device AI copilot for a trading analysis app. "
            "Use the provided internet/news context first. "
            "Answer clearly and naturally in normal conversation language. "
            "Understand natural control requests (not only strict commands), for example "
            "'please watch this stock', 'switch to 15 minutes', or 'analyze 600519'. "
            "If user asks to control app, append exactly one line using a canonical command: "
            "ACTION: analyze <code> | start monitoring | stop monitoring | scan market | "
            "refresh sentiment | set interval <1m|5m|15m|30m|60m|1d> | "
            "set forecast <bars> | set lookback <bars> | add watchlist <code> | "
            "remove watchlist <code> | train model | train llm | auto train llm. "
            "If no action is needed, append ACTION: ."
        )
        user_prompt = (
            f"[APP STATE]\n{app_state}\n\n"
            f"[INTERNET SENTIMENT]\n{sent_line}\n\n"
            f"[HEADLINES]\n{context_lines}\n\n"
            f"[RECENT CHAT]\n{chr(10).join(hist_lines)}\n\n"
            f"[USER QUESTION]\n{prompt}"
        )

        if hasattr(tok, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            model_inputs = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            raw = (
                f"System: {system_prompt}\n\n"
                f"User: {user_prompt}\n\n"
                "Assistant:"
            )
            model_inputs = tok(raw, return_tensors="pt").input_ids

        if _TORCH_AVAILABLE and self._device == "cuda" and torch.cuda.is_available():
            model_inputs = model_inputs.to("cuda")

        do_sample = bool(self.temperature > 0.0)
        with torch.no_grad() if _TORCH_AVAILABLE else _null_context():
            output_ids = model.generate(
                model_inputs,
                max_new_tokens=max(64, int(self.max_new_tokens)),
                do_sample=do_sample,
                temperature=float(max(0.0, self.temperature)),
                top_p=float(min(1.0, max(0.1, self.top_p))),
                pad_token_id=getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None),
                eos_token_id=getattr(tok, "eos_token_id", None),
            )

        generated = output_ids[0][model_inputs.shape[-1] :]
        content = tok.decode(generated, skip_special_tokens=True).strip()
        action = self._extract_action_from_text(content)
        answer = self._strip_action_line(content)
        return answer, action

    # -----------------------------
    # Retrieval
    # -----------------------------

    def _extract_keywords(self, query: str) -> list[str]:
        raw_tokens = re.findall(r"[\u4e00-\u9fff]{2,10}|[A-Za-z]{3,24}|\d{6}", str(query or ""))
        out: list[str] = []
        seen: set[str] = set()
        for token in raw_tokens:
            t = str(token or "").strip()
            if not t:
                continue
            low = t.lower()
            if low in self._STOPWORDS_EN or t in self._STOPWORDS_ZH:
                continue
            if low in seen:
                continue
            seen.add(low)
            out.append(t)
            if len(out) >= 10:
                break
        return out

    @staticmethod
    def _article_time_iso(article: NewsArticle) -> str:
        ts = getattr(article, "published_at", None)
        if isinstance(ts, datetime):
            return ts.isoformat()
        return ""

    def _collector_retrieval(self, query: str) -> list[NewsArticle]:
        collector = get_collector()
        keywords = self._extract_keywords(query)
        articles = collector.collect_news(
            keywords=keywords if keywords else None,
            limit=max(12, self.max_news_items),
            hours_back=96,
        )
        return list(articles or [])

    @staticmethod
    def _news_item_to_article(item: object, idx: int) -> NewsArticle:
        title = str(getattr(item, "title", "") or "")
        content = str(getattr(item, "content", "") or "")
        publish_time = getattr(item, "publish_time", None)
        if not isinstance(publish_time, datetime):
            publish_time = datetime.now()
        text = f"{title} {content}"
        zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        language = "zh" if zh_chars >= 4 else "en"
        return NewsArticle(
            id=f"agg:{idx}:{abs(hash(title)) % 10_000_000}",
            title=title,
            content=content,
            summary=content[:220],
            source=str(getattr(item, "source", "") or ""),
            url=str(getattr(item, "url", "") or ""),
            published_at=publish_time,
            collected_at=datetime.now(),
            language=language,
            category=str(getattr(item, "category", "") or "market"),
            sentiment_score=float(getattr(item, "sentiment_score", 0.0) or 0.0),
            relevance_score=float(getattr(item, "importance", 0.5) or 0.5),
            entities=list(getattr(item, "stock_codes", []) or []),
            tags=list(getattr(item, "keywords", []) or []),
        )

    def search_internet(self, query: str, symbol: str | None = None) -> dict[str, Any]:
        key = f"{str(symbol or '')}:{str(query or '').strip().lower()}"
        now = time.time()
        cached = self._search_cache.get(key)
        if cached and (now - float(cached[1])) <= self._search_ttl_seconds:
            return dict(cached[0])

        agg = get_news_aggregator()
        out_articles: list[NewsArticle] = []
        seen_titles: set[str] = set()

        for article in self._collector_retrieval(query):
            t = str(getattr(article, "title", "") or "").strip().lower()
            if not t or t in seen_titles:
                continue
            seen_titles.add(t)
            out_articles.append(article)

        if symbol:
            for i, item in enumerate(list(agg.get_stock_news(symbol, count=24, force_refresh=True) or [])):
                art = self._news_item_to_article(item, i)
                t = art.title.strip().lower()
                if not t or t in seen_titles:
                    continue
                seen_titles.add(t)
                out_articles.append(art)

        for i, item in enumerate(list(agg.get_market_news(count=24, force_refresh=True) or []), start=1000):
            art = self._news_item_to_article(item, i)
            t = art.title.strip().lower()
            if not t or t in seen_titles:
                continue
            seen_titles.add(t)
            out_articles.append(art)

        out_articles.sort(
            key=lambda a: getattr(a, "published_at", datetime.min),
            reverse=True,
        )
        out_articles = out_articles[: max(8, self.max_news_items)]

        sentiment = agg.get_sentiment_summary(symbol if symbol else None)
        lines: list[str] = []
        for i, article in enumerate(out_articles, start=1):
            src = str(getattr(article, "source", "") or "unknown")
            ts = self._article_time_iso(article)
            title = str(getattr(article, "title", "") or "").strip()
            if title:
                lines.append(f"{i}. [{src}] {ts} {title}")

        payload = {
            "query": str(query or ""),
            "symbol": str(symbol or ""),
            "timestamp": datetime.now().isoformat(),
            "sentiment": {
                "overall": float(sentiment.get("overall_sentiment", 0.0) or 0.0),
                "label": str(sentiment.get("label", "neutral") or "neutral"),
                "confidence": float(sentiment.get("confidence", 0.0) or 0.0),
                "total_news": int(sentiment.get("total", 0) or 0),
            },
            "headlines": lines[: max(8, self.max_news_items)],
        }
        self._search_cache[key] = (dict(payload), now)
        return payload

    # -----------------------------
    # Answer flow
    # -----------------------------

    def _fallback_answer(self, *, prompt: str, context: dict[str, Any], app_state: dict[str, Any]) -> tuple[str, str]:
        sentiment = dict(context.get("sentiment", {}) or {})
        headlines = list(context.get("headlines", []) or [])
        overall = float(sentiment.get("overall", 0.0) or 0.0)
        label = str(sentiment.get("label", "neutral") or "neutral")
        conf = float(sentiment.get("confidence", 0.0) or 0.0)
        top = "\n".join(headlines[:5]) if headlines else "No fresh headlines found."
        source = self.model_path or self.model_id
        answer = (
            f"Internet context (China-compatible sources) for: {prompt}\n"
            f"Sentiment: {overall:+.2f} ({label}), confidence {conf:.0%}.\n"
            f"Top headlines:\n{top}\n\n"
            "Local LLM is not ready yet. Set a local model and restart:\n"
            f"- TRADING_LOCAL_LLM_MODEL_PATH=<folder>  (or use model id: {source})\n"
            "- Recommended model ids: Qwen/Qwen2.5-7B-Instruct or Qwen/Qwen2.5-14B-Instruct\n"
            "- Optional: TRADING_LOCAL_LLM_DEVICE=cpu|cuda|auto"
        )
        _ = app_state
        return answer, ""

    def answer(
        self,
        *,
        prompt: str,
        symbol: str | None,
        app_state: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
        allow_search: bool = True,
    ) -> dict[str, Any]:
        context = {"query": prompt, "headlines": [], "sentiment": {}}
        if allow_search:
            try:
                context = self.search_internet(prompt, symbol=symbol)
            except Exception as exc:
                log.warning("chat search failed: %s", exc)

        answer = ""
        action = ""
        if self.configured:
            try:
                answer, action = self._run_local_llm(
                    prompt=prompt,
                    context=context,
                    app_state=app_state,
                    history=list(history or []),
                )
            except Exception as exc:
                log.warning("Local LLM failed, using fallback answer: %s", exc)
                answer, action = self._fallback_answer(
                    prompt=prompt,
                    context=context,
                    app_state=app_state,
                )
        else:
            answer, action = self._fallback_answer(
                prompt=prompt,
                context=context,
                app_state=app_state,
            )

        return {
            "answer": str(answer or "").strip(),
            "action": str(action or "").strip(),
            "context": context,
            "local_model_ready": bool(self.local_ready),
        }


class _null_context:
    """Minimal context manager for non-torch environments."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        _ = (exc_type, exc, tb)
        return None


_assistant: LLMChatAssistant | None = None


def get_chat_assistant() -> LLMChatAssistant:
    global _assistant
    if _assistant is None:
        _assistant = LLMChatAssistant()
    return _assistant


def reset_chat_assistant() -> None:
    global _assistant
    _assistant = None
