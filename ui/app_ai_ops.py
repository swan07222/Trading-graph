from __future__ import annotations

import html
import re
import time
from datetime import datetime
from typing import Any

from ui.background_tasks import WorkerThread
from utils.logger import get_logger
from utils.recoverable import COMMON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)
_UI_AI_RECOVERABLE_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    haystack = str(text or "").lower()
    return any(str(n).lower() in haystack for n in needles)


def _extract_interval_token_from_text(text: str) -> str:
    t = str(text or "").strip().lower()
    if not t:
        return ""

    direct = re.search(r"\b(1m|5m|15m|30m|60m|1d)\b", t)
    if direct:
        return str(direct.group(1) or "").strip()

    en_min = re.search(r"\b(1|5|15|30|60)\s*(m|min|mins|minute|minutes)\b", t)
    if en_min:
        return f"{int(en_min.group(1))}m"

    zh_min = re.search(r"(1|5|15|30|60)\s*分钟", t)
    if zh_min:
        return f"{int(zh_min.group(1))}m"

    if _contains_any(t, ("daily", "day", "1 day", "1d", "日线", "天线", "日k")):
        return "1d"
    return ""


def _extract_symbol_or_current(self: Any, text: str) -> str:
    m = re.search(r"\b(\d{6})\b", str(text or ""))
    if m:
        return self._ui_norm(str(m.group(1) or ""))
    current = self._ui_norm(self.stock_input.text())
    pronouns = (
        "this stock",
        "this symbol",
        "current stock",
        "that stock",
        "这只股票",
        "这个股票",
        "当前股票",
        "这支票",
        "它",
    )
    if current and _contains_any(str(text or ""), pronouns):
        return current
    return ""


def _chat_state_summary(self: Any) -> str:
    symbol = self._ui_norm(self.stock_input.text()) or "--"
    interval = self._normalize_interval_token(self.interval_combo.currentText())
    forecast = int(self.forecast_spin.value())
    lookback = int(self.lookback_spin.value())
    monitor_on = bool(self.monitor and self.monitor.isRunning())
    return (
        f"Current state: symbol={symbol}, interval={interval}, "
        f"forecast={forecast} bars, lookback={lookback} bars, "
        f"monitoring={'on' if monitor_on else 'off'}."
    )


def _append_ai_chat_message(
    self: Any,
    sender: str,
    message: str,
    *,
    role: str = "assistant",
    level: str = "info",
) -> None:
    widget = getattr(self, "ai_chat_view", None)
    if widget is None:
        return

    colors = {
        "user": "#7cc7ff",
        "assistant": "#dbe4f3",
        "system": "#9aa6bf",
        "error": "#ff6b6b",
        "warning": "#f9c74f",
        "success": "#72f1b8",
        "info": "#dbe4f3",
    }
    safe_sender = html.escape(str(sender or "AI"))
    safe_text = html.escape(str(message or "")).replace("\n", "<br/>")
    ts = datetime.now().strftime("%H:%M:%S")
    role_color = colors.get(role, colors.get(level, "#dbe4f3"))
    body_color = colors.get(level, "#dbe4f3") if role == "system" else "#dbe4f3"

    # User requested AI-only message feed in panel view.
    show_in_panel = (
        str(role or "").strip().lower() == "assistant"
        or str(sender or "").strip().lower() in {"ai", "assistant"}
    )

    if show_in_panel:
        widget.append(
            f'<span style="color:#7b88a5">[{ts}]</span> '
            f'<span style="color:{role_color};font-weight:600">{safe_sender}:</span> '
            f'<span style="color:{body_color}">{safe_text}</span>'
        )

    hist = getattr(self, "_ai_chat_history", None)
    if not isinstance(hist, list):
        hist = []
        self._ai_chat_history = hist
    hist.append(
        {
            "ts": ts,
            "sender": str(sender),
            "role": str(role),
            "text": str(message or ""),
            "level": str(level),
        }
    )
    if len(hist) > 250:
        del hist[:-250]

    if show_in_panel:
        try:
            sb = widget.verticalScrollBar()
            if sb is not None:
                sb.setValue(sb.maximum())
        except _UI_AI_RECOVERABLE_EXCEPTIONS:
            pass


def _on_ai_chat_send(self: Any) -> None:
    inp = getattr(self, "ai_chat_input", None)
    if inp is None:
        return
    text = str(inp.text() or "").strip()
    if not text:
        return
    inp.clear()

    self._append_ai_chat_message("You", text, role="user")

    # Fast path: deterministic control commands execute immediately on UI thread.
    try:
        handled, reply = self._execute_ai_chat_command(text)
    except Exception as exc:
        self._append_ai_chat_message("System", f"Command failed: {exc}", role="system", level="error")
        return
    if handled:
        self._append_ai_chat_message("AI", reply, role="assistant")
        return

    symbol = self._ui_norm(self.stock_input.text())
    interval = self._normalize_interval_token(self.interval_combo.currentText())
    forecast = int(self.forecast_spin.value())
    lookback = int(self.lookback_spin.value())
    monitor_on = bool(self.monitor and self.monitor.isRunning())
    state = {
        "symbol": symbol or "",
        "interval": interval,
        "forecast_bars": forecast,
        "lookback_bars": lookback,
        "monitoring": "on" if monitor_on else "off",
    }
    history = list(getattr(self, "_ai_chat_history", []) or [])[-20:]

    if any(tok in str(text).lower() for tok in ("news", "policy", "sentiment", "新闻", "政策", "情绪")) and symbol:
        try:
            self._refresh_news_policy_signal(symbol, force=False)
        except _UI_AI_RECOVERABLE_EXCEPTIONS:
            pass

    existing = self.workers.get("ai_chat_query")
    if existing and existing.isRunning():
        self._append_ai_chat_message(
            "System",
            "AI is still processing the previous query. Please wait.",
            role="system",
            level="warning",
        )
        return

    self._append_ai_chat_message("System", "AI is searching the internet and thinking...", role="system", level="info")

    def _work() -> dict[str, Any]:
        return self._generate_ai_chat_reply(
            prompt=text,
            symbol=symbol,
            app_state=state,
            history=history,
        )

    worker = WorkerThread(_work, timeout_seconds=240)
    self._track_worker(worker)

    def _on_done(payload: Any) -> None:
        self.workers.pop("ai_chat_query", None)
        if not isinstance(payload, dict):
            self._append_ai_chat_message("System", "AI reply failed (invalid payload).", role="system", level="error")
            return
        answer = str(payload.get("answer", "") or "").strip()
        action = str(payload.get("action", "") or "").strip()
        if action:
            try:
                handled2, action_msg = self._execute_ai_chat_command(action)
            except Exception as exc:
                answer = f"{answer}\n\n[Action Error] {exc}"
            else:
                if handled2:
                    answer = f"{answer}\n\n[Action] {action_msg}"
                else:
                    answer = f"{answer}\n\n[Action Suggested] {action}"
        self._append_ai_chat_message("AI", answer or "No response.", role="assistant")

    def _on_error(err: str) -> None:
        self.workers.pop("ai_chat_query", None)
        self._append_ai_chat_message("System", f"AI query failed: {err}", role="system", level="error")

    worker.result.connect(_on_done)
    worker.error.connect(_on_error)
    self.workers["ai_chat_query"] = worker
    worker.start()


def _handle_ai_chat_prompt(self: Any, prompt: str) -> str:
    handled, reply = self._execute_ai_chat_command(prompt)
    if handled:
        return reply
    return self._build_ai_chat_response(prompt)


def _execute_ai_chat_command(self: Any, prompt: str) -> tuple[bool, str]:
    p = str(prompt or "").strip()
    low = p.lower()

    if low in {"help", "/help", "commands", "命令", "帮助", "幫助"}:
        return True, (
            "Local AI mode (no API): ask any question and it will use internet/news context. "
            "Commands: analyze <code>, load <code>, start monitoring, stop monitoring, "
            "scan market, refresh sentiment, set interval <1m|5m|15m|30m|60m|1d>, "
            "set forecast <bars>, set lookback <bars>, add watchlist <code>, "
            "remove watchlist <code>, train gm, auto train gm, train llm, auto train llm. "
            "Chinese: 分析 <代码> / 开始监控 / 停止监控 / 刷新情绪 / 周期 5m。"
        )

    if low in {"hi", "hello", "hey", "你好", "您好", "嗨"}:
        return True, (
            "Hi. You can chat naturally and also control the app in plain language. "
            + _chat_state_summary(self)
        )

    if _contains_any(
        low,
        (
            "what can you do",
            "capability",
            "how can you help",
            "你能做什么",
            "可以做什么",
            "怎么控制",
            "如何控制",
        ),
    ):
        return True, (
            "I can chat about market/news/policy context and control the app with natural language. "
            "Examples: 'analyze 600519', 'watch this stock', 'switch to 15 minutes', "
            "'set forecast to 45', 'refresh sentiment', 'scan market'."
        )

    if _contains_any(
        low,
        (
            "status",
            "current state",
            "what are you monitoring",
            "what is current setting",
            "current settings",
            "当前状态",
            "现在状态",
            "你在监控什么",
            "参数状态",
        ),
    ):
        return True, _chat_state_summary(self)

    code = _extract_symbol_or_current(self, p)

    # Conversational monitor control.
    if (
        _contains_any(low, ("monitor", "watch", "track", "监控", "盯盘", "跟踪"))
        and _contains_any(low, ("stop", "pause", "disable", "关闭", "停止", "先别", "取消"))
    ) or _contains_any(low, ("stop monitoring", "停止监控", "关闭监控", "stop monitor")):
        self.monitor_action.setChecked(False)
        self._stop_monitoring()
        return True, "Monitoring stopped."

    if (
        _contains_any(low, ("monitor", "watch", "track", "监控", "盯盘", "跟踪"))
        and _contains_any(low, ("start", "enable", "open", "begin", "resume", "开启", "开始", "打开", "继续"))
    ) or _contains_any(low, ("start monitoring", "开启监控", "开始监控", "打开监控", "start monitor")):
        self.monitor_action.setChecked(True)
        self._start_monitoring()
        return True, "Monitoring started."

    if _contains_any(
        low,
        (
            "scan market",
            "scan for signal",
            "find opportunity",
            "search opportunities",
            "扫描市场",
            "扫市场",
            "全市场扫描",
            "扫描机会",
            "找机会",
        ),
    ):
        self._scan_stocks()
        return True, "Market scan started."

    if _contains_any(
        low,
        (
            "refresh sentiment",
            "refresh news",
            "refresh policy",
            "update sentiment",
            "update news",
            "latest policy",
            "刷新情绪",
            "刷新新闻",
            "刷新政策",
            "更新情绪",
            "更新新闻",
            "更新政策",
        ),
    ):
        self._refresh_sentiment()
        symbol = self._ui_norm(self.stock_input.text())
        if symbol:
            self._refresh_news_policy_signal(symbol, force=True)
        return True, "Sentiment refresh started."

    interval_token = _extract_interval_token_from_text(low)
    if interval_token and _contains_any(
        low,
        (
            "set interval",
            "interval",
            "timeframe",
            "switch to",
            "change to",
            "use",
            "周期",
            "级别",
            "时间框架",
            "切换到",
            "改成",
            "换成",
            "改为",
            "切到",
        ),
    ):
        token = str(interval_token).strip()
        allowed = {"1m", "5m", "15m", "30m", "60m", "1d"}
        if token not in allowed:
            return True, f"Unsupported interval '{token}'."
        self.interval_combo.setCurrentText(token)
        return True, f"Interval set to {token}."

    if _contains_any(
        low,
        (
            "set forecast",
            "forecast",
            "prediction bars",
            "predict bars",
            "horizon",
            "预测",
            "前瞻",
            "预测步数",
            "未来",
        ),
    ):
        m = re.search(r"(\d+)", low)
        if not m:
            return True, "Missing forecast bars value."
        bars = int(m.group(1))
        bars = max(int(self.forecast_spin.minimum()), min(int(self.forecast_spin.maximum()), bars))
        self.forecast_spin.setValue(bars)
        return True, f"Forecast set to {bars} bars."

    if _contains_any(
        low,
        (
            "set lookback",
            "lookback",
            "history window",
            "history length",
            "window size",
            "回看",
            "回溯",
            "历史窗口",
            "历史长度",
            "回看长度",
        ),
    ):
        m = re.search(r"(\d+)", low)
        if not m:
            return True, "Missing lookback bars value."
        bars = int(m.group(1))
        bars = max(int(self.lookback_spin.minimum()), min(int(self.lookback_spin.maximum()), bars))
        self.lookback_spin.setValue(bars)
        return True, f"Lookback set to {bars} bars."

    if code and _contains_any(
        low,
        (
            "analyze",
            "analysis",
            "load",
            "chart",
            "review",
            "look at",
            "check",
            "分析",
            "加载",
            "查看",
            "打开",
            "切到",
            "看看",
            "看下",
            "看一下",
        ),
    ):
        self.stock_input.setText(code)
        self._analyze_stock()
        self._refresh_news_policy_signal(code, force=False)
        return True, f"Analyzing {code}."

    if code and _contains_any(
        low,
        (
            "add watchlist",
            "watchlist add",
            "add to watchlist",
            "put on watchlist",
            "follow",
            "watch this",
            "加入自选",
            "添加自选",
            "加入观察",
            "关注",
        ),
    ):
        self.stock_input.setText(code)
        self._add_to_watchlist()
        return True, f"Added {code} to watchlist."

    if code and _contains_any(
        low,
        (
            "remove watchlist",
            "delete watchlist",
            "remove from watchlist",
            "unfollow",
            "stop watching",
            "移除自选",
            "删除自选",
            "取消关注",
            "移除观察",
        ),
    ):
        row = self._watchlist_row_by_code.get(code)
        if row is not None:
            self.watchlist.selectRow(int(row))
            self._remove_from_watchlist()
            return True, f"Removed {code} from watchlist."
        return True, f"{code} is not in watchlist."

    if _contains_any(
        low,
        (
            "auto train gm",
            "continue learning",
            "auto learn",
            "自动训练gm",
            "自动训练主模型",
            "继续学习",
        ),
    ):
        if hasattr(self, "_show_auto_learn"):
            self._show_auto_learn(auto_start=True)
        return True, "Auto Train GM panel opened and training started."

    if _contains_any(
        low,
        (
            "auto train llm",
            "auto llm",
            "train llm automatically",
            "自动训练llm",
            "自动训练大模型",
            "自动训练聊天模型",
        ),
    ):
        self._auto_train_llm()
        return True, "Auto Train LLM panel opened."

    if _contains_any(
        low,
        (
            "train llm",
            "train chat model",
            "fine tune llm",
            "训练llm",
            "训练大模型",
            "训练聊天模型",
            "微调llm",
        ),
    ):
        self._start_llm_training()
        return True, "LLM training started."

    if _contains_any(
        low,
        ("train gm", "train model", "训练gm", "训练模型", "训练ai", "训练主模型"),
    ):
        self._start_training()
        return True, "Train GM dialog opened."

    return False, ""


def _build_ai_chat_response(self: Any, prompt: str) -> str:
    symbol = self._ui_norm(self.stock_input.text())
    monitor_on = bool(self.monitor and self.monitor.isRunning())
    interval = self._normalize_interval_token(self.interval_combo.currentText())
    forecast = int(self.forecast_spin.value())
    lookback = int(self.lookback_spin.value())

    if hasattr(self, "_news_policy_signal_for"):
        sig = self._news_policy_signal_for(symbol if symbol else "__market__")
    else:
        sig = {}
    overall = float(sig.get("overall", 0.0) or 0.0)
    policy = float(sig.get("policy", 0.0) or 0.0)
    confidence = float(sig.get("confidence", 0.0) or 0.0)

    state = {
        "symbol": symbol or "",
        "interval": interval,
        "forecast_bars": forecast,
        "lookback_bars": lookback,
        "monitoring": "on" if monitor_on else "off",
        "news_policy_signal": {
            "overall": overall,
            "policy": policy,
            "confidence": confidence,
        },
    }

    payload = self._generate_ai_chat_reply(
        prompt=str(prompt or ""),
        symbol=symbol,
        app_state=state,
        history=list(getattr(self, "_ai_chat_history", []) or [])[-20:],
    )
    answer = str(payload.get("answer", "") or "").strip()
    action = str(payload.get("action", "") or "").strip()
    if action:
        handled, action_msg = self._execute_ai_chat_command(action)
        if handled:
            answer = f"{answer}\n\n[Action] {action_msg}"
        else:
            answer = f"{answer}\n\n[Action Suggested] {action}"
    return answer or (
        f"State: symbol={symbol or '--'}, interval={interval}, forecast={forecast}, "
        f"lookback={lookback}, monitoring={'on' if monitor_on else 'off'}. "
        f"News-policy signal: overall={overall:+.2f}, policy={policy:+.2f}, confidence={confidence:.0%}. "
        "Use 'help' to see control commands."
    )


def _generate_ai_chat_reply(
    self: Any,
    *,
    prompt: str,
    symbol: str,
    app_state: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        from data.llm_chat import get_chat_assistant

        assistant = get_chat_assistant()
        payload = assistant.answer(
            prompt=str(prompt or ""),
            symbol=str(symbol or "") or None,
            app_state=dict(app_state or {}),
            history=list(history or []),
            allow_search=True,
        )
        answer = str(payload.get("answer", "") or "").strip()
        local_ready = bool(payload.get("local_model_ready", False))
        if not local_ready:
            answer = (
                f"{answer}\n\n"
                "Tip: no external API is used. "
                "Set `TRADING_LOCAL_LLM_MODEL_PATH` (or `TRADING_LOCAL_LLM_MODEL`) "
                "to enable full on-device chat generation. "
                "Recommended: `Qwen/Qwen2.5-7B-Instruct`."
            ).strip()
        return {
            "answer": answer,
            "action": str(payload.get("action", "") or "").strip(),
            "local_model_ready": local_ready,
        }
    except Exception as exc:
        log.warning("AI chat LLM path failed: %s", exc)
        return {
            "answer": (
                f"Local AI fallback: {exc}. "
                "You can still control the app via commands (type 'help')."
            ),
            "action": "",
            "local_model_ready": False,
        }


def _start_llm_training(self: Any) -> None:
    existing = self.workers.get("llm_train")
    if existing and existing.isRunning():
        self._append_ai_chat_message("System", "LLM training is already running.", role="system", level="warning")
        if hasattr(self, "log"):
            self.log("LLM training is already running.", "warning")
        return

    self._append_ai_chat_message(
        "System",
        "Starting separate LLM hybrid training from recent news corpus...",
        role="system",
        level="info",
    )
    if hasattr(self, "log"):
        self.log("Starting separate LLM hybrid training from recent news corpus...", "info")

    def _work() -> dict[str, Any]:
        from data.llm_sentiment import get_llm_analyzer
        from data.news_collector import get_collector

        analyzer = get_llm_analyzer()
        collector = get_collector()
        articles = collector.collect_news(limit=450, hours_back=120)
        if articles:
            report = dict(analyzer.train(articles, max_samples=1000) or {})
            report.setdefault("collected_articles", int(len(articles)))
            report.setdefault("new_articles", int(len(articles)))
            report.setdefault("collection_mode", "direct_news")
            return report

        report = analyzer.auto_train_from_internet(
            hours_back=120,
            limit_per_query=120,
            max_samples=1000,
            force_china_direct=False,
            only_new=False,
            min_new_articles=1,
            auto_related_search=True,
            allow_gm_bootstrap=False,
        )
        out = dict(report or {})
        out.setdefault("collection_mode", "auto_internet_fallback")
        return out

    worker = WorkerThread(_work, timeout_seconds=1200)
    self._track_worker(worker)

    def _on_done(payload: Any) -> None:
        self.workers.pop("llm_train", None)
        if not isinstance(payload, dict):
            self._append_ai_chat_message("System", "LLM training failed (invalid payload).", role="system", level="error")
            if hasattr(self, "_refresh_model_training_statuses"):
                self._refresh_model_training_statuses()
            return
        status_text = str(payload.get("status", "unknown") or "unknown").strip()
        status = status_text.lower()
        if status in {"trained", "complete", "ok"}:
            level = "success"
        elif status in {"error", "failed"}:
            level = "error"
        else:
            level = "warning"
        msg = (
            f"LLM training complete: status={status_text}, "
            f"samples={payload.get('trained_samples', 0)}, "
            f"zh={payload.get('zh_samples', 0)}, en={payload.get('en_samples', 0)}, "
            f"arch={payload.get('training_architecture', 'hybrid_neural_network')}, "
            f"collected={payload.get('collected_articles', payload.get('new_articles', 0))}, "
            f"mode={payload.get('collection_mode', 'unknown')}."
        )
        notes = str(payload.get("notes", "") or "").strip()
        if notes:
            msg = f"{msg} notes={notes[:200]}"
        self._append_ai_chat_message("System", msg, role="system", level=level)
        if hasattr(self, "log"):
            self.log(msg, level)
        if hasattr(self, "_refresh_model_training_statuses"):
            self._refresh_model_training_statuses()

    def _on_error(err: str) -> None:
        self.workers.pop("llm_train", None)
        self._append_ai_chat_message("System", f"LLM training failed: {err}", role="system", level="error")
        if hasattr(self, "log"):
            self.log(f"LLM training failed: {err}", "error")
        if hasattr(self, "_refresh_model_training_statuses"):
            self._refresh_model_training_statuses()

    worker.result.connect(_on_done)
    worker.error.connect(_on_error)
    self.workers["llm_train"] = worker
    worker.start()


def _refresh_model_training_statuses(self: Any) -> None:
    """Refresh both GM and LLM status labels shown in the left AI panel."""
    llm_status_widget = getattr(self, "llm_status", None)
    llm_info_widget = getattr(self, "llm_info", None)
    if llm_status_widget is None or llm_info_widget is None:
        return
    try:
        from data.llm_sentiment import get_llm_analyzer

        analyzer = get_llm_analyzer()
        status_payload = analyzer.get_training_status()
    except Exception as exc:
        llm_status_widget.setText("LLM Model: Error")
        llm_info_widget.setText(str(exc))
        return

    status = str(status_payload.get("status", "not_trained") or "not_trained").strip().lower()
    architecture = str(
        status_payload.get("training_architecture", "hybrid_neural_network")
        or "hybrid_neural_network"
    )
    trained_samples = int(status_payload.get("trained_samples", 0) or 0)
    finished_at = str(
        status_payload.get("finished_at", status_payload.get("saved_at", "")) or ""
    ).strip()
    finished_short = finished_at[:19].replace("T", " ") if finished_at else ""

    if status in {"trained", "complete", "ok"}:
        llm_status_widget.setText("LLM Model: Trained")
        llm_status_widget.setStyleSheet("color: #35b57c; font-weight: 700;")
    elif status in {"partial"}:
        llm_status_widget.setText("LLM Model: Partially Trained")
        llm_status_widget.setStyleSheet("color: #d8a03a; font-weight: 700;")
    elif status in {"stopped"}:
        llm_status_widget.setText("LLM Model: Stopped")
        llm_status_widget.setStyleSheet("color: #d8a03a; font-weight: 700;")
    elif status in {"error", "failed"}:
        llm_status_widget.setText("LLM Model: Error")
        llm_status_widget.setStyleSheet("color: #e5534b; font-weight: 700;")
    else:
        llm_status_widget.setText("LLM Model: Not trained")
        llm_status_widget.setStyleSheet("")

    info_parts = [architecture]
    if trained_samples > 0:
        info_parts.append(f"samples={trained_samples}")
    if finished_short:
        info_parts.append(f"last={finished_short}")
    llm_info_widget.setText(" | ".join(info_parts))


def _on_llm_training_session_finished(self: Any, payload: dict[str, Any]) -> None:
    data = dict(payload or {})
    status = str(data.get("status", "unknown") or "unknown").strip().lower()
    if status in {"ok", "trained", "complete"}:
        self.log(
            (
                "Auto Train LLM completed: "
                f"collected={data.get('collected_articles', 0)}, "
                f"trained={data.get('trained_samples', 0)}, "
                f"arch={data.get('training_architecture', 'hybrid_neural_network')}"
            ),
            "success",
        )
    elif status == "stopped":
        self.log("Auto Train LLM stopped by user.", "warning")
    elif status in {"error", "failed"}:
        self.log(f"Auto Train LLM failed: {data.get('error', 'unknown error')}", "error")
    if hasattr(self, "_refresh_model_training_statuses"):
        self._refresh_model_training_statuses()


def _show_llm_train_dialog(self: Any, auto_start: bool = False) -> Any | None:
    try:
        from .llm_train_dialog import LLMTrainDialog
    except ImportError as exc:
        self.log(f"Auto Train LLM dialog not available: {exc}", "error")
        return None

    dialog = getattr(self, "_llm_train_dialog", None)
    if dialog is None:
        dialog = LLMTrainDialog(self)
        self._llm_train_dialog = dialog

        def _on_destroyed(*_args: object) -> None:
            self._llm_train_dialog = None

        if hasattr(dialog, "session_finished"):
            dialog.session_finished.connect(self._on_llm_training_session_finished)
        dialog.destroyed.connect(_on_destroyed)

    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    if auto_start and hasattr(dialog, "start_or_resume_auto_train"):
        dialog.start_or_resume_auto_train()
    return dialog


def _auto_train_llm(self: Any) -> None:
    """Open Auto Train LLM control panel (non-modal)."""
    self._show_llm_train_dialog(auto_start=False)
    if hasattr(self, "log"):
        self.log("Auto Train LLM panel opened.", "info")


def _set_news_policy_signal(self: Any, symbol: str, payload: dict[str, Any]) -> None:
    key = self._ui_norm(symbol) if symbol != "__market__" else "__market__"
    if not key:
        return
    cache = getattr(self, "_news_policy_signal_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        self._news_policy_signal_cache = cache
    cache[key] = dict(payload)


def _news_policy_signal_for(self: Any, symbol: str) -> dict[str, Any]:
    cache = getattr(self, "_news_policy_signal_cache", {}) or {}
    key = self._ui_norm(symbol) if symbol and symbol != "__market__" else "__market__"
    out = dict(cache.get(key, {}))
    if not out and key != "__market__":
        out = dict(cache.get("__market__", {}))
    return out


def _refresh_news_policy_signal(self: Any, symbol: str, force: bool = False) -> None:
    code = self._ui_norm(symbol)
    if not code:
        return

    cache = self._news_policy_signal_for(code)
    now = time.time()
    ts = float(cache.get("ts", 0.0) or 0.0)
    if (not force) and ts > 0 and (now - ts) < 300.0:
        return

    worker_name = f"news_policy_signal:{code}"
    existing = self.workers.get(worker_name)
    if existing and existing.isRunning():
        return

    def _work() -> dict[str, Any]:
        from data.llm_sentiment import get_llm_analyzer
        from data.news_aggregator import get_news_aggregator
        from data.news_collector import NewsArticle

        agg = get_news_aggregator()
        news = agg.get_stock_news(code, count=36, force_refresh=bool(force))
        summary = agg.get_sentiment_summary(code, _news=news)
        features = agg.get_news_features(code, hours_lookback=24)

        llm_articles: list[NewsArticle] = []
        now_dt = datetime.now()
        for i, item in enumerate(list(news or [])):
            title = str(getattr(item, "title", "") or "")
            content = str(getattr(item, "content", "") or "")
            text = f"{title} {content}"
            zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
            language = "zh" if zh_chars >= 4 else "en"
            llm_articles.append(
                NewsArticle(
                    id=f"{code}:{i}:{abs(hash(title)) % 10_000_000}",
                    title=title,
                    content=content,
                    summary=content[:180],
                    source=str(getattr(item, "source", "") or ""),
                    url=str(getattr(item, "url", "") or ""),
                    published_at=getattr(item, "publish_time", now_dt) or now_dt,
                    collected_at=now_dt,
                    language=language,
                    category=str(getattr(item, "category", "") or "market"),
                    sentiment_score=float(getattr(item, "sentiment_score", 0.0) or 0.0),
                    relevance_score=float(getattr(item, "importance", 0.5) or 0.5),
                    entities=[code],
                    tags=list(getattr(item, "keywords", []) or []),
                )
            )

        llm = get_llm_analyzer()
        llm_summary = llm.summarize_articles(llm_articles, hours_back=48)

        overall = (0.55 * float(summary.get("overall_sentiment", 0.0) or 0.0)) + (
            0.45 * float(llm_summary.get("overall", 0.0) or 0.0)
        )
        policy = (0.50 * float(features.get("policy_sentiment", 0.0) or 0.0)) + (
            0.50 * float(llm_summary.get("policy", 0.0) or 0.0)
        )
        market = (0.60 * float(summary.get("simple_sentiment", 0.0) or 0.0)) + (
            0.40 * float(llm_summary.get("market", 0.0) or 0.0)
        )
        confidence = (0.65 * float(summary.get("confidence", 0.0) or 0.0)) + (
            0.35 * float(llm_summary.get("confidence", 0.0) or 0.0)
        )

        return {
            "symbol": code,
            "overall": float(max(-1.0, min(1.0, overall))),
            "policy": float(max(-1.0, min(1.0, policy))),
            "market": float(max(-1.0, min(1.0, market))),
            "confidence": float(max(0.0, min(1.0, confidence))),
            "news_count": int(summary.get("total", 0) or 0),
            "ts": float(time.time()),
        }

    worker = WorkerThread(_work, timeout_seconds=90)
    self._track_worker(worker)

    def _on_done(payload: Any) -> None:
        self.workers.pop(worker_name, None)
        if not isinstance(payload, dict):
            return
        self._set_news_policy_signal(code, payload)
        if self._ui_norm(self.stock_input.text()) == code:
            try:
                self._refresh_live_chart_forecast()
            except _UI_AI_RECOVERABLE_EXCEPTIONS:
                pass

    def _on_error(err: str) -> None:
        self.workers.pop(worker_name, None)
        log.debug("news/policy signal refresh failed for %s: %s", code, err)

    worker.result.connect(_on_done)
    worker.error.connect(_on_error)
    self.workers[worker_name] = worker
    worker.start()
