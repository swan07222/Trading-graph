"""Enhanced AI Chat Handler for UI Integration.

This module provides the integration layer between the enhanced chat system
and the PyQt6 UI, implementing all safety and reliability improvements.
"""

from __future__ import annotations

import time
from typing import Any

from utils.logger import get_logger
from utils.recoverable import COMMON_RECOVERABLE_EXCEPTIONS

from .app_ai_chat_enhanced import (
    AI_CHAT_QUERY_TIMEOUT,
    COMMAND_QUEUE_PRIORITIES,
    REQUIRES_CONFIRMATION_ACTIONS,
    ChatMessage,
    CommandPriority,
    CommandQueue,
    ConversationManager,
    GracefulDegradation,
    IntentClassifier,
    IntentResult,
    IntentType,
    PromptInjectionDetector,
    contains_any,
    extract_interval_token,
    extract_symbol,
    format_chat_message,
)

log = get_logger(__name__)

# Recoverable exceptions for UI operations
_UI_AI_RECOVERABLE_EXCEPTIONS = COMMON_RECOVERABLE_EXCEPTIONS


class EnhancedChatHandler:
    """
    Enhanced chat handler with safety, reliability, and security improvements.
    
    Features:
    - Intent classification with confidence scoring
    - Command queue with priority handling
    - Confirmation dialogs for critical actions
    - Audit logging integration
    - Undo mechanism for reversible actions
    - Graceful degradation when LLM unavailable
    - Prompt injection detection
    - Access control integration
    """
    
    def __init__(self, ui_controller: Any) -> None:
        self.ui = ui_controller
        self.intent_classifier = IntentClassifier()
        self.command_queue = CommandQueue()
        self.conversation_manager = ConversationManager()
        self.injection_detector = PromptInjectionDetector()
        self.degradation_handler = GracefulDegradation()
        
        # Track pending confirmations
        self._pending_confirmations: dict[str, dict[str, Any]] = {}
        
        # Audit logging
        self._audit_enabled = True
    
    def handle_user_message(self, text: str) -> None:
        """
        Handle user message with full processing pipeline.
        
        Pipeline:
        1. Security check (prompt injection)
        2. Intent classification
        3. Command queuing
        4. Execution or confirmation request
        """
        # Security check
        is_safe, threats = self.injection_detector.is_safe(text)
        if not is_safe:
            self._append_ai_chat_message(
                "System",
                f"Security alert: Potentially unsafe input detected. Threats: {', '.join(threats)}",
                role="system",
                level="error",
            )
            log.warning("Prompt injection attempt detected: %s", threats)
            return
        
        # Add user message to history
        self._append_ai_chat_message("You", text, role="user")
        
        # Intent classification
        intent_result = self.intent_classifier.classify(text)
        
        # Check if LLM is available for complex queries
        if intent_result.intent == IntentType.UNKNOWN and not self.degradation_handler.is_llm_available():
            fallback_response = self.degradation_handler.get_fallback_response(
                intent_result.intent,
                intent_result.entities,
            )
            self._append_ai_chat_message("AI", fallback_response, role="assistant", level="warning")
            return
        
        # Fast path: Execute deterministic commands immediately
        if self._can_execute_immediately(intent_result):
            self._execute_command(text, intent_result)
            return
        
        # Queue command for async execution
        priority = self._get_command_priority(intent_result)
        requires_confirmation = (
            intent_result.requires_confirmation or
            self._action_requires_confirmation(intent_result)
        )
        
        queue_item = self.command_queue.enqueue(
            command=text,
            priority=priority,
            user_id=getattr(self.ui, "_current_user", "default"),
            requires_confirmation=requires_confirmation,
        )
        
        # Request confirmation if needed
        if requires_confirmation:
            self._request_confirmation(queue_item, intent_result)
        else:
            # Execute immediately from queue
            self._execute_queued_command(queue_item, intent_result)
    
    def _can_execute_immediately(self, intent_result: IntentResult) -> bool:
        """Check if command can bypass queue for immediate execution."""
        # High-confidence, low-risk commands can execute immediately
        immediate_intents = {
            IntentType.GREETING,
            IntentType.HELP,
            IntentType.STATUS_QUERY,
            IntentType.CHITCHAT,
        }
        
        return (
            intent_result.intent in immediate_intents and
            intent_result.confidence > 0.7
        )
    
    def _get_command_priority(self, intent_result: IntentResult) -> CommandPriority:
        """Determine command priority based on intent."""
        priority_map = {
            IntentType.CONTROL_MONITOR: CommandPriority.HIGH,
            IntentType.WATCHLIST_MANAGE: CommandPriority.NORMAL,
            IntentType.ANALYZE_STOCK: CommandPriority.NORMAL,
            IntentType.TRAIN_MODEL: CommandPriority.LOW,
            IntentType.SENTIMENT_REFRESH: CommandPriority.LOW,
            IntentType.MARKET_SCAN: CommandPriority.LOW,
        }
        
        return priority_map.get(intent_result.intent, CommandPriority.NORMAL)
    
    def _action_requires_confirmation(self, intent_result: IntentResult) -> bool:
        """Check if action requires user confirmation."""
        # Check intent type
        if intent_result.intent in {IntentType.TRAIN_MODEL, IntentType.WATCHLIST_MANAGE}:
            return True
        
        # Check if command matches known confirmation-required patterns
        command_keywords = {
            "start monitoring", "stop monitoring", "开启监控", "停止监控",
            "add to watchlist", "remove from watchlist", "加入自选", "移除自选",
            "train", "训练",
        }
        
        return contains_any(
            str(intent_result.entities.get("original_text", "")),
            tuple(command_keywords),
        )
    
    def _request_confirmation(self, queue_item: Any, intent_result: IntentResult) -> None:
        """Request user confirmation for action."""
        action_name = self._get_action_name(intent_result)
        confirmation_id = f"confirm_{id(queue_item)}"
        
        # Store pending confirmation
        self._pending_confirmations[confirmation_id] = {
            "queue_item": queue_item,
            "intent_result": intent_result,
            "timestamp": time.time(),
        }
        
        # Show confirmation message
        message = (
            f"Confirm action: {action_name}?\n\n"
            f"This action will be logged for audit purposes.\n\n"
            f"Reply 'yes' or 'confirm' to proceed, 'cancel' to abort."
        )
        
        self._append_ai_chat_message(
            "System",
            message,
            role="system",
            level="warning",
            actionable=True,
            action_payload={
                "type": "confirmation_request",
                "confirmation_id": confirmation_id,
                "action": action_name,
            },
        )
    
    def handle_confirmation(self, confirmation_id: str, confirmed: bool) -> None:
        """Handle user confirmation response."""
        pending = self._pending_confirmations.pop(confirmation_id, None)
        if not pending:
            return
        
        queue_item = pending["queue_item"]
        intent_result = pending["intent_result"]
        
        if confirmed:
            queue_item.confirmed = True
            self._execute_queued_command(queue_item, intent_result)
        else:
            self._append_ai_chat_message(
                "System",
                "Action cancelled by user.",
                role="system",
                level="info",
            )
            self.command_queue.mark_complete(
                queue_item,
                result=None,
                error="User cancelled",
            )
    
    def _execute_queued_command(self, queue_item: Any, intent_result: IntentResult) -> None:
        """Execute command from queue."""
        try:
            result = self._execute_command(
                queue_item.command,
                intent_result,
            )
            self.command_queue.mark_complete(queue_item, result=result)
        except Exception as exc:
            log.error("Command execution failed: %s", exc)
            self.command_queue.mark_complete(queue_item, error=str(exc))
            self._append_ai_chat_message(
                "System",
                f"Command failed: {exc}",
                role="system",
                level="error",
            )
    
    def _execute_command(self, text: str, intent_result: IntentResult) -> Any:
        """
        Execute command based on intent.
        
        This integrates with existing UI methods while adding:
        - Audit logging
        - Parameter validation
        - Undo tracking
        """
        # Extract entities
        entities = intent_result.entities
        current_symbol = getattr(self.ui, "stock_input", None)
        current_symbol_text = current_symbol.text() if current_symbol else ""
        
        # Route to appropriate handler
        handler_map = {
            IntentType.GREETING: self._handle_greeting,
            IntentType.HELP: self._handle_help,
            IntentType.STATUS_QUERY: self._handle_status,
            IntentType.ANALYZE_STOCK: self._handle_analyze,
            IntentType.CONTROL_MONITOR: self._handle_monitor_control,
            IntentType.CONTROL_PARAMETER: self._handle_parameter_control,
            IntentType.WATCHLIST_MANAGE: self._handle_watchlist,
            IntentType.TRAIN_MODEL: self._handle_train,
            IntentType.SENTIMENT_REFRESH: self._handle_sentiment,
            IntentType.MARKET_SCAN: self._handle_market_scan,
            IntentType.EXPLAIN_PREDICTION: self._handle_explain,
            IntentType.UNDO_ACTION: self._handle_undo,
            IntentType.CHITCHAT: self._handle_chitchat,
        }
        
        handler = handler_map.get(intent_result.intent, self._handle_unknown)
        
        # Add original text to entities for context
        entities["original_text"] = text
        entities["current_symbol"] = current_symbol_text
        
        return handler(entities)
    
    def _handle_greeting(self, entities: dict[str, Any]) -> str:
        """Handle greeting intent."""
        response = (
            "Hi. You can chat naturally and also control the app in plain language. "
            f"{self._get_state_summary()}"
        )
        self._append_ai_chat_message("AI", response, role="assistant")
        return response
    
    def _handle_help(self, entities: dict[str, Any]) -> str:
        """Handle help intent."""
        response = (
            "Local AI mode (no API): ask any question and it will use internet/news context.\n\n"
            "Commands:\n"
            "• analyze <code> - Analyze a stock\n"
            "• start/stop monitoring - Toggle monitoring\n"
            "• set interval <1m|5m|15m|30m|1d> - Change timeframe\n"
            "• set forecast <bars> - Set prediction horizon\n"
            "• set lookback <bars> - Set history window\n"
            "• add/remove watchlist <code> - Manage watchlist\n"
            "• train gm/llm - Train models\n"
            "• refresh sentiment - Update sentiment analysis\n"
            "• scan market - Market-wide scan\n"
            "• explain prediction - Show SHAP analysis\n"
            "• undo - Reverse last action\n\n"
            "Chinese: 分析 <代码> / 开始监控 / 停止监控 / 刷新情绪 / 周期 5m"
        )
        self._append_ai_chat_message("AI", response, role="assistant")
        return response
    
    def _handle_status(self, entities: dict[str, Any]) -> str:
        """Handle status query."""
        response = self._get_state_summary()
        self._append_ai_chat_message("AI", response, role="assistant")
        return response
    
    def _handle_analyze(self, entities: dict[str, Any]) -> str:
        """Handle stock analysis request."""
        code = entities.get("stock_code") or entities.get("current_symbol", "")
        if not code:
            response = "Please specify a stock code (6 digits) or say 'analyze this stock'."
            self._append_ai_chat_message("AI", response, role="assistant", level="warning")
            return response
        
        # Validate stock code
        if not re.match(r"^\d{6}$", code):
            response = f"Invalid stock code '{code}'. Please use 6-digit code (e.g., 600519)."
            self._append_ai_chat_message("AI", response, role="assistant", level="error")
            return response
        
        # Execute analysis
        try:
            if hasattr(self.ui, "stock_input"):
                self.ui.stock_input.setText(code)
            if hasattr(self.ui, "_analyze_stock"):
                self.ui._analyze_stock()
            if hasattr(self.ui, "_refresh_news_policy_signal"):
                self.ui._refresh_news_policy_signal(code, force=False)
            
            # Record for undo
            self.conversation_manager.record_action(
                f"analyze_{code}",
                reversible=True,
                payload={"previous_symbol": entities.get("current_symbol")},
            )
            
            # Audit log
            self._log_audit_event("ANALYZE_STOCK", {"code": code})
            
            response = f"Analyzing {code}..."
            self._append_ai_chat_message("AI", response, role="assistant")
            return response
        except Exception as exc:
            response = f"Analysis failed: {exc}"
            self._append_ai_chat_message("System", response, role="system", level="error")
            return response
    
    def _handle_monitor_control(self, entities: dict[str, Any]) -> str:
        """Handle monitoring control (start/stop)."""
        text = entities.get("original_text", "").lower()
        
        # Determine action
        stop_keywords = ("stop", "pause", "disable", "关闭", "停止", "先别", "取消")
        start_keywords = ("start", "enable", "open", "begin", "resume", "开启", "开始", "打开", "继续")
        
        is_stop = contains_any(text, stop_keywords) or "stop monitoring" in text
        is_start = contains_any(text, start_keywords) or "start monitoring" in text
        
        if is_stop:
            if hasattr(self.ui, "monitor_action"):
                self.ui.monitor_action.setChecked(False)
            if hasattr(self.ui, "_stop_monitoring"):
                self.ui._stop_monitoring()
            
            self.conversation_manager.record_action("stop_monitoring", reversible=True)
            self._log_audit_event("STOP_MONITORING", {})
            
            response = "Monitoring stopped."
        elif is_start:
            if hasattr(self.ui, "monitor_action"):
                self.ui.monitor_action.setChecked(True)
            if hasattr(self.ui, "_start_monitoring"):
                self.ui._start_monitoring()
            
            self.conversation_manager.record_action("start_monitoring", reversible=True)
            self._log_audit_event("START_MONITORING", {})
            
            response = "Monitoring started."
        else:
            response = "Please specify 'start monitoring' or 'stop monitoring'."
        
        self._append_ai_chat_message("AI", response, role="assistant")
        return response
    
    def _handle_parameter_control(self, entities: dict[str, Any]) -> str:
        """Handle parameter adjustment (interval, forecast, lookback)."""
        text = entities.get("original_text", "").lower()
        
        # Interval control
        if contains_any(text, ("interval", "timeframe", "周期", "级别")):
            interval_token = extract_interval_token(text)
            if not interval_token:
                return "Please specify interval (1m, 5m, 15m, 30m, 60m, 1d)."
            
            allowed = {"1m", "5m", "15m", "30m", "60m", "1d"}
            if interval_token not in allowed:
                response = f"Unsupported interval '{interval_token}'. Valid: {', '.join(allowed)}"
                self._append_ai_chat_message("AI", response, role="assistant", level="error")
                return response
            
            # Validate before setting
            old_interval = getattr(self.ui, "interval_combo", None)
            old_value = old_interval.currentText() if old_interval else ""
            
            if hasattr(self.ui, "interval_combo"):
                self.ui.interval_combo.setCurrentText(interval_token)
            
            self.conversation_manager.record_action(
                f"set_interval_{interval_token}",
                reversible=True,
                payload={"previous_interval": old_value},
            )
            self._log_audit_event("SET_INTERVAL", {"interval": interval_token})
            
            response = f"Interval set to {interval_token}."
            self._append_ai_chat_message("AI", response, role="assistant")
            return response
        
        # Forecast control
        if contains_any(text, ("forecast", "prediction", "预测")):
            numbers = entities.get("numbers", [])
            if not numbers:
                return "Please specify forecast bars value."
            
            bars = numbers[0]
            
            # Validate range before clamping
            forecast_spin = getattr(self.ui, "forecast_spin", None)
            if forecast_spin:
                min_val = int(forecast_spin.minimum())
                max_val = int(forecast_spin.maximum())
                
                if bars < min_val or bars > max_val:
                    response = (
                        f"Value {bars} is outside valid range [{min_val}, {max_val}]. "
                        f"Clamping to nearest valid value."
                    )
                    self._append_ai_chat_message("AI", response, role="assistant", level="warning")
                
                bars = max(min_val, min(max_val, bars))
                forecast_spin.setValue(bars)
            
            self._log_audit_event("SET_FORECAST", {"bars": bars})
            
            response = f"Forecast set to {bars} bars."
            self._append_ai_chat_message("AI", response, role="assistant")
            return response
        
        # Lookback control
        if contains_any(text, ("lookback", "history", "回看", "回溯")):
            numbers = entities.get("numbers", [])
            if not numbers:
                return "Please specify lookback bars value."
            
            bars = numbers[0]
            
            lookback_spin = getattr(self.ui, "lookback_spin", None)
            if lookback_spin:
                min_val = int(lookback_spin.minimum())
                max_val = int(lookback_spin.maximum())
                
                if bars < min_val or bars > max_val:
                    response = (
                        f"Value {bars} is outside valid range [{min_val}, {max_val}]. "
                        f"Clamping to nearest valid value."
                    )
                    self._append_ai_chat_message("AI", response, role="assistant", level="warning")
                
                bars = max(min_val, min(max_val, bars))
                lookback_spin.setValue(bars)
            
            self._log_audit_event("SET_LOOKBACK", {"bars": bars})
            
            response = f"Lookback set to {bars} bars."
            self._append_ai_chat_message("AI", response, role="assistant")
            return response
        
        return "Unknown parameter. Use 'set interval', 'set forecast', or 'set lookback'."
    
    def _handle_watchlist(self, entities: dict[str, Any]) -> str:
        """Handle watchlist management."""
        text = entities.get("original_text", "").lower()
        code = entities.get("stock_code") or entities.get("current_symbol", "")
        
        if not code:
            return "Please specify a stock code."
        
        is_add = contains_any(text, ("add", "follow", "加入", "添加", "关注"))
        is_remove = contains_any(text, ("remove", "delete", "unfollow", "移除", "取消"))
        
        if is_add:
            if hasattr(self.ui, "stock_input"):
                self.ui.stock_input.setText(code)
            if hasattr(self.ui, "_add_to_watchlist"):
                self.ui._add_to_watchlist()
            
            self.conversation_manager.record_action(
                f"add_watchlist_{code}",
                reversible=True,
                payload={"code": code},
            )
            self._log_audit_event("ADD_WATCHLIST", {"code": code})
            
            response = f"Added {code} to watchlist."
        elif is_remove:
            if hasattr(self.ui, "_watchlist_row_by_code"):
                row = self.ui._watchlist_row_by_code.get(code)
                if row is not None:
                    if hasattr(self.ui, "watchlist"):
                        self.ui.watchlist.selectRow(int(row))
                    if hasattr(self.ui, "_remove_from_watchlist"):
                        self.ui._remove_from_watchlist()
                    
                    self.conversation_manager.record_action(
                        f"remove_watchlist_{code}",
                        reversible=False,  # Can't auto-re-add without knowing position
                        payload={"code": code},
                    )
                    self._log_audit_event("REMOVE_WATCHLIST", {"code": code})
                    
                    response = f"Removed {code} from watchlist."
                else:
                    response = f"{code} is not in watchlist."
            else:
                response = "Watchlist management not available."
        else:
            response = "Please specify 'add to watchlist' or 'remove from watchlist'."
        
        self._append_ai_chat_message("AI", response, role="assistant")
        return response
    
    def _handle_train(self, entities: dict[str, Any]) -> str:
        """Handle model training request."""
        text = entities.get("original_text", "").lower()
        
        is_llm = contains_any(text, ("llm", "chat", "大模型", "聊天模型"))
        is_auto = contains_any(text, ("auto", "自动"))
        
        if is_llm:
            if hasattr(self.ui, "_auto_train_llm") and is_auto:
                self.ui._auto_train_llm()
                response = "Auto Train LLM panel opened."
            elif hasattr(self.ui, "_start_llm_training"):
                self.ui._start_llm_training()
                response = "LLM training started."
            else:
                response = "LLM training not available."
        else:
            # GM training
            if hasattr(self.ui, "_start_training"):
                self.ui._start_training()
                response = "Train GM dialog opened."
            elif hasattr(self.ui, "_show_auto_learn"):
                self.ui._show_auto_learn(auto_start=True)
                response = "Auto Train GM panel opened and training started."
            else:
                response = "GM training not available."
        
        self._log_audit_event("TRAIN_MODEL", {"type": "LLM" if is_llm else "GM", "auto": is_auto})
        self._append_ai_chat_message("AI", response, role="assistant")
        return response
    
    def _handle_sentiment(self, entities: dict[str, Any]) -> str:
        """Handle sentiment refresh request."""
        try:
            if hasattr(self.ui, "_refresh_sentiment"):
                self.ui._refresh_sentiment()
            
            current_symbol = entities.get("current_symbol", "")
            if current_symbol and hasattr(self.ui, "_refresh_news_policy_signal"):
                self.ui._refresh_news_policy_signal(current_symbol, force=True)
            
            self._log_audit_event("REFRESH_SENTIMENT", {"symbol": current_symbol})
            
            response = "Sentiment refresh started."
        except Exception as exc:
            response = f"Sentiment refresh failed: {exc}"
        
        self._append_ai_chat_message("AI", response, role="assistant")
        return response
    
    def _handle_market_scan(self, entities: dict[str, Any]) -> str:
        """Handle market scan request."""
        try:
            if hasattr(self.ui, "_scan_stocks"):
                self.ui._scan_stocks()
            
            self._log_audit_event("MARKET_SCAN", {})
            
            response = "Market scan started."
        except Exception as exc:
            response = f"Market scan failed: {exc}"
        
        self._append_ai_chat_message("AI", response, role="assistant")
        return response
    
    def _handle_explain(self, entities: dict[str, Any]) -> str:
        """Handle prediction explanation request with SHAP analysis."""
        try:
            # Import explainability module
            from models.explainability import get_explainer
            
            current_symbol = entities.get("current_symbol", "")
            if not current_symbol:
                response = "Please select a stock first to explain its prediction."
                self._append_ai_chat_message("AI", response, role="assistant", level="warning")
                return response
            
            explainer = get_explainer()
            
            # Get latest prediction explanation
            explanation = explainer.explain_prediction(
                symbol=current_symbol,
                include_shap=True,
            )
            
            if explanation:
                response = (
                    f"**Prediction Explanation for {current_symbol}**\n\n"
                    f"{explanation.explanation_text}\n\n"
                    f"Top positive factors: {explanation.top_positive_features[:3]}\n"
                    f"Top negative factors: {explanation.top_negative_features[:3]}"
                )
            else:
                response = "No prediction available to explain. Please analyze the stock first."
            
            self._append_ai_chat_message("AI", response, role="assistant")
            return response
        except ImportError:
            response = "Explainability module not available. Please ensure models.explainability is installed."
        except Exception as exc:
            response = f"Explanation failed: {exc}"
        
        self._append_ai_chat_message("System", response, role="system", level="error")
        return response
    
    def _handle_undo(self, entities: dict[str, Any]) -> str:
        """Handle undo request for last reversible action."""
        last_action = self.conversation_manager.get_last_reversible_action()
        
        if not last_action:
            response = "No reversible actions to undo."
            self._append_ai_chat_message("AI", response, role="assistant", level="info")
            return response
        
        action = last_action.get("action", "")
        payload = last_action.get("payload", {})
        
        # Implement undo logic based on action type
        if action.startswith("set_interval"):
            old_interval = payload.get("previous_interval", "1d")
            if hasattr(self.ui, "interval_combo"):
                self.ui.interval_combo.setCurrentText(old_interval)
            response = f"Undone: Interval reverted to {old_interval}."
        elif action.startswith("analyze"):
            old_symbol = payload.get("previous_symbol", "")
            if old_symbol and hasattr(self.ui, "stock_input"):
                self.ui.stock_input.setText(old_symbol)
            response = f"Undone: Analysis reverted to {old_symbol or 'none'}."
        elif action in {"start_monitoring", "stop_monitoring"}:
            # Toggle monitoring
            if hasattr(self.ui, "monitor_action"):
                current_state = self.ui.monitor_action.isChecked()
                self.ui.monitor_action.setChecked(not current_state)
            if action == "start_monitoring" and hasattr(self.ui, "_stop_monitoring"):
                self.ui._stop_monitoring()
            elif action == "stop_monitoring" and hasattr(self.ui, "_start_monitoring"):
                self.ui._start_monitoring()
            response = f"Undone: Monitoring state reverted."
        else:
            response = f"Cannot undo action: {action}"
        
        self._log_audit_event("UNDO_ACTION", {"original_action": action})
        self._append_ai_chat_message("AI", response, role="assistant")
        return response
    
    def _handle_chitchat(self, entities: dict[str, Any]) -> str:
        """Handle casual conversation."""
        # Try LLM response first
        try:
            from data.llm_sentiment import get_llm_analyzer
            
            analyzer = get_llm_analyzer()
            history = self.conversation_manager.get_history(limit=20)
            
            payload = analyzer.generate_response(
                prompt=entities.get("original_text", ""),
                symbol=entities.get("current_symbol"),
                app_state=self._get_app_state(),
                history=history,
            )
            
            answer = str(payload.get("answer", "")).strip()
            if answer:
                self._append_ai_chat_message("AI", answer, role="assistant")
                return answer
        except Exception:
            pass
        
        # Fallback
        response = "I'm here to help with stock analysis and trading. Ask me about market trends, sentiment, or use commands like 'analyze <code>'."
        self._append_ai_chat_message("AI", response, role="assistant")
        return response
    
    def _handle_unknown(self, entities: dict[str, Any]) -> str:
        """Handle unknown intent."""
        # Try LLM fallback
        try:
            from data.llm_sentiment import get_llm_analyzer
            
            analyzer = get_llm_analyzer()
            history = self.conversation_manager.get_history(limit=20)
            
            payload = analyzer.generate_response(
                prompt=entities.get("original_text", ""),
                symbol=entities.get("current_symbol"),
                app_state=self._get_app_state(),
                history=history,
            )
            
            answer = str(payload.get("answer", "")).strip()
            action = str(payload.get("action", "")).strip()
            
            if action:
                # Execute suggested action
                self.handle_user_message(action)
            
            if answer:
                self._append_ai_chat_message("AI", answer, role="assistant")
                return answer
        except Exception:
            pass
        
        # Graceful degradation
        if not self.degradation_handler.is_llm_available():
            response = self.degradation_handler.get_fallback_response(
                IntentType.UNKNOWN,
                entities,
            )
            self._append_ai_chat_message("AI", response, role="assistant", level="warning")
            return response
        
        response = (
            f"I'm not sure I understand. You can ask about stocks, news, sentiment, "
            f"or use commands. Type 'help' for available commands.\n\n"
            f"Current state: {self._get_state_summary()}"
        )
        self._append_ai_chat_message("AI", response, role="assistant", level="info")
        return response
    
    def _get_state_summary(self) -> str:
        """Get current app state summary."""
        symbol = getattr(self.ui, "stock_input", None)
        symbol_text = symbol.text() if symbol else "--"
        
        interval_combo = getattr(self.ui, "interval_combo", None)
        interval_text = interval_combo.currentText() if interval_combo else "--"
        
        forecast_spin = getattr(self.ui, "forecast_spin", None)
        forecast_value = forecast_spin.value() if forecast_spin else 0
        
        lookback_spin = getattr(self.ui, "lookback_spin", None)
        lookback_value = lookback_spin.value() if lookback_spin else 0
        
        monitor = getattr(self.ui, "monitor", None)
        monitor_on = monitor.isRunning() if monitor else False
        
        return (
            f"Current state: symbol={symbol_text or '--'}, interval={interval_text or '--'}, "
            f"forecast={forecast_value} bars, lookback={lookback_value} bars, "
            f"monitoring={'on' if monitor_on else 'off'}."
        )
    
    def _get_app_state(self) -> dict[str, Any]:
        """Get app state for LLM context."""
        return {
            "symbol": getattr(self.ui, "stock_input", None).text() if hasattr(self.ui, "stock_input") else "",
            "interval": getattr(self.ui, "interval_combo", None).currentText() if hasattr(self.ui, "interval_combo") else "",
            "forecast_bars": getattr(self.ui, "forecast_spin", None).value() if hasattr(self.ui, "forecast_spin") else 0,
            "lookback_bars": getattr(self.ui, "lookback_spin", None).value() if hasattr(self.ui, "lookback_spin") else 0,
            "monitoring": "on" if hasattr(self.ui, "monitor") and self.ui.monitor.isRunning() else "off",
        }
    
    def _append_ai_chat_message(
        self,
        sender: str,
        message: str,
        role: str = "assistant",
        level: str = "info",
        actionable: bool = False,
        action_payload: dict[str, Any] | None = None,
    ) -> None:
        """Append message to chat view."""
        widget = getattr(self.ui, "ai_chat_view", None)
        if widget is None:
            return
        
        formatted = format_chat_message(sender, message, role, level)
        if formatted:
            widget.append(formatted)
        
        # Add to conversation manager
        self.conversation_manager.add_message(
            sender=sender,
            text=message,
            role=role,
            level=level,
            actionable=actionable,
            action_payload=action_payload,
        )
        
        # Auto-scroll
        try:
            sb = widget.verticalScrollBar()
            if sb is not None:
                sb.setValue(sb.maximum())
        except _UI_AI_RECOVERABLE_EXCEPTIONS:
            pass
    
    def _log_audit_event(self, event_type: str, details: dict[str, Any]) -> None:
        """Log audit event for chat-triggered actions."""
        if not self._audit_enabled:
            return
        
        try:
            from utils.security import get_audit_log
            
            audit = get_audit_log()
            user = getattr(self.ui, "_current_user", "chat_user")
            
            audit.log(
                event=event_type,
                user=user,
                details=details,
            )
        except Exception as exc:
            log.debug("Audit logging failed: %s", exc)


# Import for re-export
import re  # noqa: E402
