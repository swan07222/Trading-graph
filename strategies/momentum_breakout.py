"""
Example strategy script for StrategyScriptEngine.
"""


def strategy_meta():
    return {
        "name": "Momentum Breakout",
        "version": "1.0",
        "description": "Simple EMA/MACD momentum filter with RSI guardrails.",
    }


def generate_signal(df, indicators, context):
    rsi = float(indicators.get("rsi_14", 50.0))
    ema9 = float(indicators.get("ema_9", 0.0))
    ema21 = float(indicators.get("ema_21", 0.0))
    macd_hist = float(indicators.get("macd_hist", 0.0))
    close = float(indicators.get("close", 0.0))

    if close <= 0:
        return {"action": "hold", "score": 0.0, "reason": ""}

    if ema9 > ema21 and macd_hist > 0 and 45 <= rsi <= 72:
        return {
            "action": "buy",
            "score": 0.75,
            "reason": "EMA9 above EMA21 with positive momentum",
        }

    if ema9 < ema21 and macd_hist < 0 and (rsi >= 65 or rsi <= 35):
        return {
            "action": "sell",
            "score": 0.70,
            "reason": "EMA9 below EMA21 with negative momentum",
        }

    return {"action": "hold", "score": 0.0, "reason": ""}
