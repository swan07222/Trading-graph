from data.llm_chat import LLMChatAssistant


def test_canonicalize_action_normalizes_common_variants() -> None:
    assert LLMChatAssistant._canonicalize_action("Analyze stock 600519.") == "analyze 600519"
    assert LLMChatAssistant._canonicalize_action("set interval 15 minutes") == "set interval 15m"
    assert LLMChatAssistant._canonicalize_action("set interval 1 hour") == "set interval 60m"
    assert LLMChatAssistant._canonicalize_action("watchlist add 000001") == "add watchlist 000001"
    assert LLMChatAssistant._canonicalize_action("remove watchlist 000001") == "remove watchlist 000001"


def test_canonicalize_action_rejects_unknown_commands() -> None:
    assert LLMChatAssistant._canonicalize_action("buy now 600519") == ""
    assert LLMChatAssistant._canonicalize_action("set interval 2h") == ""
    assert LLMChatAssistant._canonicalize_action("ACTION:") == ""


def test_infer_action_from_prompt_supports_natural_language() -> None:
    assert (
        LLMChatAssistant._infer_action_from_prompt("please analyze 600519 today")
        == "analyze 600519"
    )
    assert (
        LLMChatAssistant._infer_action_from_prompt("could you monitor this stock")
        == "start monitoring"
    )
    assert (
        LLMChatAssistant._infer_action_from_prompt("set interval to 30 minutes")
        == "set interval 30m"
    )
    assert (
        LLMChatAssistant._infer_action_from_prompt("please add 000001 to watchlist")
        == "add watchlist 000001"
    )
    assert (
        LLMChatAssistant._infer_action_from_prompt("please remove 000001 from watchlist")
        == "remove watchlist 000001"
    )


def test_answer_falls_back_to_prompt_inference_when_action_missing(
    monkeypatch,
) -> None:
    assistant = LLMChatAssistant()
    monkeypatch.setattr(
        assistant,
        "_fallback_answer",
        lambda **_: ("fallback", ""),
    )
    monkeypatch.setattr(
        assistant,
        "_run_local_llm",
        lambda **_: ("local", ""),
    )
    out = assistant.answer(
        prompt="analyze 600519",
        symbol=None,
        app_state={},
        history=[],
        allow_search=False,
    )
    assert out["action"] == "analyze 600519"
