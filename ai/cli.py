#!/usr/bin/env python3
"""CLI interface for local LLM chat and trading control.

Usage:
    python -m ai.cli                    # Interactive chat mode
    python -m ai.cli --command "Buy 100 AAPL"  # Single command
    python -m ai.cli --chat "What's the market outlook?"  # Single chat
    python -m ai.cli --health           # System health check
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click

from config.settings import CONFIG
from utils.logger import get_logger

log = get_logger(__name__)


@click.group(invoke_without_command=True)
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.option("--model", "-m", default="self-chat-transformer", help="LLM model name")
@click.option(
    "--backend",
    "-b",
    default="transformers_local",
    type=click.Choice(["transformers_local", "ollama", "llama_cpp", "vllm"]),
    help="LLM backend",
)
@click.option("--host", default="127.0.0.1", help="LLM server host")
@click.option("--port", "-p", default=11434, help="LLM server port")
@click.option("--security", "-s", default="high", type=click.Choice(["none", "low", "medium", "high", "maximum"]), help="Security level")
@click.option("--quiet", "-q", is_flag=True, help="Quiet mode (minimal output)")
@click.pass_context
def cli(ctx, config, model, backend, host, port, security, quiet):
    """Local LLM Chat and Trading Control System.
    
    A production-grade AI assistant for stock trading with:
    - Fully offline LLM inference
    - Deterministic command parsing
    - Safety validation and confirmations
    - Comprehensive audit logging
    - RAG-based knowledge grounding
    - Prompt injection protection
    """
    ctx.ensure_object(dict)
    ctx.obj["config_file"] = config
    ctx.obj["model"] = model
    ctx.obj["backend"] = backend
    ctx.obj["host"] = host
    ctx.obj["port"] = port
    ctx.obj["security"] = security
    ctx.obj["quiet"] = quiet
    
    # Initialize AI controller if no subcommand
    if ctx.invoked_subcommand is None:
        ctx.invoke(interactive)


@cli.command()
@click.pass_context
def interactive(ctx):
    """Start interactive chat mode."""
    config_file = ctx.obj.get("config_file")
    model = ctx.obj.get("model", "self-chat-transformer")
    backend = ctx.obj.get("backend", "transformers_local")
    host = ctx.obj.get("host", "127.0.0.1")
    port = ctx.obj.get("port", 11434)
    security = ctx.obj.get("security", "high")
    
    print("=" * 60)
    print("Local LLM Chat & Trading Control System")
    print("=" * 60)
    print(f"Model: {model} ({backend})")
    print(f"Server: {host}:{port}")
    print(f"Security: {security}")
    print("=" * 60)
    print("Commands:")
    print("  /help     - Show help")
    print("  /status   - System status")
    print("  /clear    - Clear conversation")
    print("  /export   - Export conversation")
    print("  /quit     - Exit")
    print("=" * 60)
    print()
    
    async def run_interactive():
        from ai.controller import AIController, ChatConfig
        
        # Create config
        chat_config = ChatConfig(
            llm_model=model,
            llm_backend=backend,
            llm_host=host,
            llm_port=port,
            security_level=security,
        )
        
        # Initialize controller
        controller = AIController(chat_config)
        await controller.initialize()
        
        print("AI Controller initialized. Start chatting!\n")
        
        while True:
            try:
                user_input = click.prompt("You", type=str)
            except (EOFError, KeyboardInterrupt):
                print("\n")
                break
            
            user_input = user_input.strip()
            
            if not user_input:
                continue
            
            # Handle slash commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]
                
                if cmd in ["/quit", "/exit", "/q"]:
                    break
                elif cmd == "/help":
                    print_help()
                    continue
                elif cmd == "/status":
                    stats = controller.get_stats()
                    click.echo(json.dumps(stats, indent=2, default=str))
                    continue
                elif cmd == "/clear":
                    controller.clear_conversation()
                    print("Conversation cleared.")
                    continue
                elif cmd == "/export":
                    export_path = Path(CONFIG.logs_dir) / f"conversation_{model.replace(':', '_')}.json"
                    # TODO: Implement export
                    print(f"Export to: {export_path}")
                    continue
                else:
                    print(f"Unknown command: {cmd}")
                    continue
            
            # Process message
            print("\nAI: ", end="", flush=True)
            
            try:
                response = await controller.chat(user_input)
                
                # Print response
                print(response.response_text)
                
                # Show command info if detected
                if response.command:
                    print(f"\n[Detected Command: {response.command.command_type.name}]")
                    if response.requires_confirmation:
                        print("[Confirmation Required]")
                    
                    if response.safety_report:
                        if response.safety_report.is_approved():
                            print("[Safety Check: PASSED]")
                        else:
                            print(f"[Safety Check: BLOCKED] {response.safety_report.blocked_reasons}")
                
                # Show sources if RAG used
                if response.sources:
                    print(f"\n[Sources: {', '.join(response.sources)}]")
                
                print(f"\n[Latency: {response.latency_ms:.0f}ms | Tokens: {response.tokens_used}]")
                print()
                
            except Exception as e:
                print(f"\nError: {e}")
                print()
        
        await controller.shutdown()
    
    asyncio.run(run_interactive())


@cli.command()
@click.argument("message")
@click.option("--confirm", is_flag=True, help="Auto-confirm commands")
@click.pass_context
def chat(ctx, message, confirm):
    """Send a single chat message."""
    model = ctx.obj.get("model", "self-chat-transformer")
    backend = ctx.obj.get("backend", "transformers_local")
    host = ctx.obj.get("host", "127.0.0.1")
    port = ctx.obj.get("port", 11434)
    security = ctx.obj.get("security", "high")
    
    async def run_chat():
        from ai.controller import AIController, ChatConfig
        
        chat_config = ChatConfig(
            llm_model=model,
            llm_backend=backend,
            llm_host=host,
            llm_port=port,
            security_level=security,
        )
        
        controller = AIController(chat_config)
        await controller.initialize()
        
        response = await controller.chat(message)
        
        print(response.response_text)
        
        if response.command:
            print(f"\n[Command: {response.command.command_type.name}]")
            if response.requires_confirmation and not confirm:
                print("[Waiting for confirmation...]")
            elif confirm:
                result = await controller.execute_command(response.command, confirmation=True)
                print(f"\n[Result: {json.dumps(result, indent=2)}]")
        
        await controller.shutdown()
    
    asyncio.run(run_chat())


@cli.command()
@click.argument("command_text")
@click.option("--confirm", "-y", is_flag=True, help="Auto-confirm")
@click.pass_context
def command(ctx, command_text, confirm):
    """Execute a trading command directly."""
    model = ctx.obj.get("model", "self-chat-transformer")
    backend = ctx.obj.get("backend", "transformers_local")
    host = ctx.obj.get("host", "127.0.0.1")
    port = ctx.obj.get("port", 11434)
    
    async def run_command():
        from ai.controller import AIController, ChatConfig
        from ai.command_parser import get_parser
        
        chat_config = ChatConfig(
            llm_model=model,
            llm_backend=backend,
            llm_host=host,
            llm_port=port,
        )
        
        controller = AIController(chat_config)
        await controller.initialize()
        
        # Parse command
        parser = get_parser()
        parsed = parser.parse(command_text)
        
        print(f"Parsed Command: {parsed.command_type.name}")
        print(f"Parameters: {parsed.parameters}")
        print(f"Requires Confirmation: {parsed.requires_confirmation}")
        print()
        
        # Execute
        result = await controller.execute_command(parsed, confirmation=confirm)
        print(f"Result: {json.dumps(result, indent=2)}")
        
        await controller.shutdown()
    
    asyncio.run(run_command())


@cli.command()
@click.pass_context
def health(ctx):
    """Check system health."""
    model = ctx.obj.get("model", "self-chat-transformer")
    backend = ctx.obj.get("backend", "transformers_local")
    host = ctx.obj.get("host", "127.0.0.1")
    port = ctx.obj.get("port", 11434)
    
    async def run_health():
        from ai.controller import AIController, ChatConfig
        
        chat_config = ChatConfig(
            llm_model=model,
            llm_backend=backend,
            llm_host=host,
            llm_port=port,
        )
        
        controller = AIController(chat_config)
        await controller.initialize()
        
        # Get health
        from ai.command_parser import CommandType, get_parser
        
        parser = get_parser()
        health_cmd = parser.parse("health check")
        
        result = await controller.execute_command(health_cmd)
        
        print("System Health Check")
        print("=" * 40)
        print(json.dumps(result, indent=2, default=str))
        
        await controller.shutdown()
    
    asyncio.run(run_health())


@cli.command("train-self-chat")
@click.option("--chat-history", default="data/chat_history/chat_history.json", help="Chat history JSON path")
@click.option("--epochs", default=2, type=int, help="Training epochs")
@click.option("--max-steps", default=1200, type=int, help="Max optimizer steps")
def train_self_chat(chat_history, epochs, max_steps):
    """Train a self-owned local transformer chat model."""
    from ai.self_chat_trainer import SelfChatTrainingConfig, train_self_chat_model

    cfg = SelfChatTrainingConfig.from_defaults()
    cfg.chat_history_path = Path(str(chat_history))
    cfg.epochs = max(1, int(epochs))
    cfg.max_steps = max(100, int(max_steps))

    print("Training self chat transformer...")
    print(f"- chat history: {cfg.chat_history_path}")
    print(f"- output dir: {cfg.output_dir}")
    print(f"- epochs: {cfg.epochs}")
    print(f"- max steps: {cfg.max_steps}")
    print("")

    report = train_self_chat_model(cfg)
    print(json.dumps(report, indent=2, ensure_ascii=False))


@cli.command()
@click.option("--model", default="qwen2.5:7b", help="Model to pull")
def pull(model):
    """Pull a model from Ollama."""
    import httpx
    
    async def run_pull():
        print(f"Pulling model: {model}")
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    "http://127.0.0.1:11434/api/pull",
                    json={"name": model},
                ) as response:
                    async for line in response.aiter_lines():
                        try:
                            data = json.loads(line)
                            if "status" in data:
                                print(data["status"])
                        except json.JSONDecodeError:
                            pass
        except httpx.ConnectError:
            print("Error: Cannot connect to Ollama. Is it running?")
            print("Start with: ollama serve")
    
    asyncio.run(run_pull())


@cli.command()
def models():
    """List available models."""
    import httpx
    
    async def run_models():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://127.0.0.1:11434/api/tags")
                
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    
                    print("Available Models:")
                    print("=" * 40)
                    
                    for m in models:
                        name = m.get("name", "unknown")
                        size = m.get("size", 0) / (1024**3)  # GB
                        print(f"  {name} ({size:.1f} GB)")
                else:
                    print(f"Error: {response.status_code}")
                    
        except httpx.ConnectError:
            print("Error: Cannot connect to Ollama.")
    
    asyncio.run(run_models())


def print_help():
    """Print interactive mode help."""
    print("""
Interactive Mode Commands:
  /help     - Show this help
  /status   - Show system status
  /clear    - Clear conversation history
  /export   - Export conversation to file
  /quit     - Exit the application

Chat Examples:
  "What's the current price of AAPL?"
  "Predict TSLA for next week"
  "Show me sentiment for NVDA"
  "Analyze stock 600519"

Trading Commands:
  "Buy 100 shares of AAPL at $150"
  "Sell 50 shares of MSFT"
  "Get quote for GOOGL"
""")


if __name__ == "__main__":
    cli()
