# LAUA — Local Autonomous Utility Agent

A locally-hosted autonomous AI utility agent for managing an Ubuntu workstation via natural language. Runs fully local with Ollama for LLM inference.

## Key Architecture Rules

- **Ollama runs in Docker** — always communicate via HTTP to `http://localhost:11434` (configurable). Never call `ollama` CLI.
- **No `shell=True` ever** — all commands use explicit argument arrays via `subprocess`/`pty`.
- **No LangChain** — custom orchestration only.
- **No raw LLM text → shell** — LLM outputs JSON tool calls; execution layer dispatches them.
- **MAX_AGENT_STEPS = 10** — hard ceiling per request, not a soft warning.

## Project Structure

```
laua/
  config.py          # YAML config loader with env var overrides
  ollama_client.py   # Async HTTP client for Ollama Docker API
  cli.py             # Entry point → Textual UI
  executor/
    pty_session.py   # Stateful pty, $PWD tracking, argument-array only
    safety.py        # Command blacklist + confirmation classifier
    audit.py         # Append-only SQLite audit log
  tools/
    registry.py      # Tool registration + jsonschema validation before dispatch
    core.py          # Phase 1 tools: run_command, get_system_info, read_file
  planner/
    orchestrator.py  # Plan → tool → result loop (MAX_AGENT_STEPS enforced)
  monitor/
    system.py        # psutil snapshot, filters own PID + Ollama container PIDs
  memory/            # Phase 2: SQLite session persistence
  ui/
    app.py           # Textual TUI
config/
  default.yaml       # Default configuration
tests/
```

## Development Phases

| Phase | Scope |
|-------|-------|
| **1 — Core** | pty executor, safety layer, audit log, Ollama client, basic planner, system monitor, minimal Textual UI |
| **2 — Tools & Memory** | Plugin tool registry, SQLite memory, Docker management, file manager, model routing, context window management |
| **3 — Autonomy** | Multi-step planning, model fallback, dry-run mode, workflow recording, proactive recommendations |
| **4 — Extended** | Voice, browser automation, GUI automation, scheduled tasks |

## Running

```bash
pip install -e ".[dev]"
laua
```

Ollama must be running in Docker with port 11434 mapped to localhost.

## Config

`~/.laua/config.yaml` overrides `config/default.yaml`. Environment variables:
- `OLLAMA_BASE_URL` — e.g. `http://localhost:11434`
- `OLLAMA_MODEL` — default model name
- `LAUA_CONFIG` — path to config file

## Tests

```bash
pytest
```
