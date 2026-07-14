# LAUA — Local Autonomous Utility Agent

A locally-hosted autonomous AI utility agent for managing an Ubuntu workstation via natural language. Runs fully local with Ollama for LLM inference.

## Key Architecture Rules

- **Ollama runs natively** — systemd service (`ollama.service`), not Docker. Always communicate via HTTP to `http://localhost:11434`. Never call `ollama` CLI directly from tool code.
- **No `shell=True` ever** — all commands use explicit argument arrays via `subprocess`/`pty`.
- **No LangChain** — custom orchestration only.
- **No raw LLM text → shell** — LLM outputs JSON tool calls; execution layer dispatches them.
- **MAX_AGENT_STEPS = 10** — hard ceiling per request, not a soft warning.
- **User config overrides default** — `~/.laua/config.yaml` always wins over `config/default.yaml`. Check both when debugging model/routing issues.

## Project Structure

```
laua/
  config.py          # YAML config loader with env var overrides
  ollama_client.py   # Async HTTP client for Ollama Docker API
  cli.py             # Entry point → Textual UI
  scheduler.py       # /schedule command parsing (pure, no I/O)
  executor/
    pty_session.py   # Stateful pty, $PWD tracking, argument-array only
    safety.py        # Command blacklist + confirmation classifier + shell-metachar guard
    audit.py         # Append-only SQLite audit log
  tools/
    registry.py      # Tool registration + jsonschema validation before dispatch
    core.py          # Phase 1 tools: run_command, get_system_info, read_file
  planner/
    orchestrator.py  # Plan → tool → result loop (MAX_AGENT_STEPS enforced)
  monitor/
    system.py        # psutil snapshot, filters own PID + Ollama container PIDs
  memory/            # SQLite persistence: sessions/messages (store.py), named
                      # workflows (workflows.py), fixed-interval schedules (schedules.py)
  voice/             # Phase 4 slice 2: push-to-talk STT/TTS (audio.py/stt.py/tts.py)
  ui/
    app.py           # Textual TUI
config/
  default.yaml       # Default configuration
tests/
```

## Development Phases

| Phase | Scope | Status |
|-------|-------|--------|
| **1 — Core** | pty executor, safety layer, audit log, Ollama client, basic planner, system monitor, minimal Textual UI | Done |
| **2 — Tools & Memory** | Plugin tool registry, SQLite memory, Docker management, file manager, model routing, context window management | Done |
| **3 — Autonomy** | Multi-step planning, model fallback, dry-run mode, workflow recording, proactive recommendations, background monitor act-and-notify | Done |
| **4 — Extended** | Scheduled tasks (slice 1), voice input/output (slice 2), browser automation, GUI automation | 2 of 4 done — browser/GUI automation not started |

## Running

```bash
pip install -e ".[dev]"
laua
```

`laua` is symlinked to `~/.local/bin/laua` — callable from any terminal without activating the venv.

Ollama runs as a native systemd service. Start/stop: `sudo systemctl start|stop ollama`. Models live at `/usr/share/ollama/.ollama/models/`.

## GPU

- **Hardware**: NVIDIA RTX 4050 6GB VRAM
- **Driver**: 580.142, CUDA 13.0
- **Ollama**: 33/33 layers offloaded to GPU (~5GB VRAM for qwen3.5:4b). ~130 tok/s.
- If GPU stops being used: check `ollama ps` and `nvidia-smi`. Common cause: NVML init failure in a stale service — `sudo systemctl restart ollama`.

## Model Routing (current)

All routes use `qwen3.5:4b` — consistent quality, fits fully on GPU. Routing tiers exist for future differentiation. Active config in `~/.laua/config.yaml`:

```yaml
model_routing:
  fast: "qwen3.5:4b"
  contextual: "qwen3.5:4b"
  reasoning: "qwen3.5:4b"
  coding: "qwen3.5:4b"
  fallback: "qwen3.5:2b"
```

Routing keywords in `laua/planner/router.py`. Network/docker/journal queries route to `reasoning` tier.

## Models on disk (at `/usr/share/ollama/.ollama/models/`)

| Model | Size | Tool calling | Notes |
|-------|------|-------------|-------|
| qwen3.5:4b | 3.4GB | ✓ good | Primary model, full GPU |
| qwen3.5:2b | 2.7GB | ✓ ok | Fallback |
| qwen3.5:0.8b | 1GB | ✓ poor | Hallucinates enum values |
| llama3.2:3b | 2GB | ✓ poor | Passes args as JSON strings |
| mistral:latest | 4.4GB | ✓ ok | Hallucinates subfields |
| qwen3.5:9b | 6.6GB | — | Exceeds VRAM |
| gemma4:e2b | 7.2GB | ✓ smart | Exceeds VRAM, too slow |
| gemma4:e4b | 9.6GB | — | Way too large |
| deepseek-r1:8b | 5.2GB | ✗ | 400 on tool calls — reasoning-only |
| nomic-embed-text | 274MB | — | Embedding only |
| glm-ocr | 2.2GB | — | OCR only |

## Config

`~/.laua/config.yaml` overrides `config/default.yaml`. Environment variables:
- `OLLAMA_BASE_URL` — e.g. `http://localhost:11434`
- `OLLAMA_MODEL` — default model name
- `LAUA_CONFIG` — path to config file

## Phase 2 status — complete (as of this session)

Core tool reliability fixes done this session:
- **JSON coercion**: orchestrator pre-processes string args to native JSON before dispatch
- **Enum normalization**: `RAM→memory`, `processor→cpu`, `storage→disk`, `procs→processes`
- **Safety layer**: read-only cmds (`cat`, `grep`, `head`…) on `/proc` and `/sys` skip sudo gate; `/etc` still requires confirmation
- **System prompt**: strict format rules — max 4 sentences, no code blocks, no command suggestions, conversational gate for greetings

Known remaining issues before Phase 3:
- Model still sometimes summarises tool output with minor fabrications (reports data not in the output)
- Network output (`ip addr`) produces verbose raw text that 4b sometimes misreads — mitigated with `ip -brief` in lingo map
- No streaming indicator for tool steps (only spinner total elapsed shown)

## Phase 4 — scheduled tasks (slice 1, TUI-only)

`/schedule <name> <workflow_name> every <N> <minutes|hours>` fires a previously-`/record`ed workflow on a fixed interval while the app is open, via a second `set_interval` timer (`_scheduler_tick`/`_run_scheduled` in `laua/ui/app.py`), storing state in `SchedulesStore` (`laua/memory/schedules.py`, `~/.laua/schedules.db`). Manage with `/schedules`, `/schedule-enable`, `/schedule-disable`, `/schedule-delete`. Replay dispatches recorded tool calls directly (same primitive as `/run`), auto-confirming since the steps were already reviewed once at record time.

Known issue: `PtySession` (`self._session`) is a single shared instance whose `cwd` is live, mutable state. A scheduled workflow containing a `cd` step changes that shared cwd for the next interactive command too — same pre-existing behavior as manual `/run` and monitor-autonomous-actions, just more likely to surprise since it can fire without the user doing anything.

## Phase 4 — voice input/output (slice 2, opt-in)

Ctrl+T toggles push-to-talk: press once to start recording (via `arecord`, no shell), press again to stop, transcribe locally with faster-whisper (CPU only — Ollama already uses ~5GB of the RTX 4050's 6GB VRAM when active, too tight to share), and submit the text through the same `_submit_prompt`/`_process_request` path typed input uses. Optionally speaks the response back via Piper (`voice.tts.speak_responses`). Toggle-to-talk, not hold-to-talk — terminals don't reliably deliver key-release events. Esc cancels an in-progress recording without submitting.

Fully opt-in and off by default (`voice.enabled: false` in `config/default.yaml`) — `laua/voice/stt.py` defers its `faster_whisper` import into the lazy model-loader so the app still starts with no ML/audio deps installed. Install with `pip install -e ".[voice]"`; needs a Piper voice model downloaded separately into `~/.laua/voices/`.

## Tests

```bash
pytest
```
