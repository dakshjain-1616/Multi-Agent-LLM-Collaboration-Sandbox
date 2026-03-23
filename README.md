[![Try NEO in VS Code](https://img.shields.io/badge/VS%20Code-Try%20NEO-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)
<div align="center">

# 🤖 Multi-Agent LLM Collaboration Sandbox

**Orchestrate multiple AI agents. Watch them debate, plan, create, and think — together.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Gradio](https://img.shields.io/badge/gradio-4.0+-orange.svg)](https://gradio.app)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-compatible-green.svg)](https://openrouter.ai)
[![Tests](https://img.shields.io/badge/tests-65%20passing-brightgreen.svg)]()
[![Built by NEO](https://img.shields.io/badge/built%20by-NEO%20AI-purple.svg)](https://heyneo.so)

*Autonomously built by [NEO](https://heyneo.so) — your autonomous AI software engineer.*

</div>

---

## What is this?

A Gradio web app where you define agents with unique roles, models, and personas — then watch them collaborate in real time. Each agent sees the full conversation history and responds in turn, producing emergent multi-perspective dialogues. Built on [OpenRouter](https://openrouter.ai), it gives you access to 200+ LLMs from a single API key, with 6 ready-to-run scenarios and full support for custom configurations.

## ✨ Features

- 🎭 **6 built-in scenarios** — Debate, Code Review, Story Writing, Planning Session, Brainstorm, Socratic Dialogue
- ✏️ **Custom scenario support** — write your own description and agent configurations in JSON
- 🤖 **Per-agent controls** — set `model`, `temperature`, and `max_tokens` individually for each agent
- 🔄 **Live streaming** — messages appear one at a time as each agent responds; no waiting for the full run
- 🪙 **Live token usage counter** — prompt, completion, and total tokens updated after every turn
- 📊 **Stats row** — at-a-glance Turns / Tokens / Agents / Rounds dashboard
- 💾 **Conversation checkpointing** — save full session state to JSON and reload it later
- 📝 **Markdown rendering in chat** — bold, italic, and inline code rendered in message bubbles
- ⏱️ **Per-message timestamps** — each bubble shows the time it was generated
- ⬇️ **Export conversation as JSON** — download the full history and agent configuration
- 🗒️ **Auto-generated bullet-point summary** — triggered automatically after enough turns
- 🌍 **OpenRouter backend** — use any model on OpenRouter: DeepSeek, Claude, GPT-4o, Gemini, Llama, and more
- 🎨 **Colour-coded agent roster** — each agent gets a distinct pill and message colour
- 🚀 **Quick model picker** — swap all agents to a popular model with one click

## 🚀 Quick Start

```bash
git clone <repo>
cd multi-agent-sandbox
pip install -r requirements.txt
cp .env.example .env       # add your OPENROUTER_API_KEY
python app.py              # open http://localhost:7860
```

**No API key?** Run the headless demo:
```bash
python demo.py
```

## 🎭 Built-in Scenarios

| Scenario | Agents | Use case |
|----------|--------|----------|
| 🗣️ Debate | Advocate, Skeptic, Moderator | Explore both sides of any topic |
| 💻 Code Review | Developer, SecurityReviewer, PerformanceReviewer | Review code from multiple angles |
| 📖 Story Writing | PlotArchitect, CharacterWriter, WorldBuilder | Collaborative fiction |
| 📋 Planning Session | TechLead, ProductManager, RiskAnalyst | Project planning |
| 💡 Brainstorm | IdeaGenerator, CriticFilter, Synthesiser | Creative problem-solving |
| 🤔 Socratic | Questioner, Responder | Deep philosophical dialogue |

## 🤖 Supported Models

The sandbox uses [OpenRouter](https://openrouter.ai) — a single API key gives access to 200+ models. Set `DEFAULT_MODEL` in `.env` or pick any model per-agent in the JSON config.

### Recommended models

| Model | OpenRouter ID | Best for | Cost |
|-------|--------------|----------|------|
| **GPT-5.4 Pro** | `openai/gpt-5.4-pro` | OpenAI flagship, 1M ctx | 💛 Premium |
| **GPT-5.4** | `openai/gpt-5.4` | High-quality creative & planning | 💛 Premium |
| **Claude Opus 4.6** | `anthropic/claude-opus-4.6` | Complex multi-agent scenarios | 💛 Premium |
| **Claude Sonnet 4.6** | `anthropic/claude-sonnet-4.6` | Long conversations, nuanced roles | 💛 Premium |
| **Grok 4.20** | `x-ai/grok-4.20-beta` | Massive 2M context debates | 💛 Premium |
| **DeepSeek Chat** | `deepseek/deepseek-chat` | Default — fast, cheap, great reasoning | 💚 Budget |
| **DeepSeek R1** | `deepseek/deepseek-r1` | Deep analytical debates | 💚 Budget |
| **GPT-5.4 Mini** | `openai/gpt-5.4-mini` | Fast GPT-5 class, 400k ctx | 💚 Budget |
| **GPT-5.4 Nano** | `openai/gpt-5.4-nano` | Ultra-fast, low cost | 💚 Budget |
| **Mistral Small 4** | `mistralai/mistral-small-2603` | Efficient, 262k ctx | 💚 Budget |
| **Gemini 2.0 Flash** | `google/gemini-2.0-flash-001` | Speed + quality balance | 💚 Budget |
| **Qwen 2.5 72B** | `qwen/qwen-2.5-72b-instruct` | Multilingual, strong reasoning | 💚 Budget |
| **Nemotron 3 Super 120B** | `nvidia/nemotron-3-super-120b-a12b` | NVIDIA 120B, 262k ctx | 🆓 Free tier |
| **Qwen 3.5 9B** | `qwen/qwen3.5-9b` | Qwen latest, 256k ctx | 🆓 Free tier |
| **Llama 3.3 70B** | `meta-llama/llama-3.3-70b-instruct` | Strong open-source option | 🆓 Free tier |

### Switching models

```bash
# Set default for all agents
DEFAULT_MODEL=anthropic/claude-3.5-sonnet

# Per-agent (in the UI JSON config or examples):
{"name": "Analyst", "role": "...", "model": "openai/gpt-4o"}
{"name": "Assistant", "role": "...", "model": "meta-llama/llama-3.1-8b-instruct"}
```

Mix models freely — each agent can use a different one in the same conversation.

Browse all available models at [openrouter.ai/models](https://openrouter.ai/models).

---

## ⚙️ Agent Configuration

Agents are configured as a JSON array in the sidebar. Each agent supports:

```json
[
  {
    "name": "Advocate",
    "role": "You argue in favour of the proposition. Be concise and persuasive.",
    "model": "deepseek/deepseek-chat",
    "temperature": 0.8,
    "max_tokens": 512
  },
  {
    "name": "Skeptic",
    "role": "You argue against the proposition. Challenge every assumption.",
    "model": "openai/gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 512
  }
]
```

`temperature` and `max_tokens` are optional and default to `0.8` and `512`. Any model on [OpenRouter](https://openrouter.ai/models) works — e.g. `anthropic/claude-3.5-sonnet`, `openai/gpt-4o`, `google/gemini-2.0-flash-001`.

## 🌍 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | *(required)* | Your OpenRouter API key |
| `DEFAULT_MODEL` | `deepseek/deepseek-chat` | Default model for new agents |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | API base URL |
| `SUMMARY_TRIGGER_TURNS` | `6` | Auto-summary fires after this many agent turns |
| `PORT` | `7860` | Port the Gradio server listens on |
| `HOST` | `0.0.0.0` | Host the Gradio server binds to |

## 🧪 Running Tests

```bash
pytest tests/ -v   # 65 tests, all passing
```

All OpenAI API calls are mocked — no real API key required to run the test suite.

## 📁 Project Structure

```
├── sandbox/                  # Core package
│   ├── __init__.py           # Exports Agent, AgentOrchestrator, Scenario, SCENARIOS
│   ├── agents.py             # Agent dataclass, AgentOrchestrator, run_turn/run_scenario
│   └── scenarios.py          # Six preset scenario definitions
├── examples/                 # Runnable example scripts
│   ├── README.md
│   ├── 01_basic_debate.py
│   ├── 02_code_review.py
│   ├── 03_custom_agents.py
│   └── 04_checkpoint_resume.py
├── app.py                    # Gradio UI — layout, callbacks, session state
├── demo.py                   # Headless mocked demo — all features exercised without a browser
├── .env.example              # Environment variable template
├── requirements.txt          # Python dependencies
└── tests/
    ├── test_agents.py        # Agent creation, orchestrator, export/import, summary
    ├── test_enhancements.py  # Temperature controls, token tracking, checkpoints, markdown
    └── test_scenarios.py     # Preset scenarios, custom scenario support
```

## 💡 Examples

Clone the repo and run any example directly:

```bash
export OPENROUTER_API_KEY=sk-or-...

# Basic debate between 3 agents
python examples/01_basic_debate.py

# Multi-angle code review
python examples/02_code_review.py

# Fully custom agents with temperature control
python examples/03_custom_agents.py

# Save a conversation and resume it later
python examples/04_checkpoint_resume.py
```

### Example output - Custom Agents

```
Topic: Working remotely full-time is better than going to an office.
────────────────────────────────────────────────────────────
[Optimist] Remote work unlocks incredible flexibility - you save hours
on commutes, design your ideal workspace, and can work during your
most productive hours. It's a massive quality-of-life upgrade!

[Pessimist] Without the office, collaboration suffers and the boundary
between work and personal life blurs dangerously. Many people feel
isolated and miss the serendipitous conversations that spark creativity.

[Realist] Remote works best as a hybrid - 2-3 days at home for deep
focus, 2 days in-person for collaboration. The right balance depends
on role, team, and personal preference.

Tokens used: {'prompt': 1240, 'completion': 187, 'total': 1427}
```

## 🏗️ Architecture

```
User Input → AgentOrchestrator
                ├── Agent 1 (name, role, model, temperature, max_tokens)
                │     └── build_messages() → OpenRouter API → response
                ├── Agent 2 …
                └── Agent N …
                      └── history grows with each turn
                            ├── token usage accumulated
                            └── summary generated after SUMMARY_TRIGGER_TURNS turns

Gradio UI (app.py)
    └── gr.State (one AgentOrchestrator per browser session)
          ├── on_start()  — streams turn-by-turn updates via yield
          ├── on_stop()   — sets running=False to interrupt mid-run
          ├── on_reset()  — clears orchestrator and chat display
          └── checkpoint  — save/load full session state as JSON
```

The Gradio UI holds one `AgentOrchestrator` per session in Gradio `State`. The `on_start` handler runs agent turns one at a time and yields UI updates after each, giving a live streaming effect. Checkpoints serialise the full orchestrator state (agents + history + summary) to JSON and restore it via `AgentOrchestrator.load_checkpoint()`.

## 🤖 Built by NEO

This project was autonomously designed, coded, tested, and documented by
**[NEO](https://heyneo.so)** — an autonomous AI software engineer that turns ideas
into production-ready projects without human intervention.

> "From idea to working software, automatically."
> — [heyneo.so](https://heyneo.so)

## 📄 License

MIT
