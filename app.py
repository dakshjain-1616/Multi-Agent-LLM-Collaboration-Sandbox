"""Gradio web app for multi-agent LLM collaboration sandbox."""

import json
import os
import re
import tempfile
import threading
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

from sandbox.agents import Agent, AgentOrchestrator, DEFAULT_MODEL
from sandbox.scenarios import SCENARIOS, get_scenario, list_scenarios

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette for agents (cycles if more agents than colours)
# ──────────────────────────────────────────────────────────────────────────────
AGENT_COLOURS = [
    "#4f86f7",  # blue
    "#f76f4f",  # orange-red
    "#4fc97e",  # green
    "#c94fc9",  # purple
    "#f7c74f",  # yellow
    "#4fc9c9",  # teal
]

USER_COLOUR = "#888888"

POPULAR_MODELS = [
    "deepseek/deepseek-chat",
    "openai/gpt-5.4-mini",
    "anthropic/claude-sonnet-4.6",
    "mistralai/mistral-small-2603",
    "nvidia/nemotron-3-super-120b-a12b",
]

SCENARIO_ICONS = {
    "debate": "🗣️",
    "code_review": "💻",
    "story_writing": "📖",
    "planning_session": "📋",
    "brainstorm": "💡",
    "socratic": "🤔",
    "custom": "✏️",
}

SCENARIO_SHORT_DESCS = {
    "debate": "Explore both sides of any topic",
    "code_review": "Review code from multiple angles",
    "story_writing": "Collaborative fiction creation",
    "planning_session": "Project planning with expert agents",
    "brainstorm": "Creative problem-solving session",
    "socratic": "Deep philosophical dialogue",
    "custom": "Define your own agents and prompt",
}


def colour_for_agent(name: str, agent_names: List[str]) -> str:
    if name == "User":
        return USER_COLOUR
    try:
        idx = agent_names.index(name) % len(AGENT_COLOURS)
        return AGENT_COLOURS[idx]
    except ValueError:
        return "#aaaaaa"


def _markdown_to_html(text: str) -> str:
    """Convert a small subset of markdown to HTML using regex."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    text = text.replace('\n', '<br>')
    return text


def format_message_html(msg_dict: Dict[str, Any], agent_names: List[str]) -> str:
    name = msg_dict.get("agent_name", "")
    content = msg_dict.get("content", "")
    timestamp = msg_dict.get("timestamp", "")
    model = msg_dict.get("model", "")
    colour = colour_for_agent(name, agent_names)
    escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    rendered = _markdown_to_html(escaped)

    meta_parts = []
    if model:
        meta_parts.append(
            f'<span style="color:#6a7a99;font-size:0.72em;font-style:italic;">{model}</span>'
        )
    if timestamp:
        meta_parts.append(
            f'<span style="color:#555e78;font-size:0.72em;">{timestamp}</span>'
        )
    meta_html = (
        '<span style="display:flex;gap:8px;align-items:center;">' + "".join(meta_parts) + "</span>"
        if meta_parts
        else ""
    )

    return (
        f'<div style="margin:8px 0; padding:10px 14px; border-radius:12px; '
        f'border-left: 4px solid {colour}; background:#1a1a2e; '
        f'box-shadow: 0 2px 8px rgba(0,0,0,0.35); transition: all 0.2s ease;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="color:{colour}; font-weight:bold; font-size:0.85em;">{name}</span>'
        f'{meta_html}'
        f'</div>'
        f'<div style="margin-top:6px; color:#e0e0e0;">'
        f'{rendered}</div>'
        f'</div>'
    )


def history_to_html(
    history: List[Dict[str, Any]],
    agent_names: List[str],
    timestamps: Optional[List[str]] = None,
    agent_models: Optional[Dict[str, str]] = None,
) -> str:
    if not history:
        return (
            '<p style="color:#666; text-align:center; padding:40px 0;">'
            'No messages yet. Configure agents and press ▶ Start.</p>'
        )
    parts = []
    for i, m in enumerate(history):
        msg = dict(m)
        if timestamps and i < len(timestamps):
            msg["timestamp"] = timestamps[i]
        if agent_models and msg.get("agent_name") in agent_models:
            msg["model"] = agent_models[msg["agent_name"]]
        parts.append(format_message_html(msg, agent_names))
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Global orchestrator state (one per session via Gradio State)
# ──────────────────────────────────────────────────────────────────────────────

def make_default_state() -> Dict[str, Any]:
    return {
        "orchestrator": None,
        "agent_names": [],
        "running": False,
        "timestamps": [],
        "agent_models": {},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_agents_config(config_text: str) -> List[Agent]:
    """
    Parse a JSON list of agent configs from the text area.
    Each item: {"name": ..., "role": ..., "model": ...}
    """
    try:
        data = json.loads(config_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in agent config: {e}")
    agents = []
    for item in data:
        agents.append(
            Agent(
                name=item.get("name", "Agent"),
                role=item.get("role", "You are a helpful assistant."),
                model=item.get("model", DEFAULT_MODEL),
                temperature=float(item.get("temperature", 0.8)),
                max_tokens=int(item.get("max_tokens", 512)),
            )
        )
    return agents


def load_scenario_config(scenario_key: str) -> Tuple[str, str]:
    """Return (agents_json, initial_prompt) for a preset scenario."""
    if scenario_key == "custom":
        return (
            json.dumps(
                [{"name": "Agent1", "role": "You are a helpful assistant.", "model": DEFAULT_MODEL}],
                indent=2,
            ),
            "",
        )
    scenario = get_scenario(scenario_key)
    agents_json = json.dumps(scenario.suggested_agents, indent=2)
    return agents_json, scenario.initial_prompt


def build_agent_roster_html(agents_json: str) -> str:
    """Render a row of colour-coded pills, one per agent."""
    try:
        data = json.loads(agents_json)
        pills = []
        for i, agent in enumerate(data):
            color = AGENT_COLOURS[i % len(AGENT_COLOURS)]
            name = agent.get("name", "?")
            pills.append(
                f'<span style="background:{color};color:white;padding:3px 11px;'
                f'border-radius:12px;margin:2px;display:inline-block;font-size:0.78em;'
                f'font-weight:600;box-shadow:0 1px 4px rgba(0,0,0,0.25);">{name}</span>'
            )
        if pills:
            return '<div style="padding:4px 0;">' + " ".join(pills) + "</div>"
        return ""
    except Exception:
        return ""


def build_stats_html(turn_count: int, total_tokens: int, agent_count: int, rounds: int) -> str:
    return (
        '<div style="display:flex;gap:20px;padding:8px 14px;background:#f5f6fa;'
        'border-radius:8px;font-size:0.83em;color:#444;border:1px solid #dde0ea;'
        'flex-wrap:wrap;">'
        f'<span>🔄 <strong>Turns:</strong> {turn_count}</span>'
        f'<span>🪙 <strong>Tokens:</strong> {total_tokens}</span>'
        f'<span>🤖 <strong>Agents:</strong> {agent_count}</span>'
        f'<span>📦 <strong>Rounds:</strong> {rounds}</span>'
        '</div>'
    )


# ──────────────────────────────────────────────────────────────────────────────
# Scenario radio choices
# ──────────────────────────────────────────────────────────────────────────────

SCENARIO_RADIO_CHOICES = [
    ("✏️ Custom — Define your own agents and prompt", "custom")
] + [
    (
        f"{SCENARIO_ICONS.get(key, '')} {s.name} — {SCENARIO_SHORT_DESCS.get(key, s.description[:55])}",
        key,
    )
    for key, s in SCENARIOS.items()
]

DEFAULT_SCENARIO_KEY = "debate"
DEFAULT_AGENTS_JSON, DEFAULT_INITIAL_PROMPT = load_scenario_config(DEFAULT_SCENARIO_KEY)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Chat area ───────────────────────────────────────────────────── */
#chat-display {
    min-height: 420px;
    max-height: 640px;
    overflow-y: auto;
    background: #0f0f1a;
    border-radius: 12px;
    padding: 12px;
    border: 1px solid #2a2a4a;
}

/* ── Branded header ──────────────────────────────────────────────── */
#app-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 16px;
    border: 1px solid #2a2a5e;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* ── Welcome banner ──────────────────────────────────────────────── */
#welcome-banner {
    background: linear-gradient(90deg, #1a3a5c, #1a2a4a);
    border: 1px solid #2a4a7c;
    border-radius: 10px;
    margin-bottom: 12px;
}

/* ── Sidebar ─────────────────────────────────────────────────────── */
.sidebar-col {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 8px;
}

/* ── Buttons ─────────────────────────────────────────────────────── */
button { transition: all 0.2s ease !important; }
button:hover { transform: translateY(-1px); box-shadow: 0 3px 10px rgba(0,0,0,0.2); }

/* ── Accordions ──────────────────────────────────────────────────── */
.gr-accordion { border-radius: 10px !important; margin-bottom: 8px !important; }

/* ── Hide footer ─────────────────────────────────────────────────── */
footer { display: none !important; }
"""


# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────────────

def build_ui():
    env_has_key = bool(os.environ.get("OPENROUTER_API_KEY", ""))

    with gr.Blocks(
        title="Multi-Agent LLM Collaboration Sandbox",
        theme=gr.themes.Soft(
            primary_hue="blue",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
    ) as demo:
        # ── State ──────────────────────────────────────────────────────────
        state = gr.State(make_default_state)

        # ── Welcome banner ─────────────────────────────────────────────────
        with gr.Group(elem_id="welcome-banner") as welcome_group:
            with gr.Row():
                gr.HTML(
                    '<div style="padding:10px 16px;color:#a0c4ff;font-size:0.9em;">'
                    '👋 <strong>Welcome!</strong> Pick a scenario, configure your agents, '
                    'and press ▶ Start. '
                    'No API key? Use the headless demo: '
                    '<code style="background:#0a1628;padding:2px 6px;border-radius:4px;">'
                    'python demo.py</code>'
                    '</div>'
                )
                dismiss_btn = gr.Button("✕", size="sm", scale=0, min_width=44)

        # ── Branded header ─────────────────────────────────────────────────
        gr.HTML(
            '<div id="app-header">'
            '<div style="display:flex;align-items:center;gap:14px;">'
            '<span style="font-size:2.6em;line-height:1;">🤖</span>'
            '<div>'
            '<h1 style="margin:0;color:#e0e8ff;font-size:1.55em;font-weight:700;'
            'letter-spacing:-0.3px;">Multi-Agent LLM Collaboration Sandbox</h1>'
            '<p style="margin:5px 0 0;color:#8899cc;font-size:0.88em;">'
            'Orchestrate AI agents. Watch them think together.</p>'
            '</div>'
            '</div>'
            '</div>'
        )

        with gr.Row():
            # ── Sidebar ────────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=340, elem_classes="sidebar-col"):

                with gr.Accordion("🔑 API Setup", open=not env_has_key):
                    api_key_input = gr.Textbox(
                        label="OpenRouter API Key",
                        placeholder="sk-or-... (or set OPENROUTER_API_KEY env var)",
                        type="password",
                        value=os.environ.get("OPENROUTER_API_KEY", ""),
                    )

                with gr.Accordion("🎭 Scenario", open=True):
                    scenario_radio = gr.Radio(
                        label="Choose a scenario",
                        choices=SCENARIO_RADIO_CHOICES,
                        value=DEFAULT_SCENARIO_KEY,
                    )
                    scenario_desc = gr.Markdown(
                        value=f"**{SCENARIOS[DEFAULT_SCENARIO_KEY].name}**: "
                        f"{SCENARIOS[DEFAULT_SCENARIO_KEY].description}",
                    )
                    custom_scenario_desc = gr.Textbox(
                        label="Custom scenario description (used in system prompt)",
                        placeholder="Describe the collaboration context…",
                        lines=3,
                        visible=False,
                    )

                with gr.Accordion("⚙️ Agent Config", open=True):
                    model_picker = gr.Dropdown(
                        label="Quick model picker (sets model for all agents)",
                        choices=POPULAR_MODELS,
                        value=DEFAULT_MODEL,
                        allow_custom_value=True,
                    )
                    agents_config = gr.Code(
                        label="Agent Configuration (JSON)",
                        value=DEFAULT_AGENTS_JSON,
                        language="json",
                        lines=12,
                    )
                    agent_roster_html = gr.HTML(
                        value=build_agent_roster_html(DEFAULT_AGENTS_JSON),
                    )

                with gr.Accordion("▶ Run Controls", open=True):
                    num_rounds = gr.Slider(
                        label="Rounds (each agent speaks once per round)",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=2,
                    )
                    initial_prompt = gr.Textbox(
                        label="Initial Prompt / Task",
                        value=DEFAULT_INITIAL_PROMPT,
                        lines=4,
                        placeholder="Enter the task or opening message for agents…",
                    )
                    with gr.Row():
                        start_btn = gr.Button("▶ Start", variant="primary")
                        stop_btn = gr.Button("⏹ Stop", variant="stop")
                        reset_btn = gr.Button("↺ Reset")
                    status_box = gr.Textbox(
                        label="Status",
                        value="Ready",
                        interactive=False,
                        lines=1,
                    )
                    current_speaker_md = gr.Markdown(value="", elem_id="current-speaker")

            # ── Main chat area ─────────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### Conversation")

                chat_display = gr.HTML(
                    value=(
                        '<p style="color:#888; text-align:center; padding:40px 0;">'
                        'Configure agents and press ▶ Start.</p>'
                    ),
                    elem_id="chat-display",
                )

                stats_html = gr.HTML(
                    value=build_stats_html(0, 0, 0, 0),
                    elem_id="stats-row",
                )

                token_counter = gr.Markdown(
                    value="Tokens used: 0",
                    elem_id="token-counter",
                )

                gr.Markdown("### Summary")
                summary_box = gr.Textbox(
                    label="Auto-generated summary (appears after enough turns)",
                    interactive=False,
                    lines=6,
                    placeholder="Summary will appear here after the conversation completes.",
                )

                with gr.Row():
                    export_btn = gr.Button("⬇ Export conversation (JSON)")
                    export_file = gr.File(label="Download", visible=False)

                with gr.Row():
                    save_ckpt_btn = gr.Button("💾 Save Checkpoint")
                    load_ckpt_btn = gr.Button("📂 Load Checkpoint")

                load_ckpt_file = gr.File(
                    label="Checkpoint file to load",
                    file_types=[".json"],
                    visible=False,
                )

        # ── Callbacks ──────────────────────────────────────────────────────

        dismiss_btn.click(lambda: gr.update(visible=False), outputs=[welcome_group])

        def on_scenario_change(scenario_key):
            agents_json, prompt = load_scenario_config(scenario_key)
            is_custom = scenario_key == "custom"
            if is_custom:
                desc_md = "**Custom**: Define your own agents and prompt."
            else:
                s = SCENARIOS[scenario_key]
                desc_md = f"**{s.name}**: {s.description}"
            roster = build_agent_roster_html(agents_json)
            return (
                gr.update(value=desc_md),
                gr.update(value=agents_json),
                gr.update(value=prompt),
                gr.update(visible=is_custom),
                gr.update(value=roster),
            )

        scenario_radio.change(
            on_scenario_change,
            inputs=[scenario_radio],
            outputs=[scenario_desc, agents_config, initial_prompt, custom_scenario_desc, agent_roster_html],
        )

        def on_model_pick(model_value, current_json):
            try:
                data = json.loads(current_json)
                for agent in data:
                    agent["model"] = model_value
                new_json = json.dumps(data, indent=2)
                return gr.update(value=new_json), gr.update(value=build_agent_roster_html(new_json))
            except Exception:
                return gr.update(), gr.update()

        model_picker.change(
            on_model_pick,
            inputs=[model_picker, agents_config],
            outputs=[agents_config, agent_roster_html],
        )

        def on_agents_change(agents_json):
            return gr.update(value=build_agent_roster_html(agents_json))

        agents_config.change(
            on_agents_change,
            inputs=[agents_config],
            outputs=[agent_roster_html],
        )

        def on_start(
            api_key,
            scenario_key,
            custom_desc,
            agents_json_text,
            rounds,
            prompt,
            current_state,
        ):
            if current_state.get("running"):
                yield (
                    gr.update(),
                    gr.update(value="Already running. Press Stop first."),
                    gr.update(),
                    current_state,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )
                return

            # Parse agents
            try:
                agents = parse_agents_config(agents_json_text)
            except ValueError as e:
                yield (
                    gr.update(),
                    gr.update(value=f"Error: {e}"),
                    gr.update(),
                    current_state,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )
                return

            if not agents:
                yield (
                    gr.update(),
                    gr.update(value="Error: No agents defined."),
                    gr.update(),
                    current_state,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )
                return

            if not prompt.strip():
                yield (
                    gr.update(),
                    gr.update(value="Error: Initial prompt is empty."),
                    gr.update(),
                    current_state,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )
                return

            scenario_desc_text = ""
            if scenario_key == "custom":
                scenario_desc_text = custom_desc
            elif scenario_key in SCENARIOS:
                scenario_desc_text = SCENARIOS[scenario_key].description

            resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")

            orchestrator = AgentOrchestrator(
                agents=agents,
                api_key=resolved_key,
                scenario_description=scenario_desc_text,
            )

            current_state["orchestrator"] = orchestrator
            current_state["agent_names"] = [a.name for a in agents]
            current_state["agent_models"] = {a.name: a.model for a in agents}
            current_state["running"] = True
            current_state["timestamps"] = []

            agent_names = current_state["agent_names"]
            agent_models = current_state["agent_models"]

            # Add initial user message
            orchestrator.add_user_message(prompt)
            current_state["timestamps"].append(datetime.now().strftime("%H:%M:%S"))

            html = history_to_html(
                [m.to_dict() for m in orchestrator.history],
                agent_names,
                timestamps=current_state["timestamps"],
                agent_models=agent_models,
            )
            yield (
                gr.update(value=html),
                gr.update(value="Running…"),
                gr.update(value=""),
                current_state,
                gr.update(value="Tokens used: 0"),
                gr.update(value=build_stats_html(0, 0, len(agents), int(rounds))),
                gr.update(value=""),
            )

            # Run rounds turn by turn, yielding after each
            try:
                for round_idx in range(int(rounds)):
                    if not current_state.get("running"):
                        break
                    for agent in orchestrator.agents:
                        if not current_state.get("running"):
                            break
                        # Show "currently speaking" before turn
                        yield (
                            gr.update(),
                            gr.update(value=f"Round {round_idx + 1} — {agent.name} thinking…"),
                            gr.update(),
                            current_state,
                            gr.update(),
                            gr.update(),
                            gr.update(value=f"🔄 Currently speaking: **{agent.name}**"),
                        )
                        msg = orchestrator.run_turn(agent)
                        current_state["timestamps"].append(datetime.now().strftime("%H:%M:%S"))
                        html = history_to_html(
                            [m.to_dict() for m in orchestrator.history],
                            agent_names,
                            timestamps=current_state["timestamps"],
                            agent_models=agent_models,
                        )
                        summary_text = orchestrator.summary or ""
                        usage = orchestrator.token_usage
                        yield (
                            gr.update(value=html),
                            gr.update(value=f"Round {round_idx + 1} — {agent.name} responded"),
                            gr.update(value=summary_text),
                            current_state,
                            gr.update(value=f"Tokens used: {usage['total']} (prompt: {usage['prompt']}, completion: {usage['completion']})"),
                            gr.update(value=build_stats_html(orchestrator._turn_count, usage["total"], len(agents), int(rounds))),
                            gr.update(value=f"🔄 Currently speaking: **{agent.name}**"),
                        )

                # Generate summary if not yet done
                if (
                    current_state.get("running")
                    and orchestrator.summary is None
                    and len(orchestrator.history) > 1
                ):
                    summary = orchestrator.generate_summary()
                    orchestrator._summary = summary
                    usage = orchestrator.token_usage
                    yield (
                        gr.update(),
                        gr.update(value="Done. Summary generated."),
                        gr.update(value=summary),
                        current_state,
                        gr.update(),
                        gr.update(value=build_stats_html(orchestrator._turn_count, usage["total"], len(agents), int(rounds))),
                        gr.update(value=""),
                    )
                else:
                    usage = orchestrator.token_usage
                    yield (
                        gr.update(),
                        gr.update(value="Done."),
                        gr.update(value=orchestrator.summary or ""),
                        current_state,
                        gr.update(),
                        gr.update(value=build_stats_html(orchestrator._turn_count, usage["total"], len(agents), int(rounds))),
                        gr.update(value=""),
                    )

            except Exception as e:
                yield (
                    gr.update(),
                    gr.update(value=f"Error: {e}"),
                    gr.update(),
                    current_state,
                    gr.update(),
                    gr.update(),
                    gr.update(value=""),
                )
            finally:
                current_state["running"] = False

        start_btn.click(
            on_start,
            inputs=[
                api_key_input,
                scenario_radio,
                custom_scenario_desc,
                agents_config,
                num_rounds,
                initial_prompt,
                state,
            ],
            outputs=[chat_display, status_box, summary_box, state, token_counter, stats_html, current_speaker_md],
        )

        def on_stop(current_state):
            current_state["running"] = False
            return current_state, gr.update(value="Stopped.")

        stop_btn.click(on_stop, inputs=[state], outputs=[state, status_box])

        def on_reset(current_state):
            current_state["orchestrator"] = None
            current_state["agent_names"] = []
            current_state["running"] = False
            current_state["timestamps"] = []
            current_state["agent_models"] = {}
            return (
                current_state,
                gr.update(
                    value='<p style="color:#888; text-align:center; padding:40px 0;">Reset. Configure agents and press ▶ Start.</p>'
                ),
                gr.update(value=""),
                gr.update(value="Ready"),
                gr.update(value=build_stats_html(0, 0, 0, 0)),
                gr.update(value=""),
            )

        reset_btn.click(
            on_reset,
            inputs=[state],
            outputs=[state, chat_display, summary_box, status_box, stats_html, current_speaker_md],
        )

        def on_export(current_state):
            orchestrator: Optional[AgentOrchestrator] = current_state.get("orchestrator")
            if orchestrator is None or not orchestrator.history:
                return gr.update(visible=False)
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
                encoding="utf-8",
                prefix="multi_agent_",
            )
            tmp.write(orchestrator.export_json())
            tmp.close()
            return gr.update(value=tmp.name, visible=True)

        export_btn.click(
            on_export,
            inputs=[state],
            outputs=[export_file],
        )

        def on_save_checkpoint(current_state):
            orchestrator: Optional[AgentOrchestrator] = current_state.get("orchestrator")
            if orchestrator is None or not orchestrator.history:
                return gr.update(visible=False), gr.update(value="Nothing to save.")
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
                encoding="utf-8",
                prefix="checkpoint_",
            )
            tmp.close()
            orchestrator.save_checkpoint(tmp.name)
            return gr.update(value=tmp.name, visible=True), gr.update(value="Checkpoint saved.")

        save_ckpt_btn.click(
            on_save_checkpoint,
            inputs=[state],
            outputs=[export_file, status_box],
        )

        def on_load_checkpoint_click():
            return gr.update(visible=True)

        def on_load_checkpoint(file_obj, api_key, current_state):
            if file_obj is None:
                return current_state, gr.update(), gr.update(value="No file selected.")
            try:
                restored = AgentOrchestrator.load_checkpoint(
                    file_obj.name,
                    api_key=api_key or os.environ.get("OPENROUTER_API_KEY", ""),
                )
                current_state["orchestrator"] = restored
                current_state["agent_names"] = [a.name for a in restored.agents]
                current_state["agent_models"] = {a.name: a.model for a in restored.agents}
                current_state["running"] = False
                current_state["timestamps"] = []
                html = history_to_html(
                    [m.to_dict() for m in restored.history],
                    current_state["agent_names"],
                    agent_models=current_state["agent_models"],
                )
                return current_state, gr.update(value=html), gr.update(value="Checkpoint loaded.")
            except Exception as e:
                return current_state, gr.update(), gr.update(value=f"Error loading checkpoint: {e}")

        load_ckpt_btn.click(on_load_checkpoint_click, outputs=[load_ckpt_file])
        load_ckpt_file.change(
            on_load_checkpoint,
            inputs=[load_ckpt_file, api_key_input, state],
            outputs=[state, chat_display, status_box],
        )

    return demo


def main():
    demo = build_ui()
    demo.launch(
        server_name=os.environ.get("HOST", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )


if __name__ == "__main__":
    main()
