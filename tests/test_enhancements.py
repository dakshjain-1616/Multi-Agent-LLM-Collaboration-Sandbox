"""Tests for all new enhancement features."""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox.agents import Agent, AgentOrchestrator, Message, DEFAULT_MODEL
from sandbox.scenarios import SCENARIOS
from app import format_message_html


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_mock_client(content="Mock response", prompt_tokens=10, completion_tokens=20):
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    client.chat.completions.create.return_value = response
    return client


def make_orchestrator(agents=None, content="Mock response", prompt_tokens=10, completion_tokens=20):
    with patch("sandbox.agents.OpenAI") as MockOpenAI:
        mock_client = make_mock_client(content, prompt_tokens, completion_tokens)
        MockOpenAI.return_value = mock_client
        orch = AgentOrchestrator(
            agents=agents or [Agent(name="Alpha", role="You are helpful.", model=DEFAULT_MODEL)],
            api_key="test-key",
        )
        orch._client = mock_client
        return orch, mock_client


# ──────────────────────────────────────────────────────────────────────────────
# Feature 1: Per-agent temperature & token controls
# ──────────────────────────────────────────────────────────────────────────────

class TestAgentTemperatureField:
    def test_agent_temperature_field(self):
        agent = Agent(name="T", role="r")
        assert agent.temperature == 0.8

    def test_agent_max_tokens_field(self):
        agent = Agent(name="T", role="r")
        assert agent.max_tokens == 512

    def test_agent_custom_temperature(self):
        agent = Agent(name="T", role="r", temperature=0.3)
        assert agent.temperature == 0.3

    def test_agent_custom_max_tokens(self):
        agent = Agent(name="T", role="r", max_tokens=256)
        assert agent.max_tokens == 256

    def test_run_turn_uses_agent_temperature(self):
        agent = Agent(name="A", role="r", temperature=0.2, max_tokens=128)
        orch, mock_client = make_orchestrator(agents=[agent])
        orch.add_user_message("hello")
        orch.run_turn(agent)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_tokens"] == 128


# ──────────────────────────────────────────────────────────────────────────────
# Feature 2: Token usage tracker
# ──────────────────────────────────────────────────────────────────────────────

class TestTokenUsageTracking:
    def test_token_usage_tracking(self):
        agent = Agent(name="A", role="r")
        orch, _ = make_orchestrator(agents=[agent], prompt_tokens=15, completion_tokens=25)
        assert orch.token_usage == {"prompt": 0, "completion": 0, "total": 0}
        orch.add_user_message("go")
        orch.run_turn(agent)
        usage = orch.token_usage
        assert usage["prompt"] == 15
        assert usage["completion"] == 25
        assert usage["total"] == 40

    def test_token_usage_accumulates(self):
        agent = Agent(name="A", role="r")
        orch, _ = make_orchestrator(agents=[agent], prompt_tokens=10, completion_tokens=20)
        orch.add_user_message("go")
        orch.run_turn(agent)
        orch.run_turn(agent)
        assert orch.token_usage["prompt"] == 20
        assert orch.token_usage["completion"] == 40

    def test_token_usage_reset_on_reset(self):
        agent = Agent(name="A", role="r")
        orch, _ = make_orchestrator(agents=[agent], prompt_tokens=10, completion_tokens=20)
        orch.add_user_message("go")
        orch.run_turn(agent)
        orch.reset()
        assert orch.token_usage == {"prompt": 0, "completion": 0, "total": 0}

    def test_token_usage_property_structure(self):
        orch, _ = make_orchestrator()
        usage = orch.token_usage
        assert "prompt" in usage
        assert "completion" in usage
        assert "total" in usage


# ──────────────────────────────────────────────────────────────────────────────
# Feature 3: New scenario presets
# ──────────────────────────────────────────────────────────────────────────────

class TestBrainstormScenario:
    def test_brainstorm_scenario_exists(self):
        assert "brainstorm" in SCENARIOS

    def test_brainstorm_has_three_agents(self):
        scenario = SCENARIOS["brainstorm"]
        assert len(scenario.suggested_agents) == 3

    def test_brainstorm_agent_names(self):
        scenario = SCENARIOS["brainstorm"]
        names = [a["name"] for a in scenario.suggested_agents]
        assert "IdeaGenerator" in names
        assert "CriticFilter" in names
        assert "Synthesiser" in names

    def test_brainstorm_has_initial_prompt(self):
        assert SCENARIOS["brainstorm"].initial_prompt.strip()


class TestSocraticScenario:
    def test_socratic_scenario_exists(self):
        assert "socratic" in SCENARIOS

    def test_socratic_has_two_agents(self):
        scenario = SCENARIOS["socratic"]
        assert len(scenario.suggested_agents) == 2

    def test_socratic_agent_names(self):
        scenario = SCENARIOS["socratic"]
        names = [a["name"] for a in scenario.suggested_agents]
        assert "Questioner" in names
        assert "Responder" in names

    def test_socratic_has_initial_prompt(self):
        assert SCENARIOS["socratic"].initial_prompt.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Feature 4: Checkpoint save/load
# ──────────────────────────────────────────────────────────────────────────────

class TestCheckpointSaveLoad:
    def test_checkpoint_save_load(self):
        agents = [
            Agent(name="Alice", role="Role A"),
            Agent(name="Bob", role="Role B"),
        ]
        orch, _ = make_orchestrator(agents=agents, content="Hello from mock")
        orch.add_user_message("Start")
        orch.run_turn(agents[0])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            orch.save_checkpoint(path)

            with patch("sandbox.agents.OpenAI"):
                restored = AgentOrchestrator.load_checkpoint(path, api_key="test")

            assert [a.name for a in restored.agents] == ["Alice", "Bob"]
            assert len(restored.history) == len(orch.history)
            assert restored.history[0].content == orch.history[0].content
        finally:
            os.unlink(path)

    def test_checkpoint_file_is_valid_json(self):
        orch, _ = make_orchestrator()
        orch.add_user_message("hi")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            orch.save_checkpoint(path)
            with open(path) as f:
                data = json.load(f)
            assert "agents" in data
            assert "history" in data
        finally:
            os.unlink(path)

    def test_checkpoint_restores_agent_temperature(self):
        agent = Agent(name="A", role="r", temperature=0.1, max_tokens=64)
        orch, _ = make_orchestrator(agents=[agent])
        orch.add_user_message("go")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            orch.save_checkpoint(path)
            with patch("sandbox.agents.OpenAI"):
                restored = AgentOrchestrator.load_checkpoint(path, api_key="test")
            assert restored.agents[0].temperature == 0.1
            assert restored.agents[0].max_tokens == 64
        finally:
            os.unlink(path)


# ──────────────────────────────────────────────────────────────────────────────
# Feature 5: Markdown rendering
# ──────────────────────────────────────────────────────────────────────────────

class TestMarkdownRendering:
    def _make_msg(self, content):
        return {"agent_name": "Bot", "content": content, "role": "assistant"}

    def test_markdown_bold(self):
        html = format_message_html(self._make_msg("Hello **world**"), ["Bot"])
        assert "<strong>world</strong>" in html

    def test_markdown_italic(self):
        html = format_message_html(self._make_msg("Hello *world*"), ["Bot"])
        assert "<em>world</em>" in html

    def test_markdown_code(self):
        html = format_message_html(self._make_msg("Use `print()` here"), ["Bot"])
        assert "<code>print()</code>" in html

    def test_markdown_newline(self):
        html = format_message_html(self._make_msg("line1\nline2"), ["Bot"])
        assert "<br>" in html

    def test_markdown_no_double_escape(self):
        # Ampersand should be HTML-escaped before markdown processing
        html = format_message_html(self._make_msg("A & B"), ["Bot"])
        assert "&amp;" in html
        assert "A & B" not in html
