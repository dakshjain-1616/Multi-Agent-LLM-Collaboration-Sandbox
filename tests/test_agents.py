"""Tests for agents.py."""

import json
import sys
import os
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure workspace root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox.agents import Agent, AgentOrchestrator, Message, DEFAULT_MODEL


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def make_mock_client(content="Mock response"):
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    return client


def make_orchestrator(agents=None, content="Mock response"):
    with patch("sandbox.agents.OpenAI") as MockOpenAI:
        mock_client = make_mock_client(content)
        MockOpenAI.return_value = mock_client
        orch = AgentOrchestrator(
            agents=agents or [Agent(name="Alpha", role="You are helpful.", model=DEFAULT_MODEL)],
            api_key="test-key",
            scenario_description="A test scenario.",
        )
        orch._client = mock_client  # inject directly so mock persists
        return orch, mock_client


# ──────────────────────────────────────────────────────────────────────────────
# 1. test_agent_creation
# ──────────────────────────────────────────────────────────────────────────────

class TestAgentCreation:
    def test_basic_creation(self):
        agent = Agent(name="TestAgent", role="You are a tester.", model="gpt-4")
        assert agent.name == "TestAgent"
        assert agent.role == "You are a tester."
        assert agent.model == "gpt-4"

    def test_default_model(self):
        agent = Agent(name="DefaultModel", role="A role.")
        assert agent.model == DEFAULT_MODEL

    def test_name_field(self):
        agent = Agent(name="Unique Name", role="role", model="some-model")
        assert agent.name == "Unique Name"

    def test_role_field(self):
        role = "You are an expert in quantum computing."
        agent = Agent(name="Expert", role=role)
        assert agent.role == role

    def test_build_messages_includes_system(self):
        agent = Agent(name="Alice", role="You are Alice.", model=DEFAULT_MODEL)
        messages = agent.build_messages([], scenario_description="Test scenario")
        assert messages[0]["role"] == "system"
        assert "Test scenario" in messages[0]["content"]
        assert "You are Alice." in messages[0]["content"]

    def test_build_messages_history_labelled(self):
        agent = Agent(name="Alice", role="You are Alice.")
        history = [
            Message(agent_name="Bob", content="Hello Alice!", role="assistant"),
            Message(agent_name="Alice", content="Hi Bob!", role="assistant"),
        ]
        messages = agent.build_messages(history)
        # Bob's message should appear as user message with [Bob] label
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert any("[Bob]" in m["content"] for m in user_msgs)
        # Alice's own message should appear as assistant
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert any("Hi Bob!" in m["content"] for m in assistant_msgs)


# ──────────────────────────────────────────────────────────────────────────────
# 2. test_orchestrator_turn
# ──────────────────────────────────────────────────────────────────────────────

class TestOrchestratorTurn:
    def test_run_turn_calls_api(self):
        agent = Agent(name="Alpha", role="You are Alpha.")
        orch, mock_client = make_orchestrator(agents=[agent], content="Alpha says hello")

        msg = orch.run_turn(agent)

        assert mock_client.chat.completions.create.called
        assert msg.content == "Alpha says hello"
        assert msg.agent_name == "Alpha"

    def test_run_turn_appends_to_history(self):
        agent = Agent(name="Beta", role="You are Beta.")
        orch, mock_client = make_orchestrator(agents=[agent], content="Beta responds")

        assert len(orch.history) == 0
        orch.run_turn(agent)
        assert len(orch.history) == 1
        assert orch.history[0].agent_name == "Beta"
        assert orch.history[0].content == "Beta responds"

    def test_run_turn_increments_turn_count(self):
        agent = Agent(name="Gamma", role="You are Gamma.")
        orch, _ = make_orchestrator(agents=[agent])

        assert orch._turn_count == 0
        orch.run_turn(agent)
        assert orch._turn_count == 1
        orch.run_turn(agent)
        assert orch._turn_count == 2

    def test_run_turn_correct_model_passed(self):
        agent = Agent(name="Delta", role="You are Delta.", model="anthropic/claude-3")
        orch, mock_client = make_orchestrator(agents=[agent])

        orch.run_turn(agent)

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("model") == "anthropic/claude-3"

    def test_run_round_all_agents_called(self):
        agents = [
            Agent(name="A1", role="Role 1."),
            Agent(name="A2", role="Role 2."),
            Agent(name="A3", role="Role 3."),
        ]
        orch, mock_client = make_orchestrator(agents=agents)

        msgs = orch.run_round()

        assert len(msgs) == 3
        assert mock_client.chat.completions.create.call_count == 3
        assert [m.agent_name for m in msgs] == ["A1", "A2", "A3"]

    def test_run_scenario_adds_user_message_first(self):
        agent = Agent(name="Solo", role="You are Solo.")
        orch, _ = make_orchestrator(agents=[agent])

        orch.run_scenario("Test prompt", num_rounds=1)

        assert orch.history[0].agent_name == "User"
        assert orch.history[0].content == "Test prompt"

    def test_run_scenario_empty_prompt_raises(self):
        agent = Agent(name="Solo", role="Role.")
        orch, _ = make_orchestrator(agents=[agent])

        with pytest.raises(ValueError, match="initial_prompt must not be empty"):
            orch.run_scenario("   ", num_rounds=1)


# ──────────────────────────────────────────────────────────────────────────────
# 4. test_export_format
# ──────────────────────────────────────────────────────────────────────────────

class TestExportFormat:
    def test_export_has_agents_and_history_keys(self):
        agent = Agent(name="ExportAgent", role="Export role.")
        orch, _ = make_orchestrator(agents=[agent])
        orch.add_user_message("Hello")

        exported = orch.export()

        assert "agents" in exported
        assert "history" in exported

    def test_export_agents_list(self):
        agents = [
            Agent(name="A", role="Role A.", model="model-a"),
            Agent(name="B", role="Role B.", model="model-b"),
        ]
        orch, _ = make_orchestrator(agents=agents)

        exported = orch.export()

        assert len(exported["agents"]) == 2
        assert exported["agents"][0]["name"] == "A"
        assert exported["agents"][1]["name"] == "B"
        assert exported["agents"][0]["model"] == "model-a"

    def test_export_json_valid(self):
        agent = Agent(name="JsonAgent", role="Role.")
        orch, _ = make_orchestrator(agents=[agent])
        orch.add_user_message("Test message")

        json_str = orch.export_json()

        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "agents" in parsed
        assert "history" in parsed

    def test_export_history_contains_messages(self):
        agent = Agent(name="HistAgent", role="Role.")
        orch, mock_client = make_orchestrator(agents=[agent], content="A response")

        orch.add_user_message("Initial")
        orch.run_turn(agent)

        exported = orch.export()
        assert len(exported["history"]) == 2
        assert exported["history"][0]["agent_name"] == "User"
        assert exported["history"][1]["agent_name"] == "HistAgent"

    def test_export_includes_scenario_description(self):
        agent = Agent(name="X", role="Role.")
        with patch("sandbox.agents.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = make_mock_client()
            orch = AgentOrchestrator(
                agents=[agent],
                api_key="test",
                scenario_description="My custom scenario",
            )
            orch._client = MockOpenAI.return_value
        exported = orch.export()
        assert exported["scenario_description"] == "My custom scenario"

    def test_roundtrip_from_export(self):
        agent = Agent(name="Round", role="Role.", model=DEFAULT_MODEL)
        orch, mock_client = make_orchestrator(agents=[agent], content="reply")
        orch.add_user_message("prompt")
        orch.run_turn(agent)

        data = orch.export()

        with patch("sandbox.agents.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = mock_client
            orch2 = AgentOrchestrator.from_export(data, api_key="test")
            assert len(orch2.history) == len(orch.history)
            assert orch2.agents[0].name == "Round"


# ──────────────────────────────────────────────────────────────────────────────
# 5. test_summary_generation
# ──────────────────────────────────────────────────────────────────────────────

class TestSummaryGeneration:
    def test_generate_summary_returns_nonempty_string(self):
        agent = Agent(name="Summariser", role="Role.")
        orch, mock_client = make_orchestrator(
            agents=[agent], content="This is a summary."
        )
        orch.add_user_message("Discuss something important.")
        orch.history.append(
            Message(agent_name="Summariser", content="Important point A.")
        )

        summary = orch.generate_summary()

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summary_calls_llm_once(self):
        agent = Agent(name="SumAgent", role="Role.")
        orch, mock_client = make_orchestrator(
            agents=[agent], content="Mock summary text."
        )
        orch.add_user_message("Start")
        orch.history.append(Message(agent_name="SumAgent", content="Response"))

        orch.generate_summary()

        assert mock_client.chat.completions.create.call_count == 1

    def test_auto_summary_triggered_after_enough_turns(self):
        agents = [Agent(name=f"A{i}", role=f"Role {i}.") for i in range(3)]
        orch, mock_client = make_orchestrator(agents=agents, content="resp")
        # summary_trigger_turns defaults to 6; we'll run 2 rounds × 3 agents = 6 turns
        orch.run_scenario("Initial prompt", num_rounds=2)

        assert orch.summary is not None
        assert isinstance(orch.summary, str)
        assert len(orch.summary) > 0

    def test_no_agents_generate_summary_returns_empty(self):
        with patch("sandbox.agents.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = MagicMock()
            orch = AgentOrchestrator(agents=[], api_key="test")
        result = orch.generate_summary()
        assert result == ""

    def test_empty_history_generate_summary_returns_empty(self):
        agent = Agent(name="X", role="Role.")
        orch, _ = make_orchestrator(agents=[agent])
        result = orch.generate_summary()
        assert result == ""
