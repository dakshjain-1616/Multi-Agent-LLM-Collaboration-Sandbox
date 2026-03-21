"""Tests for scenarios.py."""

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox.scenarios import SCENARIOS, Scenario, get_scenario, list_scenarios, get_scenario_names
from sandbox.agents import Agent, AgentOrchestrator


# ──────────────────────────────────────────────────────────────────────────────
# 3. test_scenario_presets
# ──────────────────────────────────────────────────────────────────────────────

class TestScenarioPresets:
    REQUIRED_KEYS = {"debate", "code_review", "story_writing", "planning_session"}

    def test_all_preset_keys_exist(self):
        for key in self.REQUIRED_KEYS:
            assert key in SCENARIOS, f"Missing scenario key: {key!r}"

    def test_all_preset_descriptions_nonempty(self):
        for key in self.REQUIRED_KEYS:
            scenario = SCENARIOS[key]
            assert scenario.description.strip(), f"Scenario {key!r} has empty description"

    def test_all_preset_names_nonempty(self):
        for key in self.REQUIRED_KEYS:
            scenario = SCENARIOS[key]
            assert scenario.name.strip(), f"Scenario {key!r} has empty name"

    def test_all_presets_have_suggested_agents(self):
        for key in self.REQUIRED_KEYS:
            scenario = SCENARIOS[key]
            assert len(scenario.suggested_agents) >= 2, (
                f"Scenario {key!r} should have at least 2 suggested agents"
            )

    def test_all_presets_have_initial_prompt(self):
        for key in self.REQUIRED_KEYS:
            scenario = SCENARIOS[key]
            assert scenario.initial_prompt.strip(), (
                f"Scenario {key!r} should have a non-empty initial_prompt"
            )

    def test_suggested_agents_have_required_fields(self):
        for key in self.REQUIRED_KEYS:
            scenario = SCENARIOS[key]
            for agent_def in scenario.suggested_agents:
                assert "name" in agent_def, f"Agent in {key!r} missing 'name'"
                assert "role" in agent_def, f"Agent in {key!r} missing 'role'"
                assert "model" in agent_def, f"Agent in {key!r} missing 'model'"

    def test_get_scenario_returns_correct_type(self):
        for key in self.REQUIRED_KEYS:
            scenario = get_scenario(key)
            assert isinstance(scenario, Scenario)

    def test_get_scenario_unknown_raises_key_error(self):
        with pytest.raises(KeyError):
            get_scenario("nonexistent_scenario_xyz")

    def test_list_scenarios_returns_all_keys(self):
        keys = list_scenarios()
        for key in self.REQUIRED_KEYS:
            assert key in keys

    def test_get_scenario_names_returns_dict(self):
        names = get_scenario_names()
        assert isinstance(names, dict)
        for key in self.REQUIRED_KEYS:
            assert key in names
            assert isinstance(names[key], str)
            assert names[key].strip()


# ──────────────────────────────────────────────────────────────────────────────
# 6. test_custom_scenario
# ──────────────────────────────────────────────────────────────────────────────

class TestCustomScenario:
    def _make_orchestrator_with_scenario(self, description: str):
        agent = Agent(name="CustomAgent", role="You are a custom agent.")
        with patch("sandbox.agents.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            choice = MagicMock()
            choice.message.content = "Custom agent responds."
            response = MagicMock()
            response.choices = [choice]
            mock_client.chat.completions.create.return_value = response
            MockOpenAI.return_value = mock_client

            orch = AgentOrchestrator(
                agents=[agent],
                api_key="test-key",
                scenario_description=description,
            )
            orch._client = mock_client
        return orch, agent, mock_client

    def test_custom_description_stored_on_orchestrator(self):
        desc = "A custom scenario where agents brainstorm product names."
        orch, _, _ = self._make_orchestrator_with_scenario(desc)
        assert orch.scenario_description == desc

    def test_custom_description_in_agent_system_prompt(self):
        desc = "Agents are collaborating to design a bridge."
        orch, agent, _ = self._make_orchestrator_with_scenario(desc)

        messages = agent.build_messages([], scenario_description=desc)

        system_content = messages[0]["content"]
        assert desc in system_content

    def test_custom_description_appears_in_api_call(self):
        desc = "A unique brainstorming session about renewable energy."
        orch, agent, mock_client = self._make_orchestrator_with_scenario(desc)

        orch.add_user_message("Let's start.")
        orch.run_turn(agent)

        call_args = mock_client.chat.completions.create.call_args
        messages_sent = call_args.kwargs.get("messages") or call_args[1].get("messages")
        system_message = next(m for m in messages_sent if m["role"] == "system")
        assert desc in system_message["content"]

    def test_custom_description_survives_export_reimport(self):
        desc = "Custom round-table discussion about climate policy."
        orch, _, mock_client = self._make_orchestrator_with_scenario(desc)
        orch.add_user_message("Begin.")

        exported = orch.export()

        with patch("sandbox.agents.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = mock_client
            orch2 = AgentOrchestrator.from_export(exported, api_key="test")

        assert orch2.scenario_description == desc

    def test_empty_custom_description_allowed(self):
        orch, _, _ = self._make_orchestrator_with_scenario("")
        assert orch.scenario_description == ""

    def test_scenario_description_used_across_multiple_turns(self):
        desc = "Agents debate the merits of TypeScript."
        orch, agent, mock_client = self._make_orchestrator_with_scenario(desc)

        orch.add_user_message("Start the debate.")
        orch.run_turn(agent)
        orch.run_turn(agent)

        for call_args in mock_client.chat.completions.create.call_args_list:
            messages_sent = call_args.kwargs.get("messages") or call_args[1].get("messages")
            system_msg = next(m for m in messages_sent if m["role"] == "system")
            assert desc in system_msg["content"]
