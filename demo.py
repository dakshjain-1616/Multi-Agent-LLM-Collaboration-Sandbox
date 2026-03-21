"""
Headless demo: runs a 3-turn debate without launching Gradio.
All OpenAI API calls are mocked so no real API key is needed.
"""

import json
import sys
from unittest.mock import MagicMock, patch

from sandbox.agents import Agent, AgentOrchestrator
from sandbox.scenarios import get_scenario


def make_mock_response(content: str, prompt_tokens: int = 0, completion_tokens: int = 0):
    """Build a fake openai ChatCompletion response object."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


def run_demo():
    print("=" * 60)
    print("ALL FEATURES DEMO — Multi-Agent LLM Collaboration Sandbox")
    print("=" * 60)

    scenario = get_scenario("debate")

    print(f"\nScenario: {scenario.name}")
    print(f"Description: {scenario.description[:80]}...")
    print(f"Agents: {', '.join(a['name'] for a in scenario.suggested_agents)}")

    # Build agents from scenario config
    agents = [
        Agent(name=a["name"], role=a["role"], model=a["model"])
        for a in scenario.suggested_agents
    ]

    # Prepare sequential mock responses
    mock_responses = [
        make_mock_response(
            "AI regulation is essential to protect citizens from algorithmic harm, "
            "ensure accountability, and prevent monopolistic control by a few large "
            "corporations. Without oversight, we risk repeating the mistakes of social "
            "media — deploying powerful systems before understanding their consequences.",
            prompt_tokens=120, completion_tokens=60,
        ),
        make_mock_response(
            "Heavy-handed regulation stifles innovation and risks pushing AI development "
            "to less regulated jurisdictions. The technology moves too fast for legislation "
            "to keep up; self-regulation with industry standards is far more adaptive.",
            prompt_tokens=130, completion_tokens=55,
        ),
        make_mock_response(
            "Both sides raise valid points. The Advocate highlights accountability risks; "
            "the Skeptic warns of innovation drag. Key question: can we design 'light-touch' "
            "regulations that set safety floors without locking in today's architectures?",
            prompt_tokens=140, completion_tokens=58,
        ),
        make_mock_response(
            "Targeted regulation — focused on high-risk applications like healthcare and "
            "criminal justice — balances safety with innovation. We don't need to regulate "
            "all AI, just the deployments with significant societal impact.",
            prompt_tokens=145, completion_tokens=52,
        ),
        make_mock_response(
            "Even 'targeted' regulation requires defining 'high-risk', which is politically "
            "contested. Regulators often lack technical expertise, leading to rules that "
            "harm incumbents least and new entrants most.",
            prompt_tokens=150, completion_tokens=48,
        ),
        make_mock_response(
            "The debate highlights a genuine tension: safety vs innovation speed. "
            "Both agree high-risk AI warrants scrutiny; they diverge on mechanism. "
            "Open question: who should define 'high-risk' — governments, technologists, or civil society?",
            prompt_tokens=155, completion_tokens=62,
        ),
        # Summary response
        make_mock_response(
            "• Advocate supports regulation for accountability and public safety.\n"
            "• Skeptic warns against stifling innovation and regulatory capture.\n"
            "• Both agree high-risk AI deployments need oversight.\n"
            "• Key unresolved: who defines 'high-risk' and how to keep rules adaptive.\n"
            "• Possible middle ground: risk-tiered, sector-specific frameworks.",
            prompt_tokens=200, completion_tokens=80,
        ),
    ]

    call_idx = 0

    def mock_create(**kwargs):
        nonlocal call_idx
        resp = mock_responses[min(call_idx, len(mock_responses) - 1)]
        call_idx += 1
        return resp

    with patch("sandbox.agents.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = mock_create
        MockOpenAI.return_value = mock_client

        orchestrator = AgentOrchestrator(
            agents=agents,
            api_key="mock-key",
            scenario_description=scenario.description,
            summary_trigger_turns=6,
        )

        def print_message(msg):
            if msg.agent_name == "User":
                print(f"\n[USER PROMPT]\n{msg.content}\n")
            else:
                print(f"\n[{msg.agent_name}]\n{msg.content}\n")

        print(f"\nInitial prompt:\n{scenario.initial_prompt}\n")
        print("-" * 60)

        history = orchestrator.run_scenario(
            initial_prompt=scenario.initial_prompt,
            num_rounds=2,
            on_message=print_message,
        )

    print("=" * 60)
    print("CONVERSATION SUMMARY")
    print("=" * 60)
    print(orchestrator.summary or "(No summary generated)")

    print()
    usage = orchestrator.token_usage
    print("TOKEN USAGE")
    print(f"  Prompt tokens:     {usage['prompt']}")
    print(f"  Completion tokens: {usage['completion']}")
    print(f"  Total tokens:      {usage['total']}")

    print()
    print(f"Total messages: {len(history)}")
    print(f"Total agent turns: {orchestrator._turn_count}")

    # Checkpoint save/load round-trip
    print()
    print("CHECKPOINT SAVE/LOAD")
    ckpt_path = "/tmp/demo_checkpoint.json"
    orchestrator.save_checkpoint(ckpt_path)

    with patch("sandbox.agents.OpenAI") as MockOpenAI2:
        MockOpenAI2.return_value = MagicMock()
        restored = AgentOrchestrator.load_checkpoint(ckpt_path, api_key="mock-key")

    assert len(restored.agents) == len(orchestrator.agents), "Agent count mismatch after load"
    assert len(restored.history) == len(orchestrator.history), "History length mismatch after load"
    print(f"  Saved to: {ckpt_path}")
    print(f"  Loaded {len(restored.agents)} agents, {len(restored.history)} messages")
    print("✓ Checkpoint save/load working")

    print()
    print("Demo complete.")
    return orchestrator


if __name__ == "__main__":
    run_demo()
