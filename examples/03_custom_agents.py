"""
Example 3: Build completely custom agents from scratch.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python examples/03_custom_agents.py

Without an API key the script runs in dry-run mode with mock responses.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sandbox import Agent, AgentOrchestrator

def _mock_orchestrator(orch):
    from unittest.mock import MagicMock
    responses = [
        "Remote work unlocks incredible flexibility — you save hours of commuting and design your ideal environment!",
        "But isolation is real: no spontaneous hallway conversations, harder to onboard new hires, blurred work-life boundaries.",
        "Remote works best as a hybrid — 2-3 days at home for deep focus, 2 days in-office for collaboration.",
        "The camaraderie and serendipitous ideas from in-person time are genuinely irreplaceable for creative teams.",
        "Productivity data is mixed: individual tasks improve remotely, complex team projects often suffer.",
        "Conclusion: flexible hybrid with clear team agreements beats fully remote or fully in-office for most roles.",
    ]
    call_idx = {"i": 0}
    def mock_create(**kwargs):
        idx = call_idx["i"] % len(responses)
        call_idx["i"] += 1
        r = MagicMock()
        r.choices[0].message.content = responses[idx]
        r.usage.prompt_tokens = 80
        r.usage.completion_tokens = 40
        return r
    orch._client.chat.completions.create = mock_create
    return orch

def main():
    agents = [
        Agent(name="Optimist",
              role="You always find the silver lining. Be enthusiastic and positive. 2-3 sentences.",
              model=os.environ.get("DEFAULT_MODEL", "deepseek/deepseek-chat"),
              temperature=0.9, max_tokens=150),
        Agent(name="Pessimist",
              role="You always identify risks and downsides. Be realistic and cautious. 2-3 sentences.",
              model=os.environ.get("DEFAULT_MODEL", "deepseek/deepseek-chat"),
              temperature=0.7, max_tokens=150),
        Agent(name="Realist",
              role="You synthesise both views into a balanced, practical assessment. 2-3 sentences.",
              model=os.environ.get("DEFAULT_MODEL", "deepseek/deepseek-chat"),
              temperature=0.5, max_tokens=200),
    ]

    orch = AgentOrchestrator(agents=agents, scenario_description="Balanced perspective exercise",
                             api_key=os.environ.get("OPENROUTER_API_KEY", ""))

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[dry-run] No OPENROUTER_API_KEY — using mock responses.\n")
        _mock_orchestrator(orch)

    def on_msg(msg):
        print(f"\n[{msg.agent_name}] {msg.content}")

    topic = "Working remotely full-time is better than going to an office."
    print(f"Topic: {topic}\n{'─'*60}")
    orch.run_scenario(topic, num_rounds=2, on_message=on_msg)
    print(f"\nTokens used: {orch.token_usage}")

if __name__ == "__main__":
    main()
