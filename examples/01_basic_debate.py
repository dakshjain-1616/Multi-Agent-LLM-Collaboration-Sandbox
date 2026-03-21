"""
Example 1: Basic debate using the built-in Debate scenario.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python examples/01_basic_debate.py

Without an API key the script runs in dry-run mode with mock responses.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sandbox import AgentOrchestrator, Agent, get_scenario

# ── mock helper (used when no real API key is set) ──────────────────────────
def _mock_orchestrator(orch):
    """Patch the orchestrator's client with canned responses so no key is needed."""
    from unittest.mock import MagicMock
    responses = [
        "Regulation is essential — it protects citizens from algorithmic harm.",
        "Heavy-handed rules stifle innovation and push development offshore.",
        "Both raise valid points. Can light-touch regulation balance safety and speed?",
        "Targeted regulation on high-risk deployments is a workable middle ground.",
        "Even 'targeted' rules are politically contested and technically hard to define.",
        "The debate highlights a genuine tension: safety vs innovation speed.",
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
# ────────────────────────────────────────────────────────────────────────────


def main():
    scenario = get_scenario("debate")
    agents = [Agent(**{k: a[k] for k in ("name", "role", "model")})
              for a in scenario.suggested_agents]

    orch = AgentOrchestrator(
        agents=agents,
        scenario_description=scenario.description,
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[dry-run] No OPENROUTER_API_KEY — using mock responses.\n")
        _mock_orchestrator(orch)

    def on_msg(msg):
        print(f"\n[{msg.agent_name}]\n{msg.content}")

    print(f"Topic: {scenario.initial_prompt}\n{'─'*60}")
    orch.run_scenario(scenario.initial_prompt, num_rounds=1, on_message=on_msg)

    if orch.summary:
        print(f"\n{'─'*60}\nSUMMARY\n{orch.summary}")


if __name__ == "__main__":
    main()
