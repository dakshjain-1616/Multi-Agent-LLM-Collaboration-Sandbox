"""
Example 4: Save a checkpoint mid-conversation and resume it later.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python examples/04_checkpoint_resume.py

Without an API key the script runs in dry-run mode with mock responses.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sandbox import Agent, AgentOrchestrator

def _mock_orchestrator(orch):
    from unittest.mock import MagicMock
    responses = [
        "Consciousness might be an emergent property of sufficiently complex information processing.",
        "Or perhaps it's fundamental — a basic feature of the universe, like mass or charge.",
        "Does the hard problem of consciousness even have a scientific solution?",
        "Maybe the question itself is malformed — consciousness resists third-person description by nature.",
        "Could a sufficiently detailed simulation of a brain be conscious?",
        "If it processes information identically, pragmatically yes — though we can never verify subjectively.",
    ]
    call_idx = {"i": 0}
    def mock_create(**kwargs):
        idx = call_idx["i"] % len(responses)
        call_idx["i"] += 1
        r = MagicMock()
        r.choices[0].message.content = responses[idx]
        r.usage.prompt_tokens = 90
        r.usage.completion_tokens = 45
        return r
    orch._client.chat.completions.create = mock_create
    return orch

def main():
    agents = [
        Agent(name="Alice", role="You are a curious scientist. Ask probing questions. 1-2 sentences."),
        Agent(name="Bob",   role="You are a wise philosopher. Give thoughtful answers. 1-2 sentences."),
    ]

    orch = AgentOrchestrator(agents=agents, api_key=os.environ.get("OPENROUTER_API_KEY", ""))

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[dry-run] No OPENROUTER_API_KEY — using mock responses.\n")
        _mock_orchestrator(orch)

    def on_msg(msg):
        print(f"  [{msg.agent_name}] {msg.content}")

    print("=== Round 1 (2 turns) ===")
    orch.run_scenario("What is the nature of consciousness?", num_rounds=1, on_message=on_msg)

    checkpoint = tempfile.mktemp(suffix=".json")
    orch.save_checkpoint(checkpoint)
    print(f"\n✓ Checkpoint saved: {checkpoint}")
    print(f"  Messages so far: {len(orch.history)}")

    print("\n=== Resuming from checkpoint (1 more turn) ===")
    resumed = AgentOrchestrator.load_checkpoint(checkpoint)
    if not os.environ.get("OPENROUTER_API_KEY"):
        _mock_orchestrator(resumed)
    resumed.run_round()
    for msg in resumed.history[-len(agents):]:
        print(f"  [{msg.agent_name}] {msg.content}")

    print(f"\n✓ Total messages after resume: {len(resumed.history)}")
    os.unlink(checkpoint)

if __name__ == "__main__":
    main()
