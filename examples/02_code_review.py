"""
Example 2: Multi-agent code review of a real Python snippet.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python examples/02_code_review.py

Without an API key the script runs in dry-run mode with mock responses.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sandbox import AgentOrchestrator, Agent, get_scenario

CODE = """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id={user_id}"
    return db.execute(query).fetchone()
"""

def _mock_orchestrator(orch):
    from unittest.mock import MagicMock
    responses = [
        "SQL injection vulnerability: user_id is interpolated directly into the query. Use parameterised queries: `db.execute('SELECT * FROM users WHERE id=?', (user_id,))`.",
        "Agreed on the injection risk. Also: no error handling if `db` is None or the query fails. Wrap in try/except and validate user_id is an integer first.",
        "Both issues confirmed. Fix: parameterise the query, add type assertion, and handle db errors. Simple 5-line fix covers all three concerns.",
        "Summary: critical SQL injection, missing input validation, no error handling. Priority: fix injection immediately before deploying.",
    ]
    call_idx = {"i": 0}
    def mock_create(**kwargs):
        idx = call_idx["i"] % len(responses)
        call_idx["i"] += 1
        r = MagicMock()
        r.choices[0].message.content = responses[idx]
        r.usage.prompt_tokens = 100
        r.usage.completion_tokens = 60
        return r
    orch._client.chat.completions.create = mock_create
    return orch

def main():
    scenario = get_scenario("code_review")
    agents = [Agent(**{k: a[k] for k in ("name", "role", "model")})
              for a in scenario.suggested_agents]
    orch = AgentOrchestrator(agents=agents, scenario_description=scenario.description,
                             api_key=os.environ.get("OPENROUTER_API_KEY", ""))

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[dry-run] No OPENROUTER_API_KEY — using mock responses.\n")
        _mock_orchestrator(orch)

    prompt = f"Please review this Python function:\n```python{CODE}```"
    print(f"Reviewing code...\n{'─'*60}")

    def on_msg(msg):
        print(f"\n[{msg.agent_name}]\n{msg.content}")

    orch.run_scenario(prompt, num_rounds=1, on_message=on_msg)
    summary = orch.generate_summary()
    print(f"\n{'─'*60}\nSUMMARY\n{summary}")

if __name__ == "__main__":
    main()
