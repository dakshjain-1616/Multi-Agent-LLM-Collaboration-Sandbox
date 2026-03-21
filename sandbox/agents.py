"""Agent system: Agent class, AgentOrchestrator, run_turn(), run_scenario()."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "deepseek/deepseek-chat")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
SUMMARY_TRIGGER_TURNS = int(os.environ.get("SUMMARY_TRIGGER_TURNS", "6"))


@dataclass
class Message:
    agent_name: str
    content: str
    role: str = "assistant"  # "user" for the initial prompt, "assistant" for agents

    def to_dict(self) -> Dict[str, str]:
        return {
            "agent_name": self.agent_name,
            "role": self.role,
            "content": self.content,
        }


@dataclass
class Agent:
    name: str
    role: str  # system prompt / persona description
    model: str = DEFAULT_MODEL
    temperature: float = 0.8
    max_tokens: int = 512

    def build_messages(
        self,
        history: List[Message],
        scenario_description: str = "",
    ) -> List[Dict[str, str]]:
        """
        Build the messages list for this agent's LLM call.
        The system prompt combines the scenario context and agent role.
        History is injected as alternating user/assistant messages so the
        agent sees the full conversation.
        """
        system_parts = []
        if scenario_description:
            system_parts.append(f"Scenario context: {scenario_description}")
        system_parts.append(self.role)
        system_content = "\n\n".join(system_parts)

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]

        for msg in history:
            if msg.agent_name == self.name:
                messages.append({"role": "assistant", "content": msg.content})
            else:
                label = f"[{msg.agent_name}]" if msg.agent_name else ""
                messages.append(
                    {"role": "user", "content": f"{label} {msg.content}".strip()}
                )

        return messages


class AgentOrchestrator:
    """Manages a group of agents and orchestrates multi-turn conversations."""

    def __init__(
        self,
        agents: Optional[List[Agent]] = None,
        api_key: Optional[str] = None,
        base_url: str = OPENROUTER_BASE_URL,
        scenario_description: str = "",
        summary_trigger_turns: int = SUMMARY_TRIGGER_TURNS,
    ):
        self.agents: List[Agent] = agents or []
        self.history: List[Message] = []
        self.scenario_description = scenario_description
        self.summary_trigger_turns = summary_trigger_turns
        self._summary: Optional[str] = None
        self._turn_count = 0
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0

        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._client = OpenAI(api_key=resolved_key, base_url=base_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_agent(self, agent: Agent) -> None:
        self.agents.append(agent)

    def set_agents(self, agents: List[Agent]) -> None:
        self.agents = list(agents)

    def reset(self) -> None:
        self.history.clear()
        self._summary = None
        self._turn_count = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def add_user_message(self, content: str, agent_name: str = "User") -> Message:
        """Add an initial user/human message to history."""
        msg = Message(agent_name=agent_name, content=content, role="user")
        self.history.append(msg)
        return msg

    def run_turn(self, agent: Agent) -> Message:
        """
        Run a single turn for the given agent: call the LLM, append response
        to history, and return the Message.
        """
        messages = agent.build_messages(self.history, self.scenario_description)
        response = self._client.chat.completions.create(
            model=agent.model,
            messages=messages,
            max_tokens=agent.max_tokens,
            temperature=agent.temperature,
        )
        content = response.choices[0].message.content or ""
        if response.usage:
            self._prompt_tokens += getattr(response.usage, "prompt_tokens", 0) or 0
            self._completion_tokens += getattr(response.usage, "completion_tokens", 0) or 0
        msg = Message(agent_name=agent.name, content=content, role="assistant")
        self.history.append(msg)
        self._turn_count += 1
        return msg

    def run_round(self) -> List[Message]:
        """Run one full round (each agent takes one turn) and return messages."""
        if not self.agents:
            raise ValueError("No agents configured.")
        results = []
        for agent in self.agents:
            msg = self.run_turn(agent)
            results.append(msg)
        return results

    def run_scenario(
        self,
        initial_prompt: str,
        num_rounds: int = 2,
        on_message=None,
    ) -> List[Message]:
        """
        Run a full scenario:
          1. Add the initial prompt as a user message.
          2. Run num_rounds rounds of agent turns.
          3. Auto-generate a summary if total turns >= summary_trigger_turns.

        on_message: optional callable(Message) for streaming callbacks.
        """
        if not initial_prompt.strip():
            raise ValueError("initial_prompt must not be empty.")

        self.add_user_message(initial_prompt)
        if on_message:
            on_message(self.history[-1])

        for _ in range(num_rounds):
            for agent in self.agents:
                msg = self.run_turn(agent)
                if on_message:
                    on_message(msg)

        if self._turn_count >= self.summary_trigger_turns:
            self._summary = self.generate_summary()

        return self.history

    def generate_summary(self) -> str:
        """Generate a concise summary of the conversation using the first agent's model."""
        if not self.agents:
            return ""
        if not self.history:
            return ""

        conversation_text = "\n".join(
            f"{m.agent_name}: {m.content}" for m in self.history
        )
        summary_prompt = (
            "You are a neutral summariser. Read the following multi-agent conversation "
            "and produce a concise summary (3-5 bullet points) of the key points raised, "
            "agreements reached, and open questions remaining.\n\n"
            f"CONVERSATION:\n{conversation_text}\n\nSUMMARY:"
        )

        response = self._client.chat.completions.create(
            model=self.agents[0].model,
            messages=[
                {"role": "system", "content": "You summarise conversations concisely."},
                {"role": "user", "content": summary_prompt},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""

    @property
    def summary(self) -> Optional[str]:
        return self._summary

    @property
    def token_usage(self) -> Dict[str, int]:
        return {
            "prompt": self._prompt_tokens,
            "completion": self._completion_tokens,
            "total": self._prompt_tokens + self._completion_tokens,
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save current history + agent config to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.export(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        api_key: Optional[str] = None,
        base_url: str = OPENROUTER_BASE_URL,
    ) -> "AgentOrchestrator":
        """Restore orchestrator state from a checkpoint file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_export(data, api_key=api_key, base_url=base_url)

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export(self) -> Dict[str, Any]:
        """Export conversation to a JSON-serialisable dict."""
        return {
            "agents": [
                {
                    "name": a.name,
                    "role": a.role,
                    "model": a.model,
                    "temperature": a.temperature,
                    "max_tokens": a.max_tokens,
                }
                for a in self.agents
            ],
            "scenario_description": self.scenario_description,
            "history": [m.to_dict() for m in self.history],
            "summary": self._summary,
            "turn_count": self._turn_count,
        }

    def export_json(self) -> str:
        return json.dumps(self.export(), indent=2, ensure_ascii=False)

    def save_export(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.export_json())

    @classmethod
    def from_export(
        cls,
        data: Dict[str, Any],
        api_key: Optional[str] = None,
        base_url: str = OPENROUTER_BASE_URL,
    ) -> "AgentOrchestrator":
        agents = [
            Agent(
                name=a["name"],
                role=a["role"],
                model=a.get("model", DEFAULT_MODEL),
                temperature=a.get("temperature", 0.8),
                max_tokens=a.get("max_tokens", 512),
            )
            for a in data.get("agents", [])
        ]
        orchestrator = cls(
            agents=agents,
            api_key=api_key,
            base_url=base_url,
            scenario_description=data.get("scenario_description", ""),
        )
        for m in data.get("history", []):
            orchestrator.history.append(
                Message(
                    agent_name=m.get("agent_name", ""),
                    content=m.get("content", ""),
                    role=m.get("role", "assistant"),
                )
            )
        orchestrator._summary = data.get("summary")
        orchestrator._turn_count = data.get("turn_count", 0)
        return orchestrator
