"""Preset scenario definitions for multi-agent collaboration."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Scenario:
    key: str
    name: str
    description: str
    suggested_agents: List[Dict[str, str]] = field(default_factory=list)
    initial_prompt: str = ""


SCENARIOS: Dict[str, Scenario] = {
    "debate": Scenario(
        key="debate",
        name="Debate",
        description=(
            "Two or more agents take opposing positions on a topic and argue their "
            "case with evidence and reasoning. Each agent defends their stance while "
            "acknowledging the strongest counterarguments."
        ),
        suggested_agents=[
            {
                "name": "Advocate",
                "role": (
                    "You are a passionate advocate arguing FOR the given proposition. "
                    "Present strong arguments, cite reasoning and evidence, and rebut "
                    "counterarguments thoughtfully. Keep responses concise (3-5 sentences)."
                ),
                "model": "deepseek/deepseek-chat",
            },
            {
                "name": "Skeptic",
                "role": (
                    "You are a critical skeptic arguing AGAINST the given proposition. "
                    "Challenge assumptions, point out flaws in reasoning, and present "
                    "alternative perspectives. Keep responses concise (3-5 sentences)."
                ),
                "model": "deepseek/deepseek-chat",
            },
            {
                "name": "Moderator",
                "role": (
                    "You are a neutral moderator. Summarise the key points raised, "
                    "identify areas of agreement and disagreement, and pose a follow-up "
                    "question to deepen the discussion. Keep responses concise."
                ),
                "model": "deepseek/deepseek-chat",
            },
        ],
        initial_prompt="Topic: Should artificial intelligence be regulated by governments?",
    ),
    "code_review": Scenario(
        key="code_review",
        name="Code Review",
        description=(
            "A developer presents code or a design, and multiple agents review it from "
            "different angles: correctness, performance, security, and maintainability. "
            "Agents collaborate to produce actionable feedback."
        ),
        suggested_agents=[
            {
                "name": "Developer",
                "role": (
                    "You are the developer who wrote the code. Explain your design "
                    "decisions, respond to feedback, and propose revisions. Be receptive "
                    "to critique and ask clarifying questions when needed."
                ),
                "model": "deepseek/deepseek-chat",
            },
            {
                "name": "SecurityReviewer",
                "role": (
                    "You are a security engineer reviewing code for vulnerabilities. "
                    "Focus on injection risks, authentication flaws, data exposure, and "
                    "insecure dependencies. Provide specific, actionable recommendations."
                ),
                "model": "deepseek/deepseek-chat",
            },
            {
                "name": "PerformanceReviewer",
                "role": (
                    "You are a performance engineer. Look for algorithmic inefficiencies, "
                    "memory leaks, unnecessary I/O, and scalability bottlenecks. Suggest "
                    "concrete optimisations with trade-off analysis."
                ),
                "model": "deepseek/deepseek-chat",
            },
        ],
        initial_prompt=(
            "Please review this Python function that fetches user data:\n\n"
            "```python\ndef get_user(user_id):\n    query = f\"SELECT * FROM users WHERE id={user_id}\"\n"
            "    return db.execute(query).fetchone()\n```"
        ),
    ),
    "story_writing": Scenario(
        key="story_writing",
        name="Story Writing",
        description=(
            "Multiple agents collaborate to write a story together. Each agent plays a "
            "distinct creative role — plot architect, character writer, world-builder — "
            "contributing to an evolving narrative in a round-robin fashion."
        ),
        suggested_agents=[
            {
                "name": "PlotArchitect",
                "role": (
                    "You are the plot architect. Drive the narrative forward with "
                    "compelling story beats, introduce conflict and tension, and ensure "
                    "the plot has momentum. Write 2-3 sentences advancing the story."
                ),
                "model": "deepseek/deepseek-chat",
            },
            {
                "name": "CharacterWriter",
                "role": (
                    "You are the character writer. Develop characters' voices, motivations, "
                    "and relationships. Write dialogue and internal thoughts that reveal "
                    "personality. Add 2-3 sentences from a character's perspective."
                ),
                "model": "deepseek/deepseek-chat",
            },
            {
                "name": "WorldBuilder",
                "role": (
                    "You are the world-builder. Enrich the setting with vivid sensory "
                    "details, lore, and atmosphere. Ground each scene in a specific place "
                    "and time. Add 2-3 sentences of descriptive world detail."
                ),
                "model": "deepseek/deepseek-chat",
            },
        ],
        initial_prompt=(
            "Let's write a collaborative science fiction story. "
            "Begin with: A lone astronaut discovers an abandoned space station "
            "orbiting a gas giant no human has visited before."
        ),
    ),
    "planning_session": Scenario(
        key="planning_session",
        name="Planning Session",
        description=(
            "Agents with different expertise (technical, business, risk) collaborate to "
            "plan a project or initiative. They identify goals, constraints, risks, and "
            "action items, converging on a concrete plan."
        ),
        suggested_agents=[
            {
                "name": "TechLead",
                "role": (
                    "You are the technical lead. Assess feasibility, identify technical "
                    "dependencies and risks, estimate complexity, and propose implementation "
                    "approaches. Be specific about technical constraints."
                ),
                "model": "deepseek/deepseek-chat",
            },
            {
                "name": "ProductManager",
                "role": (
                    "You are the product manager. Define user value, prioritise features, "
                    "manage scope, and keep the discussion focused on business outcomes. "
                    "Ask clarifying questions about user needs and success metrics."
                ),
                "model": "deepseek/deepseek-chat",
            },
            {
                "name": "RiskAnalyst",
                "role": (
                    "You are the risk analyst. Identify potential failure modes, edge cases, "
                    "and mitigation strategies. Think about what could go wrong and propose "
                    "contingency plans. Be systematic and thorough."
                ),
                "model": "deepseek/deepseek-chat",
            },
        ],
        initial_prompt=(
            "We need to plan the launch of a new mobile app feature: "
            "real-time collaborative document editing for up to 50 simultaneous users. "
            "Target launch: 3 months. Let's discuss the approach."
        ),
    ),
    "brainstorm": Scenario(
        key="brainstorm",
        name="Brainstorm",
        description=(
            "A creative brainstorming session where agents generate wild ideas, filter "
            "them for feasibility and novelty, then synthesise the best into a coherent proposal."
        ),
        suggested_agents=[
            {
                "name": "IdeaGenerator",
                "role": (
                    "You are a radical creative thinker. Generate wild, unconventional ideas "
                    "without self-censoring. Quantity over quality — the more surprising the better. "
                    "List 3-5 distinct ideas per response, each in one sentence."
                ),
                "model": "deepseek/deepseek-chat",
            },
            {
                "name": "CriticFilter",
                "role": (
                    "You are a pragmatic evaluator. Review the ideas presented and score each on "
                    "feasibility (can it be built?) and novelty (is it truly new?). Keep only the "
                    "best 3 ideas, explaining briefly why each survives the cut."
                ),
                "model": "deepseek/deepseek-chat",
            },
            {
                "name": "Synthesiser",
                "role": (
                    "You are a strategic synthesiser. Take the surviving ideas and combine them "
                    "into one coherent, actionable proposal. Describe the combined concept, its "
                    "core value proposition, and a first concrete step to prototype it."
                ),
                "model": "deepseek/deepseek-chat",
            },
        ],
        initial_prompt="Brainstorm innovative solutions for reducing urban food waste using AI.",
    ),
    "socratic": Scenario(
        key="socratic",
        name="Socratic Dialogue",
        description=(
            "A Socratic dialogue where one agent probes assumptions with sharp questions and "
            "the other answers thoughtfully, revising their position when challenged."
        ),
        suggested_agents=[
            {
                "name": "Questioner",
                "role": (
                    "You are a Socratic questioner. Ask probing questions that expose hidden "
                    "assumptions, challenge definitions, and push the Responder to think more "
                    "deeply. Ask one focused question per turn — never assert, only question."
                ),
                "model": "deepseek/deepseek-chat",
            },
            {
                "name": "Responder",
                "role": (
                    "You are a thoughtful Responder engaged in a Socratic dialogue. Answer the "
                    "Questioner's questions carefully, and be willing to revise or abandon your "
                    "position when a question reveals a flaw in your reasoning."
                ),
                "model": "deepseek/deepseek-chat",
            },
        ],
        initial_prompt="Let's explore: Is free will compatible with a deterministic universe?",
    ),
}


def get_scenario(key: str) -> Scenario:
    """Return a scenario by key, raising KeyError if not found."""
    if key not in SCENARIOS:
        raise KeyError(f"Unknown scenario: {key!r}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[key]


def list_scenarios() -> List[str]:
    """Return list of available scenario keys."""
    return list(SCENARIOS.keys())


def get_scenario_names() -> Dict[str, str]:
    """Return mapping of key -> display name."""
    return {key: scenario.name for key, scenario in SCENARIOS.items()}
