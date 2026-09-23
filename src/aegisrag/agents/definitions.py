"""The five agents (§4 of the design doc) — deliberately five, not fifteen.

Each agent gets its own LLM instance with a max_tokens budget sized to what it actually needs
to produce — planner/evidence-validator/citation-quality are short JSON, synthesis is the only
step that needs real room for prose. Capping the short steps is a real, measured latency win:
on a CPU-only 7B model, generation is the dominant cost per call, so an uncapped step that
occasionally rambles past what it needs directly adds seconds to every single query.

model_tiering (agent_policy.yaml) goes a step further than the token cap: the three short
structured-JSON agents can also run on a smaller model (`light_llm_factory`) than Synthesis,
the one step that actually needs generative capability — trading a little classification
accuracy for real latency, on the steps where that trade is cheap.
"""
from typing import Callable

from crewai import LLM, Agent

from aegisrag.agents.llm import get_llm

LLMFactory = Callable[..., LLM]


def build_agents(llm_factory: LLMFactory = get_llm, light_llm_factory: LLMFactory | None = None) -> dict[str, Agent]:
    # Falls back to the primary factory when tiering is off / not applicable (e.g. building the
    # fallback-model agent set, which intentionally never tiers down — escalation means "try a
    # more capable model," the opposite of what the light tier is for).
    light = light_llm_factory or llm_factory

    planner = Agent(
        role="Query Planner",
        goal="Classify the user's question and decide what to retrieve",
        backstory=(
            "You analyze incoming questions about an indexed document corpus and decide "
            "their intent and retrieval strategy before any search happens."
        ),
        llm=light(max_tokens=200),
        verbose=False,
        allow_delegation=False,
    )

    retriever = Agent(
        role="Retrieval Specialist",
        goal="Retrieve the most relevant evidence passages for a query",
        backstory=(
            "You run hybrid (vector + lexical) search over the document index and can "
            "reformulate a query when the first pass doesn't find enough evidence."
        ),
        llm=llm_factory(max_tokens=200),
        verbose=False,
        allow_delegation=False,
    )

    evidence_validator = Agent(
        role="Evidence Validator",
        goal="Judge whether retrieved passages actually support answering the question",
        backstory=(
            "You are skeptical by default. You classify evidence as supported, "
            "partially_supported, contradictory, or insufficient — never assume more "
            "coverage than what's actually in the passages."
        ),
        # Deliberately NOT `light`, even though this agent's own output is short JSON like
        # planner/citation_quality: measured on this corpus, the light model's judgment on
        # this specific task was unreliable (false "insufficient" verdicts triggered extra
        # retries and a worse outcome, net slower) — see orchestrator.py's
        # _LIGHT_TIER_AGENTS comment. Token budget alone isn't the deciding factor for tiering;
        # how failure-tolerant the task is, is.
        llm=llm_factory(max_tokens=350),
        verbose=False,
        allow_delegation=False,
    )

    synthesizer = Agent(
        role="Synthesis Writer",
        goal="Write a grounded, cited answer strictly from approved evidence",
        backstory=(
            "You write answers exclusively from the evidence you're given, citing every "
            "claim, and you never introduce outside knowledge. Be complete but not padded — "
            "say what the evidence supports, plainly, without repeating yourself."
        ),
        llm=llm_factory(max_tokens=700),  # always the primary model — the one step tiering never touches
        verbose=False,
        allow_delegation=False,
    )

    citation_quality = Agent(
        role="Citation & Quality Reviewer",
        goal="Verify every claim is cited and score the answer's confidence",
        backstory=(
            "You are the last check before an answer reaches a user. You catch unsupported "
            "claims and assign a defensible confidence score. You only rewrite the answer "
            "when something is actually wrong with it — otherwise you leave it alone."
        ),
        # Deliberately NOT `light`: tried it, and on this corpus llama3.2:1b returned an
        # unreliable confidence score (0.00 on a genuinely answerable question) rather than a
        # defensible one — the exact score this agent exists to protect. Excluded for the same
        # reason as Evidence Validator: this is a judgment/scoring gate the rest of the system
        # trusts (human-review routing, auto vs. qualified-answer threshold), not a formatting
        # task, so a small accuracy loss here has real downstream consequences a token-budget
        # cut on Planner doesn't.
        llm=llm_factory(max_tokens=350),
        verbose=False,
        allow_delegation=False,
    )

    return {
        "planner": planner,
        "retriever": retriever,
        "evidence_validator": evidence_validator,
        "synthesizer": synthesizer,
        "citation_quality": citation_quality,
    }
