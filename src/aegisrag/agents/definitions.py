"""The five agents (§4 of the design doc) — deliberately five, not fifteen."""
from crewai import Agent

from aegisrag.agents.llm import get_llm


def build_agents() -> dict[str, Agent]:
    llm = get_llm()

    planner = Agent(
        role="Query Planner",
        goal="Classify the user's question and decide what to retrieve",
        backstory=(
            "You analyze incoming questions about an indexed document corpus and decide "
            "their intent and retrieval strategy before any search happens."
        ),
        llm=llm,
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
        llm=llm,
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
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    synthesizer = Agent(
        role="Synthesis Writer",
        goal="Write a grounded, cited answer strictly from approved evidence",
        backstory=(
            "You write answers exclusively from the evidence you're given, citing every "
            "claim, and you never introduce outside knowledge."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    citation_quality = Agent(
        role="Citation & Quality Reviewer",
        goal="Verify every claim is cited and score the answer's confidence",
        backstory=(
            "You are the last check before an answer reaches a user. You catch unsupported "
            "claims and assign a defensible confidence score."
        ),
        llm=llm,
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
