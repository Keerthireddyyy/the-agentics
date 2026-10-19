"""CrewAI starter: a three-agent crew (researcher -> writer -> editor).

Run:
    uv run python main.py "your topic here"
"""

import os
import sys
from pathlib import Path

from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# 1. MODEL
# ---------------------------------------------------------------------------

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

llm = LLM(
    model=f"openai/{MODEL}",
    base_url=os.getenv(
        "GROQ_BASE_URL",
        "https://api.groq.com/openai/v1",
    ),
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.3,
)


# ---------------------------------------------------------------------------
# 2. AGENTS
# ---------------------------------------------------------------------------

researcher = Agent(
    role="Research Analyst",
    goal="Collect the key facts and tensions about {topic}.",
    backstory=(
        "You are a careful research analyst who separates evidence "
        "from hype and clearly identifies uncertainty."
    ),
    llm=llm,
    allow_delegation=False,
)


writer = Agent(
    role="Technical Writer",
    goal=(
        "Write a clear, accurate, plain-language explanation of {topic} "
        "using the research provided by the Research Analyst."
    ),
    backstory=(
        "You are an experienced educational writer who explains difficult "
        "technical subjects to 10-year-old students. Use very simple words, "
        "short sentences, familiar examples, and clear analogies. Avoid "
        "unnecessary technical jargon while keeping the explanation accurate."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
    max_iter=3,
)


editor = Agent(
    role="Editor",
    goal=(
        "Tighten the draft, improve clarity, and flag claims "
        "that need a source."
    ),
    backstory=(
        "You are a careful editor. You remove filler, fix unclear "
        "sentences, preserve important facts, and mark unsupported "
        "claims with [source needed]."
    ),
    llm=llm,
    allow_delegation=False,
)


# ---------------------------------------------------------------------------
# 3. TASKS
# ---------------------------------------------------------------------------

research_task = Task(
    description=(
        "Research {topic}. "
        "Provide exactly 5 key facts and exactly 2 open questions. "
        "Keep the research factual and concise."
    ),
    expected_output=(
        "A concise bulleted research brief containing "
        "exactly 5 key facts and exactly 2 open questions."
    ),
    agent=researcher,
)


writing_task = Task(
    description=(
        "Using the Research Analyst's research brief, write an "
        "approximately 250-word explainer about {topic}.\n\n"
        "The explainer must:\n"
        "1. Define the topic in simple language.\n"
        "2. Explain how it works.\n"
        "3. Explain why it is useful.\n"
        "4. Mention important limitations or challenges.\n"
        "5. Use plain language and at least one simple analogy.\n\n"
        "Return ONLY the finished explainer. "
        "Do not return JSON, a list of instructions, or analysis."
    ),
    expected_output=(
        "A readable approximately 250-word plain-language explainer "
        "about the requested topic. The output must be normal text."
    ),
    agent=writer,
    context=[research_task],
)


editing_task = Task(
    description=(
        "Edit the Technical Writer's explainer.\n\n"
        "Tighten the writing, remove unnecessary repetition, "
        "correct unclear wording, and preserve the important facts.\n\n"
        "If a factual claim is not clearly supported by the research, "
        "mark it with [source needed].\n\n"
        "Return ONLY the final edited explainer."
    ),
    expected_output=(
        "The final edited explainer in plain text, with "
        "[source needed] flags where appropriate."
    ),
    agent=editor,
    context=[writing_task],
)


# ---------------------------------------------------------------------------
# 4. CREW
# ---------------------------------------------------------------------------

crew = Crew(
    agents=[
        researcher,
        writer,
        editor,
    ],
    tasks=[
        research_task,
        writing_task,
        editing_task,
    ],
    process=Process.sequential,
    verbose=True,
)


# ---------------------------------------------------------------------------
# 5. RUN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    topic = (
        " ".join(sys.argv[1:])
        or "how small teams can use AI agents in data science"
    )

    result = crew.kickoff(
        inputs={
            "topic": topic,
        }
    )

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "explainer.md"
    output_file.write_text(
        result.raw,
        encoding="utf-8",
    )

    print(
        "\n=== FINAL OUTPUT "
        "(also saved to output/explainer.md) ===\n"
    )
    print(result.raw)