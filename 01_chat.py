"""
Banking Seminar · Day 1 (LangChain) — Step 1: Talk to a model.

The tiniest possible step: send one message to the AI and print the reply.
Everything else today builds on this.

Run it:   python venv/01_chat.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"


def load_environment() -> None:
    """Load environment variables from the repo root and validate the key."""
    load_dotenv(ENV_PATH)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY. Create a .env file in the project root with:\n"
            "OPENROUTER_API_KEY=your_key_here"
        )



def build_model() -> ChatOpenAI:
    """Create the configured chat client once the environment is validated."""
    load_environment()
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,  # 0 = focused & repeatable
    )


def main() -> None:
    model = build_model()

    # Type your own question, or press Enter to use the example hint.
    EXAMPLE = "In one simple sentence, what is a savings account?"
    question = input(f"Ask the bank's AI anything [Enter for: {EXAMPLE}]: ").strip() or EXAMPLE

    reply = model.invoke(question)
    print("\n" + reply.content)


if __name__ == "__main__":
    main()

