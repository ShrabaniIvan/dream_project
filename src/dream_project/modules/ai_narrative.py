"""OpenAI narrative generation — reusable across all analysis modules."""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["dp_OPENAI_API_KEY"])
    return _client


def generate_narrative(prompt: str, model: str | None = None) -> str:
    """Send prompt to OpenAI and return the narrative text."""
    model = model or os.environ.get("dp_OPENAI_MODEL", "gpt-4o-mini")
    max_tokens = int(os.environ.get("dp_OPENAI_MAX_TOKENS", 1000))
    response = _get_client().chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a statistical analyst assistant. Interpret statistical results "
                    "clearly and concisely for a domain expert audience. "
                    "Always respond as a numbered list."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content
