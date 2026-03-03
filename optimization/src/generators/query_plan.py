"""Query Plan CoT Generator.

Based on CHASE-SQL Section 3.3 / Appendix Fig. 18.
Simulates database execution plan reasoning steps.
"""

import random
from pathlib import Path

from openai import AsyncOpenAI

PROMPT_TEMPLATE = (Path(__file__).resolve().parent.parent.parent / "prompts" / "query_plan.txt").read_text()


def _shuffle_schema(schema: str) -> str:
    """Randomly shuffle table order in schema to increase diversity."""
    blocks = schema.strip().split("\n\n")
    random.shuffle(blocks)
    return "\n\n".join(blocks)


def build_messages(
    schema: str,
    question: str,
    evidence: str = "",
    value_hint: str = "",
    shuffle: bool = False,
) -> list[dict]:
    """Build chat messages for Query Plan CoT generation."""
    if shuffle:
        schema = _shuffle_schema(schema)

    prompt = PROMPT_TEMPLATE.format(
        schema=schema,
        question=question,
        evidence=evidence,
        value_hint=value_hint,
    )

    return [{"role": "user", "content": prompt}]


async def generate_candidates(
    client: AsyncOpenAI,
    model: str,
    schema: str,
    question: str,
    evidence: str = "",
    value_hint: str = "",
    n_candidates: int = 7,
    temperature: float = 0.8,
    max_tokens: int = 4096,
) -> list[str]:
    """Generate multiple SQL candidates using Query Plan CoT.

    Returns list of raw model outputs.
    """
    import asyncio

    async def _gen_one(i: int) -> str:
        shuffle = i > 0
        messages = build_messages(schema, question, evidence, value_hint, shuffle=shuffle)
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"ERROR: {e}"

    return await asyncio.gather(*[_gen_one(i) for i in range(n_candidates)])
