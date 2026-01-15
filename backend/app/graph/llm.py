from __future__ import annotations

from typing import Any, Dict, Optional

from openai import OpenAI

from app.core.config import OPENAI_MODEL, require_openai_key


def _client() -> OpenAI:
    api_key = require_openai_key()
    return OpenAI(api_key=api_key)


async def chat_json(system: str, user: str) -> Dict[str, Any]:
    """
    Returns parsed JSON from the model. Assumes the prompt requests strict JSON.
    """
    client = _client()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content or "{}"

    # Very small guard: strip code fences if the model adds them
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        # crude: remove language token if present
        lines = content.splitlines()
        if lines and lines[0].strip().lower() in ("json", "python"):
            lines = lines[1:]
        content = "\n".join(lines).strip()

    import json
    return json.loads(content)


async def chat_text(system: str, user: str) -> str:
    client = _client()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    return (resp.choices[0].message.content or "").strip()
