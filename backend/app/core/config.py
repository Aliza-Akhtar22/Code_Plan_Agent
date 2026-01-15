from __future__ import annotations

import os
from dotenv import load_dotenv

# Load backend/.env (current working directory should be backend/)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def require_openai_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing. Put it in backend/.env")
    return OPENAI_API_KEY
