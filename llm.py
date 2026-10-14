# pyrefly: ignore [missing-import]
from langchain_openai import ChatOpenAI

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_URL,
    OPENROUTER_MODEL,
)


llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_URL,
    max_retries=6,
)
