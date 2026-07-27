import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )

    elif provider == "openrouter":
        return ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )

    raise ValueError(f"Unknown provider: {provider}")

def get_text(response):
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    text.append(part["text"])

            elif hasattr(part, "text"):
                text.append(part.text)

            else:
                text.append(str(part))

        return "".join(text).strip()
    return str(content).strip()