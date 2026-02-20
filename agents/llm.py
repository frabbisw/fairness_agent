# agents/llm.py
import os
import yaml
from pathlib import Path
from typing import Any

# Load config once
_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_CONFIG = {}

if _CONFIG_PATH.exists():
    with open(_CONFIG_PATH, "r") as f:
        _CONFIG = yaml.safe_load(f) or {}

def _cfg(path: list[str], default=None):
    cur = _CONFIG
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def get_llm(agent: str | None = None, **overrides: Any):
    """
    agent: optional agent name (e.g., 'developer', 'reviewer')
    """
    # precedence: overrides > agent-specific config > global config > env > default
    provider = (
        overrides.get("provider")
        or _cfg([agent, "llm", "provider"])
        or _cfg(["llm", "provider"])
        or os.getenv("LLM_PROVIDER", "openai")
    ).lower()

    model = (
        overrides.get("model")
        or _cfg([agent, "llm", "model"])
        or _cfg(["llm", "model"])
        or os.getenv("LLM_MODEL", "gpt-4.1-mini")
    )

    temperature = float(
        overrides.get("temperature")
        or _cfg([agent, "llm", "temperature"])
        or _cfg(["llm", "temperature"])
        or os.getenv("TEMPERATURE", 0)
    )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        base_url = overrides.get("base_url") or os.getenv("OLLAMA_BASE_URL")
        kwargs = {"model": model, "temperature": temperature}
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOllama(**kwargs)

    if provider == "huggingface":
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch

        tok = AutoTokenizer.from_pretrained(model)
        mdl = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        gen = pipeline(
            "text-generation",
            model=mdl,
            tokenizer=tok,
            max_new_tokens=int(overrides.get("max_new_tokens", 1024)),
            do_sample=temperature > 0,
            temperature=temperature,
        )
        return HuggingFacePipeline(pipeline=gen)

    raise ValueError(f"Unknown provider: {provider}")
