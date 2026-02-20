# agents/llm.py
import os
from typing import Any

def _get_env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v is not None else default

def get_llm(**overrides: Any):
    """
    Returns a LangChain chat model with a consistent interface: .invoke(messages)
    Controlled by env:
      LLM_PROVIDER: openai|anthropic|google|ollama|huggingface
      LLM_MODEL: model name/id
      TEMPERATURE: float
    """
    provider = (overrides.get("provider") or _get_env("LLM_PROVIDER", "openai")).lower()
    model = overrides.get("model") or _get_env("LLM_MODEL", "gpt-4.1-mini")
    temperature = float(overrides.get("temperature") or _get_env("TEMPERATURE", "0") or 0)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature)

    if provider == "google":
        # Gemini via langchain-google-genai
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    if provider == "ollama":
        # Local OSS models (e.g., codellama, llama3, deepseek-coder) via Ollama
        from langchain_ollama import ChatOllama
        # Optional: OLLAMA_BASE_URL if your ollama runs remote (depends on package/version)
        base_url = overrides.get("base_url") or _get_env("OLLAMA_BASE_URL")
        kwargs = {"model": model, "temperature": temperature}
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOllama(**kwargs)

    if provider == "huggingface":
        # Runs a local transformers pipeline (slower but fully local).
        # You’ll need `transformers` + a model available locally.
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch

        hf_id = model  # e.g., "codellama/CodeLlama-7b-Instruct-hf"
        tok = AutoTokenizer.from_pretrained(hf_id)
        mdl = AutoModelForCausalLM.from_pretrained(
            hf_id,
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
        # Note: HuggingFacePipeline is not “chat-native”; still works, but responses differ.
        return HuggingFacePipeline(pipeline=gen)

    raise ValueError(f"Unknown LLM_PROVIDER={provider}. Use openai|anthropic|google|ollama|huggingface.")
