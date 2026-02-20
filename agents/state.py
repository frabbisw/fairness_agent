# agents/state.py
from typing import TypedDict, Optional, List, Dict, Any
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: List[BaseMessage]

    prompt: str
    target_lang: str

    draft_code: Optional[str]

    # NEW: bias info coming from your local parser (dict or str)
    bias_info: Optional[Dict[str, Any]]  # or Optional[str]

    review_feedback: Optional[str]

    iteration: int
    max_iter: int
    next: str
