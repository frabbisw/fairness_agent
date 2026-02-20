from typing import TypedDict, Optional, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # chat traces (optional, but helpful for debugging/research)
    messages: List[BaseMessage]

    # task inputs
    prompt: str
    target_lang: str

    # artifacts
    draft_code: Optional[str]
    review_feedback: Optional[str]
    final_code: Optional[str]

    # control
    iteration: int
    max_iter: int
    next: str
