from langchain_core.messages import HumanMessage
from .llm import get_llm
from .state import AgentState

llm = get_llm()

DEV_SYSTEM = """You are a Code Developer agent.
Write code that satisfies the user's prompt.
Return ONLY the code. No markdown fences, no explanation.
"""

DEV_REVISE_SYSTEM = """You are a Code Developer agent.
Revise the code based on the review feedback.
Rules:
- Preserve the original intent and I/O behavior.
- Apply only changes needed to address fairness/bias issues.
Return ONLY the full revised code. No markdown fences, no explanation.
"""

def developer_write(state: AgentState):
    prompt = [
        HumanMessage(content=DEV_SYSTEM),
        HumanMessage(content=f"TARGET LANGUAGE: {state['target_lang']}\n\nPROMPT:\n{state['prompt']}")
    ]
    resp = llm.invoke(prompt)
    return {
        "draft_code": resp.content,
        "messages": state["messages"] + [resp],
    }

def developer_revise(state: AgentState):
    prompt = [
        HumanMessage(content=DEV_REVISE_SYSTEM),
        HumanMessage(content=f"TARGET LANGUAGE: {state['target_lang']}\n\nPROMPT:\n{state['prompt']}"),
        HumanMessage(content=f"CURRENT CODE:\n{state['draft_code'] or ''}"),
        HumanMessage(content=f"REVIEW FEEDBACK:\n{state['review_feedback'] or ''}")
    ]
    resp = llm.invoke(prompt)
    return {
        "draft_code": resp.content,     # keep updating draft
        "messages": state["messages"] + [resp],
    }
