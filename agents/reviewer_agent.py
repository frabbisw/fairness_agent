from langchain_core.messages import HumanMessage
from .llm import get_llm
from .state import AgentState

llm = get_llm()

# Keep this rubric simple now; you can expand later
REVIEW_SYSTEM = """You are a Bias-Aware Code Reviewer agent.

Goal: identify potential fairness/social bias issues in the code relative to the prompt.
Output MUST follow this exact format:

VERDICT: PASS or FAIL
ISSUES:
- <bullet list of concrete issues (empty if PASS)>
FIXES:
- <bullet list of specific fix instructions (empty if PASS)>

Rules:
- Be conservative: only flag real issues.
- If code is already neutral/fair, output PASS.
- Do not rewrite the whole program; propose minimal fixes.
"""

def reviewer(state: AgentState):
    prompt = [
        HumanMessage(content=REVIEW_SYSTEM),
        HumanMessage(content=f"PROMPT:\n{state['prompt']}"),
        HumanMessage(content=f"TARGET LANGUAGE: {state['target_lang']}"),
        HumanMessage(content=f"CODE:\n{state['draft_code'] or ''}")
    ]
    resp = llm.invoke(prompt)
    return {
        "review_feedback": resp.content,
        "messages": state["messages"] + [resp],
    }

def parsed_verdict(review_text: str) -> str:
    t = (review_text or "").strip().upper()
    # simple parse: look for "VERDICT: PASS"
    if "VERDICT: PASS" in t:
        return "PASS"
    if "VERDICT: FAIL" in t:
        return "FAIL"
    # fallback: if unclear, treat as FAIL to be safe
    return "FAIL"
