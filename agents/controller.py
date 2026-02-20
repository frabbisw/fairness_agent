from .state import AgentState
from .reviewer_agent import parsed_verdict

def controller(state: AgentState):
    """
    Deterministic routing for simplicity:
    - If no draft -> developer_write
    - Else -> reviewer
    - If PASS -> finish
    - If FAIL and iterations left -> developer_revise then loop
    - If FAIL and no iterations left -> finish (best effort)
    """
    if state["draft_code"] is None:
        nxt = "dev_write"
    elif state["review_feedback"] is None:
        nxt = "review"
    else:
        verdict = parsed_verdict(state["review_feedback"])
        if verdict == "PASS":
            nxt = "finish"
        else:
            if state["iteration"] >= state["max_iter"]:
                nxt = "finish"
            else:
                nxt = "dev_revise"

    return {"next": nxt}
