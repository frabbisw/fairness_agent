from typing import Literal
from langgraph.graph import StateGraph, END

from .state import AgentState
from .controller import controller
from .developer_agent import developer_write, developer_revise
from .reviewer_agent import reviewer

def route(state: AgentState) -> Literal["dev_write", "review", "dev_revise", END]:
    if state["next"] == "finish":
        return END
    return state["next"]  # type: ignore

def build_app():
    g = StateGraph(AgentState)

    g.add_node("controller", controller)
    g.add_node("dev_write", developer_write)
    g.add_node("review", reviewer)
    g.add_node("dev_revise", developer_revise)

    g.set_entry_point("controller")

    g.add_conditional_edges("controller", route, {
        "dev_write": "dev_write",
        "review": "review",
        "dev_revise": "dev_revise",
        END: END
    })

    # loop back
    g.add_edge("dev_write", "controller")
    g.add_edge("review", "controller")
    g.add_edge("dev_revise", "controller")

    return g.compile()

def run_agentic(prompt: str, target_lang: str, max_iter: int = 2):
    app = build_app()
    init: AgentState = {
        "messages": [],
        "prompt": prompt,
        "target_lang": target_lang,
        "draft_code": None,
        "review_feedback": None,
        "final_code": None,
        "iteration": 0,
        "max_iter": max_iter,
        "next": "dev_write",
    }
    out = app.invoke(init)
    # final_code = last draft
    return {
        "code": out["draft_code"],
        "review_feedback": out["review_feedback"],
        "iterations": out["iteration"],
    }
