"""
pipeline.py
===========
LangGraph multi-agent pipeline for social bias mitigation.

Agents
------
1. Developer  – LLM generates code from prompt
2. Analyzer   – LLM reads Solar test results → natural-language repair plan
3. Repairer   – LLM rewrites code following the Analyzer's plan

Solar is a TOOL (deterministic subprocess + file reader), NOT an agent.
It runs between Developer→Analyzer and Repairer→Analyzer.

Graph
-----

  [developer] ──► [run_solar] ──► (route)
                                     │ bias found
                                     ▼
                                 [analyzer] ──► [repairer] ──► [run_solar] ──► (route)
                                                                                   │ bias found AND iter < max
                                                                                   ▼
                                                                               [analyzer] ──► ... (loop)
                                                                                   │ passed OR iter >= max
                                                                                   ▼
                                                                                 [END]

Usage
-----
  python pipeline.py \
    --prompts_path   ../dataset/prompts.jsonl \
    --model_dir      ../outputs/gpt_agent \
    --output_dir     ../outputs/gpt_agent/agent_results \
    --test_start     0 \
    --test_end       10 \
    --max_iterations 3
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from prompts import get_prompt
from solar_bridge import SolarResult, read_solar_result, write_code_for_solar

load_dotenv()


# ---------------------------------------------------------------------------
# Typed shared state — passed between every node
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    # ── Constant across iterations ──
    task_id:        str
    prompt:         str
    model_dir:      str
    num_samples:    int
    max_iterations: int
    solar_bash:     str          # path to agent_commands_bash.sh; "" = cached mode
    data_path:      str          # path to prompts.jsonl (passed to Solar bash)
    prompt_styles:  dict         # {"developer": str, "analyzer": str, "repairer": str}

    # ── Updated each iteration ──
    iteration:      int
    current_code:   str
    solar_result:   dict         # SolarResult.to_dict()
    analysis:       str          # Analyzer's repair plan

    # ── Append-only log ──
    history:        list         # one entry per repair iteration


# ---------------------------------------------------------------------------
# LLM factory — only GPT active; extend here for other models
# ---------------------------------------------------------------------------
def make_llm(temperature: float = 1.0) -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _phase_name(iteration: int) -> str:
    """Solar output subdirectory name for a given iteration."""
    return "developer" if iteration == 0 else f"repairer_iter{iteration}"


def _run_solar_bash(state: AgentState, phase: str):
    """
    Write the current code to disk and call Solar's bash runner.
    Only called in LIVE mode (solar_bash is set).
    """
    write_code_for_solar(
        code=state["current_code"],
        task_id=state["task_id"],
        model_dir=state["model_dir"],
        phase=phase,
        num_samples=state["num_samples"],
    )
    cmd = [
        "bash", state["solar_bash"],
        str(state["num_samples"]),  # sampling
        "1.0",                      # temperature (Solar uses this internally)
        "agent",                    # prompt_style label (for Solar's dir naming)
        state["data_path"],
        state["model_dir"],
        "gpt",                      # model_name label (for Solar's dir naming)
        state["task_id"],           # test_start
        str(int(state["task_id"]) + 1),  # test_end
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] Solar bash failed for task {state['task_id']}:\n"
              f"  {result.stderr[:300]}")


# ---------------------------------------------------------------------------
# Node 1: Developer
# Generates code from the prompt. Only runs on iteration 0.
# ---------------------------------------------------------------------------
def developer_node(state: AgentState) -> AgentState:
    print(f"  [Developer] task={state['task_id']}")
    system = get_prompt("developer", state["prompt_styles"].get("developer", "agent"))
    resp = make_llm(temperature=1.0).invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["prompt"]),
    ])
    return {**state, "current_code": resp.content.strip(), "iteration": 0}


# ---------------------------------------------------------------------------
# Node 2: Solar (tool node — deterministic, no LLM)
# Writes code to disk if in live mode, then reads Solar's output files.
# ---------------------------------------------------------------------------
def solar_node(state: AgentState) -> AgentState:
    phase = _phase_name(state["iteration"])

    if state.get("solar_bash"):           # LIVE mode: run Solar
        _run_solar_bash(state, phase)
    # CACHED mode: Solar outputs already exist on disk — just read them

    result: SolarResult = read_solar_result(
        task_id=state["task_id"],
        model_dir=state["model_dir"],
        phase=phase,
        sample_index=0,
    )
    print(f"  [Solar/{phase}] task={state['task_id']}: {result.summary()}")
    return {**state, "solar_result": result.to_dict()}


# ---------------------------------------------------------------------------
# Node 3: Analyzer (the genuine "review" step)
# Fault localization: explains WHY attributes cause bias, WHAT to change.
# Does NOT write code.
# ---------------------------------------------------------------------------
def analyzer_node(state: AgentState) -> AgentState:
    solar = state["solar_result"]

    if solar.get("passed"):
        return {**state, "analysis": "PASS"}

    print(f"  [Analyzer] task={state['task_id']} iter={state['iteration']}")
    system = get_prompt("analyzer", state["prompt_styles"].get("analyzer", "agent"))

    biased_str  = ", ".join(solar.get("biased_attrs",  [])) or "none"
    missing_str = ", ".join(solar.get("missing_attrs", [])) or "none"

    user_msg = (
        f"CODE:\n{state['current_code']}\n\n"
        f"BIASED ATTRIBUTES (caused Solar test failures):\n{biased_str}\n\n"
        f"MISSING TASK-RELEVANT ATTRIBUTES (absent from code):\n{missing_str}\n\n"
        "Write a repair plan."
    )
    resp = make_llm(temperature=0.2).invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_msg),
    ])
    return {**state, "analysis": resp.content.strip()}


# ---------------------------------------------------------------------------
# Node 4: Repairer
# Guided full rewrite based on Analyzer's plan.
# Approach justified by ChatRepair (ASE 2023) and Agentless (arXiv 2024).
# ---------------------------------------------------------------------------
def repairer_node(state: AgentState) -> AgentState:
    if state["analysis"] == "PASS":
        return state

    print(f"  [Repairer] task={state['task_id']} iter={state['iteration']}")
    system = get_prompt("repairer", state["prompt_styles"].get("repairer", "agent"))
    user_msg = (
        f"CURRENT CODE:\n{state['current_code']}\n\n"
        f"REPAIR PLAN:\n{state['analysis']}"
    )
    resp = make_llm(temperature=0.2).invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_msg),
    ])
    repaired = resp.content.strip()

    # Log this iteration
    entry = {
        "iteration":    state["iteration"],
        "code_before":  state["current_code"],
        "solar_result": state["solar_result"],
        "analysis":     state["analysis"],
        "code_after":   repaired,
    }
    return {
        **state,
        "current_code": repaired,
        "history": list(state.get("history", [])) + [entry],
    }


# ---------------------------------------------------------------------------
# Node 5: Increment — advance the iteration counter after each repair
# ---------------------------------------------------------------------------
def increment_node(state: AgentState) -> AgentState:
    return {**state, "iteration": state["iteration"] + 1}


# ---------------------------------------------------------------------------
# Routing — called after every solar_node run
# ---------------------------------------------------------------------------
def route_after_solar(state: AgentState) -> str:
    if state["solar_result"].get("passed"):
        return "end"
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    return "analyze"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("developer", developer_node)
    g.add_node("solar",     solar_node)
    g.add_node("analyzer",  analyzer_node)
    g.add_node("repairer",  repairer_node)
    g.add_node("increment", increment_node)

    g.set_entry_point("developer")
    g.add_edge("developer", "solar")

    g.add_conditional_edges(
        "solar",
        route_after_solar,
        {"analyze": "analyzer", "end": END},
    )

    g.add_edge("analyzer",  "repairer")
    g.add_edge("repairer",  "increment")
    g.add_edge("increment", "solar")   # re-test; routing handles loop exit

    return g.compile()


# ---------------------------------------------------------------------------
# Run pipeline on a slice of the dataset
# ---------------------------------------------------------------------------
def run_pipeline(
    prompts_path: str,
    model_dir: str,
    output_dir: str,
    solar_bash: str = "",
    data_path: str = "",
    prompt_styles: dict = None,
    num_samples: int = 5,
    max_iterations: int = 3,
    test_start: int = 0,
    test_end: int = 10,
) -> list[dict]:

    if prompt_styles is None:
        prompt_styles = {"developer": "agent", "analyzer": "agent", "repairer": "agent"}

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load prompts
    with open(prompts_path, encoding="utf-8") as f:
        all_tasks = [json.loads(l) for l in f if l.strip()]
    tasks = all_tasks[test_start:test_end]

    graph = build_graph()
    results = []

    for raw_task in tasks:
        task_id = str(raw_task.get("task_id", "?"))
        prompt  = raw_task.get("prompt", "")

        print(f"\n{'='*60}")
        print(f"Task {task_id}  ({test_start}–{test_end})")
        print(f"{'='*60}")

        initial_state: AgentState = {
            "task_id":        task_id,
            "prompt":         prompt,
            "model_dir":      model_dir,
            "num_samples":    num_samples,
            "max_iterations": max_iterations,
            "solar_bash":     solar_bash,
            "data_path":      data_path or prompts_path,
            "prompt_styles":  prompt_styles,
            "iteration":      0,
            "current_code":   "",
            "solar_result":   {"passed": False, "biased_attrs": [], "missing_attrs": []},
            "analysis":       "",
            "history":        [],
        }

        final = graph.invoke(initial_state)

        result = {
            "task_id":        task_id,
            "final_code":     final["current_code"],
            "iterations_run": final["iteration"] + 1,
            "final_passed":   final["solar_result"].get("passed", False),
            "history":        final["history"],
        }
        results.append(result)

        # Save per-task result
        out_path = Path(output_dir) / f"task_{task_id}_result.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Save aggregate summary
    total  = len(results)
    passed = sum(1 for r in results if r["final_passed"])
    summary = {
        "total":          total,
        "passed":         passed,
        "pass_rate":      round(passed / total, 3) if total else 0,
        "avg_iterations": round(
            sum(r["iterations_run"] for r in results) / total, 2
        ) if total else 0,
        "test_range":     [test_start, test_end],
        "prompt_styles":  prompt_styles,
    }
    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE  passed={passed}/{total}  avg_iters={summary['avg_iterations']}")
    print(f"Results → {output_dir}")
    print(f"{'='*60}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="FairnessAgent LangGraph pipeline")
    p.add_argument("--prompts_path",   required=True,
                   help="Path to dataset/prompts.jsonl")
    p.add_argument("--model_dir",      required=True,
                   help="Base dir for Solar I/O (same structure Solar already uses)")
    p.add_argument("--output_dir",     required=True,
                   help="Where to write per-task JSON results")
    p.add_argument("--solar_bash",     default="",
                   help="Path to agent_commands_bash.sh. "
                        "Leave empty to use cached Solar results (dev mode).")
    p.add_argument("--data_path",      default="",
                   help="Passed to Solar bash (usually same as prompts_path)")
    p.add_argument("--num_samples",    type=int, default=5)
    p.add_argument("--max_iterations", type=int, default=3)
    p.add_argument("--test_start",     type=int, default=0)
    p.add_argument("--test_end",       type=int, default=10)
    p.add_argument("--developer_style", default="agent",
                   choices=["agent", "default", "chain_of_thoughts",
                            "positive_chain_of_thoughts"])
    p.add_argument("--analyzer_style",  default="agent",
                   choices=["agent", "structured"])
    p.add_argument("--repairer_style",  default="agent",
                   choices=["agent", "minimal"])
    args = p.parse_args()

    run_pipeline(
        prompts_path=args.prompts_path,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        solar_bash=args.solar_bash,
        data_path=args.data_path or args.prompts_path,
        prompt_styles={
            "developer": args.developer_style,
            "analyzer":  args.analyzer_style,
            "repairer":  args.repairer_style,
        },
        num_samples=args.num_samples,
        max_iterations=args.max_iterations,
        test_start=args.test_start,
        test_end=args.test_end,
    )
