"""
pipeline.py — FairnessAgent LangGraph pipeline

Flow: Developer → Solar → Analyzer → Repairer → Solar → [loop or END]

Run:
  cd fairness_agent_clean
  python pipeline.py \
    --prompts_path   ../dataset/prompts_sample.jsonl \
    --model_dir      ../outputs/agent_test \
    --output_dir     ../outputs/agent_test/results \
    --solar_bash     ../commands/agent_commands_bash.sh \
    --model_name     gpt \
    --sampling       5 \
    --temperature    1.0 \
    --max_iterations 3 \
    --test_start     0 \
    --test_end       5
"""

# ── stdlib (never hangs) ─────────────────────────────────────────────────────
import json
import logging
import os
import sys
from pathlib import Path
from typing import TypedDict

# ── logging FIRST — so every subsequent import step is visible ───────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("pipeline")

# ── .env ─────────────────────────────────────────────────────────────────────
log.info("Loading .env ...")
from dotenv import load_dotenv
load_dotenv()
log.info(".env loaded.")

# ── heavy imports ─────────────────────────────────────────────────────────────
log.info("Importing LangGraph (may take a few seconds on first run)...")
from langgraph.graph import StateGraph, END
log.info("Importing LangChain OpenAI...")
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
log.info("LangGraph/LangChain ready.")

# ── local modules ─────────────────────────────────────────────────────────────
from prompts import get_prompt
from solar_bridge import (
    SolarResult,
    write_code_for_solar,
    read_solar_result,
    run_solar_bash,
)
log.info("All imports done. Pipeline ready.")


# ── Shared state ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # Fixed for the whole run
    task_id:        str
    prompt:         str
    model_dir:      str
    output_dir:     str
    solar_bash:     str
    data_path:      str
    model_name:     str
    sampling:       int
    temperature:    float
    max_iterations: int
    prompt_styles:  dict   # {developer, analyzer, repairer}
    # Updated each iteration
    iteration:      int
    current_code:   str
    solar_result:   dict   # SolarResult.to_dict()
    analysis:       str
    # Append-only log
    history:        list


# ── LLM factory ───────────────────────────────────────────────────────────────
def make_llm(temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ── Node 1: Developer ─────────────────────────────────────────────────────────
def developer_node(state: AgentState) -> AgentState:
    log.info(f"[Developer] task={state['task_id']} — calling LLM...")
    system = get_prompt("developer", state["prompt_styles"]["developer"])
    resp = make_llm(state["temperature"]).invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["prompt"]),
    ])
    code = resp.content.strip()
    log.info(f"[Developer] task={state['task_id']} — {len(code)} chars, writing for Solar...")
    write_code_for_solar(
        code=code,
        task_id=state["task_id"],
        model_dir=state["model_dir"],
        phase="developer",
        num_samples=state["sampling"],
    )
    log.info(f"[Developer] task={state['task_id']} — done.")
    return {**state, "current_code": code, "iteration": 0}


# ── Node 2: Solar (deterministic tool — no LLM) ───────────────────────────────
def solar_node(state: AgentState) -> AgentState:
    phase   = "developer" if state["iteration"] == 0 else "repairer"
    task_id = state["task_id"]
    log.info(f"[Solar/{phase}] task={task_id} — running bash pipeline...")
    run_solar_bash(
        solar_bash=state["solar_bash"],
        data_path=state["data_path"],
        model_dir=state["model_dir"],
        model_name=state["model_name"],
        sampling=state["sampling"],
        temperature=state["temperature"],
        prompt_style=state["prompt_styles"]["developer"],
        test_start=int(task_id),
        test_end=int(task_id) + 1,
    )
    result: SolarResult = read_solar_result(
        task_id=task_id,
        model_dir=state["model_dir"],
        phase=phase,
        sample_index=0,
    )
    log.info(f"[Solar/{phase}] task={task_id}: {result.summary()}")
    return {**state, "solar_result": result.to_dict()}


# ── Node 3: Analyzer (fault localization — no code writing) ───────────────────
def analyzer_node(state: AgentState) -> AgentState:
    solar = state["solar_result"]
    if solar["passed"]:
        log.info(f"[Analyzer] task={state['task_id']} — PASS, skipping.")
        return {**state, "analysis": "PASS"}
    log.info(f"[Analyzer] task={state['task_id']} iter={state['iteration']} — calling LLM...")
    biased_str  = ", ".join(solar["biased_attrs"])  or "none"
    missing_str = ", ".join(solar["missing_attrs"]) or "none"
    system = get_prompt("analyzer", state["prompt_styles"]["analyzer"])
    user_msg = (
        f"CODE:\n{state['current_code']}\n\n"
        f"BIASED ATTRIBUTES (caused Solar test failures): {biased_str}\n\n"
        f"MISSING TASK-RELEVANT ATTRIBUTES (absent from code): {missing_str}\n\n"
        "Write a repair plan."
    )
    resp = make_llm(temperature=0.2).invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_msg),
    ])
    log.info(f"[Analyzer] task={state['task_id']} — repair plan ready.")
    return {**state, "analysis": resp.content.strip()}


# ── Node 4: Repairer ──────────────────────────────────────────────────────────
def repairer_node(state: AgentState) -> AgentState:
    if state["analysis"] == "PASS":
        return state
    log.info(f"[Repairer] task={state['task_id']} iter={state['iteration']} — calling LLM...")
    system = get_prompt("repairer", state["prompt_styles"]["repairer"])
    user_msg = (
        f"CURRENT CODE:\n{state['current_code']}\n\n"
        f"REPAIR PLAN:\n{state['analysis']}"
    )
    resp = make_llm(temperature=0.2).invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_msg),
    ])
    repaired = resp.content.strip()
    log.info(f"[Repairer] task={state['task_id']} — {len(repaired)} chars, writing for Solar...")
    write_code_for_solar(
        code=repaired,
        task_id=state["task_id"],
        model_dir=state["model_dir"],
        phase="repairer",
        num_samples=state["sampling"],
    )
    entry = {
        "iteration":    state["iteration"],
        "code_before":  state["current_code"],
        "solar_result": state["solar_result"],
        "analysis":     state["analysis"],
        "code_after":   repaired,
    }
    log.info(f"[Repairer] task={state['task_id']} — done.")
    return {
        **state,
        "current_code": repaired,
        "history": list(state.get("history", [])) + [entry],
    }


# ── Node 5: Increment ─────────────────────────────────────────────────────────
def increment_node(state: AgentState) -> AgentState:
    new_iter = state["iteration"] + 1
    log.info(f"[Increment] task={state['task_id']} → iteration {new_iter}")
    return {**state, "iteration": new_iter}


# ── Routing ───────────────────────────────────────────────────────────────────
def route_after_solar(state: AgentState) -> str:
    if state["solar_result"]["passed"]:
        log.info(f"[Router] task={state['task_id']} — PASSED.")
        return "end"
    if state["iteration"] >= state["max_iterations"]:
        log.info(f"[Router] task={state['task_id']} — max iterations reached.")
        return "end"
    log.info(f"[Router] task={state['task_id']} — bias remains, going to Analyzer.")
    return "analyze"


# ── Graph ─────────────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("developer", developer_node)
    g.add_node("solar",     solar_node)
    g.add_node("analyzer",  analyzer_node)
    g.add_node("repairer",  repairer_node)
    g.add_node("increment", increment_node)
    g.set_entry_point("developer")
    g.add_edge("developer",  "solar")
    g.add_conditional_edges(
        "solar",
        route_after_solar,
        {"analyze": "analyzer", "end": END},
    )
    g.add_edge("analyzer",  "repairer")
    g.add_edge("repairer",  "increment")
    g.add_edge("increment", "solar")
    return g.compile()


# ── Run pipeline ──────────────────────────────────────────────────────────────
def run_pipeline(
    prompts_path: str,
    model_dir: str,
    output_dir: str,
    solar_bash: str,
    model_name: str = "gpt",
    sampling: int = 5,
    temperature: float = 1.0,
    prompt_styles: dict = None,
    max_iterations: int = 3,
    test_start: int = 0,
    test_end: int = 5,
) -> list[dict]:

    if prompt_styles is None:
        prompt_styles = {"developer": "agent", "analyzer": "agent", "repairer": "agent"}

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    with open(prompts_path, encoding="utf-8") as f:
        all_tasks = [json.loads(line) for line in f if line.strip()]
    tasks = all_tasks[test_start:test_end]

    if not tasks:
        log.error(f"No tasks found in range [{test_start}, {test_end}) — check prompts_path and test_start/end")
        return []

    log.info(f"Loaded {len(tasks)} tasks | model_dir={model_dir}")
    log.info(f"prompt_styles={prompt_styles} | max_iterations={max_iterations}")

    graph   = build_graph()
    results = []

    for i, task in enumerate(tasks):
        task_id = str(task.get("task_id", i))
        prompt  = task.get("prompt", "")

        log.info(f"")
        log.info(f"{'='*50}")
        log.info(f"TASK {task_id}  ({i+1}/{len(tasks)})")
        log.info(f"{'='*50}")

        initial: AgentState = {
            "task_id":        task_id,
            "prompt":         prompt,
            "model_dir":      model_dir,
            "output_dir":     output_dir,
            "solar_bash":     solar_bash,
            "data_path":      prompts_path,
            "model_name":     model_name,
            "sampling":       sampling,
            "temperature":    temperature,
            "max_iterations": max_iterations,
            "prompt_styles":  prompt_styles,
            "iteration":      0,
            "current_code":   "",
            "solar_result":   {"passed": False, "biased_attrs": [], "missing_attrs": []},
            "analysis":       "",
            "history":        [],
        }

        final = graph.invoke(initial)

        result = {
            "task_id":        task_id,
            "final_code":     final["current_code"],
            "iterations_run": final["iteration"] + 1,
            "final_passed":   final["solar_result"]["passed"],
            "history":        final["history"],
        }
        results.append(result)

        out_file = Path(output_dir) / f"task_{task_id}_result.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log.info(f"Saved → {out_file.name}  passed={result['final_passed']}")

    total  = len(results)
    passed = sum(1 for r in results if r["final_passed"])
    summary = {
        "total":          total,
        "passed":         passed,
        "pass_rate":      round(passed / total, 3) if total else 0,
        "avg_iterations": round(sum(r["iterations_run"] for r in results) / total, 2) if total else 0,
        "test_range":     [test_start, test_end],
        "prompt_styles":  prompt_styles,
    }
    with open(Path(output_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info(f"")
    log.info(f"{'='*50}")
    log.info(f"DONE  passed={passed}/{total}  avg_iters={summary['avg_iterations']}")
    log.info(f"Results → {output_dir}")
    log.info(f"{'='*50}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="FairnessAgent — Developer → Solar → Analyzer → Repairer → Solar"
    )
    p.add_argument("--prompts_path",    required=True)
    p.add_argument("--model_dir",       required=True)
    p.add_argument("--output_dir",      required=True)
    p.add_argument("--solar_bash",      required=True)
    p.add_argument("--model_name",      default="gpt")
    p.add_argument("--sampling",        type=int,   default=5)
    p.add_argument("--temperature",     type=float, default=1.0)
    p.add_argument("--max_iterations",  type=int,   default=3)
    p.add_argument("--test_start",      type=int,   default=0)
    p.add_argument("--test_end",        type=int,   default=5)
    p.add_argument("--developer_style", default="agent",
                   choices=["agent","default","chain_of_thoughts","positive_chain_of_thoughts"])
    p.add_argument("--analyzer_style",  default="agent",
                   choices=["agent","structured"])
    p.add_argument("--repairer_style",  default="agent",
                   choices=["agent","minimal"])
    args = p.parse_args()

    run_pipeline(
        prompts_path=args.prompts_path,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        solar_bash=args.solar_bash,
        model_name=args.model_name,
        sampling=args.sampling,
        temperature=args.temperature,
        prompt_styles={
            "developer": args.developer_style,
            "analyzer":  args.analyzer_style,
            "repairer":  args.repairer_style,
        },
        max_iterations=args.max_iterations,
        test_start=args.test_start,
        test_end=args.test_end,
    )
