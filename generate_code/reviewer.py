# generate_code/reviewer.py
import json
import os
import re
import sys
import time
from itertools import islice
from openai import OpenAI
from dotenv import load_dotenv
from google.cloud import datastore
from vertexai.language_models import CodeChatModel
import anthropic

load_dotenv()

# --- Clients: keep same style as developer.py ---
gpt_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

llama_client = OpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)

# Vertex / Google
# google_client = datastore.Client()

anthropic_client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)


# --- Helpers ---
def read_jsonl_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def process_claude_response(response: str) -> str:
    """
    Reuse same trimming logic as developer.py, but for reviewer we want text-only.
    We'll still keep a small sanitizer: strip code fences if any appear.
    """
    if response is None:
        return ""
    if "```" in response:
        # keep only text before fenced code or remove fences
        # simplest: strip fences entirely
        response = response.replace("```python", "").replace("```", "")
    return response.strip()

# --- LLM call (same multi-model switch style as developer.py) ---
def review_conversation(style, qs, temp, model_name) -> str:
    if model_name == "gpt":
        response = gpt_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=temp,
            messages=[
                {"role": "system", "content": style},
                {"role": "user", "content": qs},
            ],
        )
        txt = response.choices[0].message.content
        return (txt or "").strip()

    elif model_name == "llama":
        response = llama_client.chat.completions.create(
            model="CODELLAMA/CODELLAMA-70B-INSTRUCT-HF",
            temperature=temp,
            messages=[
                {"role": "system", "content": style},
                {"role": "user", "content": qs},
            ],
        )
        txt = response.choices[0].message.content
        return process_claude_response(txt)

    elif model_name == "bison":
        parameters = {
            "temperature": temp,
            "max_output_tokens": 256,
        }
        code_chat_model = CodeChatModel.from_pretrained("codechat-bison@002")
        chat = code_chat_model.start_chat(context=style)
        response = chat.send_message(qs, **parameters)
        txt = response.text
        return (txt or "").strip()

    elif model_name == "claude":
        # keep same pause pattern as developer.py to avoid rate limits
        time.sleep(20)
        response = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=256,
            temperature=temp,
            system=style,
            messages=[{"role": "user", "content": [{"type": "text", "text": qs}]}],
        )
        txt = response.content[0].text
        return process_claude_response(txt)

    else:
        raise ValueError("Invalid model name. Choose between 'gpt', 'llama', 'bison', 'claude'.")


# --- Prompt styles (same dict pattern as developer.py) ---
prompt_styles = {
    "gpt": {
        # "agent": '''You are a fairness-aware code reviewer. 
        # Given PROMPT, GENERATED_CODE (one method), and SENSITIVE_ATTRIBUTES (comma-separated), output ONLY JSON: {"edits":[{"op":"replace","old":"...","new":"..."},{"op":"delete","old":"..."}]}. 
        # "old" MUST be an exact substring of GENERATED_CODE. 
        # Make minimal edits to remove all dependence on SENSITIVE_ATTRIBUTES. ensure the functional correctness, No code, no extra text.'''
        'agent': '''You are a fairness-aware code reviewer.
Given PROMPT, GENERATED_CODE (one method), and SENSITIVE_ATTRIBUTES (comma-separated), output ONLY JSON edits {"edits":[{"op":"replace","old":"...","new":"..."},{"op":"delete","old":"..."}]}. "old" MUST be an exact substring of GENERATED_CODE.
Goal: remove all dependence on SENSITIVE_ATTRIBUTES by deleting their boolean predicates (and adjacent and/or) while keeping other logic unchanged.
Do NOT change numeric thresholds/sets, do NOT add new predicates/attributes/functions, and do NOT delete control-structure lines (if/elif/else/return); only edit condition substrings.
No code, no extra text.'''
    }
}

def generate_reviews_from_bias_info(
    prompts_file_path: str,
    src_gc_dir: str,
    target_review_dir: str,
    bias_info_base_path: str,
    iterations: int,
    temperature: float,
    style: str,
    model_name: str,
    test_start: int,
    test_end: int
):
    """
    For each task in prompts_file_path:
      - read generated code from src_gc_dir/task_<id>_generated_code.jsonl
      - read bias info from bias_info_base_path/task_<id>_bias_info.jsonl (or direct file)
      - write review lines to src_gc_dir/task_<id>_review.jsonl

    Each line corresponds to an iteration index.
    """
    for index, json_obj in enumerate(islice(read_jsonl_file(prompts_file_path), test_start, test_end), start=test_start):
        print(f"Processing line {index}")
        print("-"*50)
        task_id = str(json_obj.get("task_id", "default"))
        prompt = json_obj.get("prompt", "")

        if not prompt:
            continue

        gen_code_path = os.path.join(src_gc_dir, f"task_{task_id}_generated_code.jsonl")
        if not os.path.exists(gen_code_path):
            print(f"[WARN] Missing generated code file: {gen_code_path}")
            continue

        # bias_file_path = resolve_bias_file(task_id, bias_info_base_path)
        bias_file_path = os.path.join(bias_info_base_path, f"bias_info{task_id}.jsonl")
        if not os.path.exists(bias_file_path):
            print(f"[WARN] Missing bias info file: {bias_file_path}")
            continue

        # Read code + bias lines
        code_lines = list(read_jsonl_file(gen_code_path))
        bias_lines = list(read_jsonl_file(bias_file_path))

        # Output review file (same dir, different file name)
        review_path = os.path.join(target_review_dir, f"task_{task_id}_review.jsonl")
        os.makedirs(os.path.dirname(review_path), exist_ok=True)

        with open(review_path, "w", encoding="utf-8") as out_f:
            # align by iteration index
            n = min(iterations, len(code_lines), len(bias_lines))
            for i in range(n):
                code_obj = code_lines[i]
                bias_obj = bias_lines[i]

                if bias_obj["bias_info"] == "none":
                    json.dump({"review": "pass"}, out_f, ensure_ascii=False)
                    out_f.write("\n")
                    continue

                # Build concise user query
                qs = (
                    "PROMPT:\n"
                    f"{prompt}\n\n"
                    "GENERATED_CODE:\n"
                    f"{code_obj.get('generated_code','')}\n\n"
                    "SENSITIVE_ATTRIBUTES:\n"
                    f"{bias_obj['bias_info']}\n"
                )

                instruction = review_conversation(style, qs, temperature, model_name)
                json.dump({"review": instruction}, out_f, ensure_ascii=False)
                out_f.write("\n")

        print(f"[OK] Wrote reviews: {review_path}")


if __name__ == "__main__":
    """
    Mirrors developer.py arg style, with ONE EXTRA arg at the end:

    python reviewer.py <prompts_jsonl> <src_gc_dir> <num_samples> [temperature] [prompt_style] <model_name> <bias_info_base_path>

    - prompts_jsonl: same input you used for developer.py
    - src_gc_dir: same output dir where task_<id>_generated_code.jsonl exists
    - num_samples: iterations (same as developer)
    - bias_info_base_path: directory or file path for bias info
    """
    print("starting reviewer agent ...")
    print("=" * 50)
    prompts_jsonl_path = sys.argv[1]
    src_gc_base_dir = sys.argv[2]
    target_review_base_dir = sys.argv[3]
    num_samples = int(sys.argv[4])

    TEMPERATURE = float(sys.argv[5])
    PROMPT_STYLE = sys.argv[6]
    MODEL_NAME = sys.argv[7]
    BIAS_INFO_BASE_PATH = sys.argv[8]

    TEST_START = sys.argv[9]
    TEST_END = sys.argv[10]

    print("prompts_jsonl_path", prompts_jsonl_path)
    print("src_gc_base_dir", src_gc_base_dir)
    print("target_review_base_dir", target_review_base_dir)
    print("num_samples", num_samples)
    print("TEMPERATURE", TEMPERATURE)
    print("PROMPT_STYLE", PROMPT_STYLE)
    print("MODEL_NAME", MODEL_NAME)
    print("BIAS_INFO_BASE_PATH", BIAS_INFO_BASE_PATH)
    print("TEST_START", TEST_START)
    print("TEST_END", TEST_END)
    

    os.makedirs(src_gc_base_dir, exist_ok=True)

    generate_reviews_from_bias_info(
        prompts_file_path=prompts_jsonl_path,
        src_gc_dir=src_gc_base_dir,
        target_review_dir=target_review_base_dir,
        bias_info_base_path=BIAS_INFO_BASE_PATH,
        iterations=num_samples,
        temperature=TEMPERATURE,
        style=prompt_styles[MODEL_NAME][PROMPT_STYLE],
        model_name=MODEL_NAME,
        test_start=int(TEST_START),
        test_end=int(TEST_END)
    )
