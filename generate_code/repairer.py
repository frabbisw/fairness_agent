# generate_code/repair.py
import json
import os
import sys
import time

from openai import OpenAI
from dotenv import load_dotenv
from google.cloud import datastore
from vertexai.language_models import CodeChatModel
import anthropic

load_dotenv()

# --- Clients (same style as developer.py / reviewer.py) ---
gpt_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

llama_client = OpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)

google_client = datastore.Client()

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

def process_model_text(response: str) -> str:
    """Keep only plain text, strip accidental fences."""
    if response is None:
        return ""
    if "```" in response:
        response = response.replace("```python", "").replace("```", "")
    return response.strip()

# --- LLM call switch (same pattern) ---
def repair_conversation(style, qs, temp, model_name) -> str:
    if model_name == "gpt":
        response = gpt_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=temp,
            messages=[
                {"role": "system", "content": style},
                {"role": "user", "content": qs},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    elif model_name == "llama":
        response = llama_client.chat.completions.create(
            model="CODELLAMA/CODELLAMA-70B-INSTRUCT-HF",
            temperature=temp,
            messages=[
                {"role": "system", "content": style},
                {"role": "user", "content": qs},
            ],
        )
        return process_model_text(response.choices[0].message.content)

    elif model_name == "bison":
        parameters = {"temperature": temp, "max_output_tokens": 1024}
        code_chat_model = CodeChatModel.from_pretrained("codechat-bison@002")
        chat = code_chat_model.start_chat(context=style)
        response = chat.send_message(qs, **parameters)
        return (response.text or "").strip()

    elif model_name == "claude":
        time.sleep(20)  # same anti-rate-limit flavor
        response = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            temperature=temp,
            system=style,
            messages=[{"role": "user", "content": [{"type": "text", "text": qs}]}],
        )
        return process_model_text(response.content[0].text)

    else:
        raise ValueError("Invalid model name. Choose between 'gpt', 'llama', 'bison', 'claude'.")


# --- Prompt styles (same dict style) ---
prompt_styles = {
    "gpt": {
        "agent_repair": (
            "You are a code developer fixing code based on review feedback. "
            "Return ONLY the repaired method code (signature + body). "
            "Do NOT output the class, markdown, or explanations. "
            "Do NOT change the method signature (including self). "
            "Apply only the requested fixes and keep behavior consistent."
        )
    },
    "llama": {
        "agent_repair": (
            "Fix the method based on the review. Output ONLY the method (signature + body). "
            "No class. No explanations. Do not change signature."
        )
    },
    "bison": {
        "agent_repair": (
            "Repair the given method using the review feedback. Output only the method code. "
            "Do not change the signature. No class."
        )
    },
    "claude": {
        "agent_repair": (
            "You are repairing a method based on review feedback. "
            "Output ONLY the full repaired method (signature + body), no class, no markdown, no extra text. "
            "Do not change the method signature (including self)."
        )
    },
}


def resolve_review_file(task_id: str, review_base_path: str) -> str:
    """
    review_base_path can be:
      - directory containing task_<id>_review.jsonl
      - a single file path (for one task)
    """
    if os.path.isdir(review_base_path):
        return os.path.join(review_base_path, f"task_{task_id}_review.jsonl")
    return review_base_path


def repair_tasks(
    prompts_file_path: str,
    output_dir: str,
    review_base_path: str,
    iterations: int,
    temperature: float,
    style: str,
    model_name: str,
):
    """
    For each task in prompts_file_path:
      - read generated method from output_dir/task_<id>_generated_code.jsonl
      - read review from review_base_path/task_<id>_review.jsonl
      - if review == 'pass': keep original method
      - else: LLM repairs method using (class/prompt context + original method + review)
      - write to output_dir/task_<id>_repaired_code.jsonl line-by-line
    """
    for obj in read_jsonl_file(prompts_file_path):
        task_id = str(obj.get("task_id", "default"))
        prompt = obj.get("prompt", "")

        if not prompt:
            continue

        gen_path = os.path.join(output_dir, f"task_{task_id}_generated_code.jsonl")
        if not os.path.exists(gen_path):
            print(f"[WARN] Missing generated code: {gen_path}")
            continue

        review_path = resolve_review_file(task_id, review_base_path)
        if not os.path.exists(review_path):
            print(f"[WARN] Missing review file: {review_path}")
            continue

        gen_lines = list(read_jsonl_file(gen_path))
        rev_lines = list(read_jsonl_file(review_path))

        out_path = os.path.join(output_dir, f"task_{task_id}_repaired_code.jsonl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        n = min(iterations, len(gen_lines), len(rev_lines))
        if n == 0:
            print(f"[WARN] No aligned lines for task_{task_id}")
            continue

        with open(out_path, "w", encoding="utf-8") as out_f:
            for i in range(n):
                gen_obj = gen_lines[i] if isinstance(gen_lines[i], dict) else {"generated_code": str(gen_lines[i])}
                rev_obj = rev_lines[i] if isinstance(rev_lines[i], dict) else {"review": str(rev_lines[i])}

                method_code = gen_obj.get("generated_code", "")
                review_text = (rev_obj.get("review", "") or "").strip()

                # If pass, keep as-is
                if review_text.lower() == "pass":
                    json.dump({"generated_code": method_code}, out_f, ensure_ascii=False)
                    out_f.write("\n")
                    continue

                # Build user query:
                # - prompt contains the class + method stub in your dataset
                # - include current generated method and reviewer instruction
                qs = (
                    "CONTEXT (class + stub):\n"
                    f"{prompt}\n\n"
                    "CURRENT METHOD:\n"
                    f"{method_code}\n\n"
                    "REVIEW FEEDBACK:\n"
                    f"{review_text}\n\n"
                    "Return ONLY the repaired method (signature + body)."
                )

                repaired = repair_conversation(style, qs, temperature, model_name)

                json.dump({"generated_code": repaired}, out_f, ensure_ascii=False)
                out_f.write("\n")

        print(f"[OK] Wrote repaired code: {out_path}")


if __name__ == "__main__":
    """
    Similar arg style to developer/reviewer, with one extra input for review_base_path.

    python repair.py <prompts_jsonl> <output_dir> <num_samples> [temperature] [prompt_style] <model_name> <review_base_path>

    - prompts_jsonl: dataset prompts.jsonl
    - output_dir: where task_<id>_generated_code.jsonl exists
    - review_base_path: directory or single file for task_<id>_review.jsonl
    """
    prompts_jsonl_path = sys.argv[1]
    output_base_dir = sys.argv[2]
    num_samples = int(sys.argv[3])

    TEMPERATURE = 1.0 if len(sys.argv) < 5 else float(sys.argv[4])
    PROMPT_STYLE = "agent_repair" if len(sys.argv) < 6 else sys.argv[5]
    MODEL_NAME = sys.argv[6]
    REVIEW_BASE_PATH = sys.argv[7]

    print("prompts_jsonl_path", prompts_jsonl_path)
    print("output_base_dir", output_base_dir)
    print("num_samples", num_samples)
    print("TEMPERATURE", TEMPERATURE)
    print("PROMPT_STYLE", PROMPT_STYLE)
    print("MODEL_NAME", MODEL_NAME)
    print("REVIEW_BASE_PATH", REVIEW_BASE_PATH)

    os.makedirs(output_base_dir, exist_ok=True)

    repair_tasks(
        prompts_file_path=prompts_jsonl_path,
        output_dir=output_base_dir,
        review_base_path=REVIEW_BASE_PATH,
        iterations=num_samples,
        temperature=TEMPERATURE,
        style=prompt_styles[MODEL_NAME][PROMPT_STYLE],
        model_name=MODEL_NAME,
    )
