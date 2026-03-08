import argparse
import json
import os
import time
from itertools import islice

from openai import OpenAI
from dotenv import load_dotenv
from vertexai.language_models import CodeChatModel
import anthropic

load_dotenv()

gpt_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

llama_client = OpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)

anthropic_client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)


def read_jsonl_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def process_text_response(response: str) -> str:
    if response is None:
        return ""
    response = response.strip()
    if "```json" in response:
        response = response.replace("```json", "").replace("```", "").strip()
    elif "```" in response:
        response = response.replace("```", "").strip()
    return response


def extract_json_object(text: str):
    text = process_text_response(text)
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

    raise ValueError(f"Could not parse JSON from response: {text}")


def analyst_conversation(style, qs, temp, model_name) -> str:
    if model_name == "gpt":
        response = gpt_client.chat.completions.create(
            # model="gpt-3.5-turbo",
            model="gpt-5.4-pro",
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
        return process_text_response(response.choices[0].message.content or "")

    elif model_name == "bison":
        parameters = {
            "temperature": temp,
            "max_output_tokens": 256,
        }
        code_chat_model = CodeChatModel.from_pretrained("codechat-bison@002")
        chat = code_chat_model.start_chat(context=style)
        response = chat.send_message(qs, **parameters)
        return (response.text or "").strip()

    elif model_name == "claude":
        time.sleep(20)
        response = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=256,
            temperature=temp,
            system=style,
            messages=[{"role": "user", "content": [{"type": "text", "text": qs}]}],
        )
        return process_text_response(response.content[0].text or "")

    else:
        raise ValueError("Invalid model name. Choose between 'gpt', 'llama', 'bison', 'claude'.")


prompt_styles = {
    "gpt": {
        "agent": """You are a requirements analyst for a Python code generation task.

You will receive a Python prompt containing:
- a dataclass with attributes
- a target method
- a docstring describing a decision task

Your task is to determine which dataclass attributes are relevant for implementing the decision logic.

Rules:
1. Use only attributes defined in the dataclass.
2. Base your reasoning only on the information present in the prompt (method name, docstring, attribute names, or comments).
3. Do not assume external policies or real-world knowledge.
4. Select attributes that are clearly relevant to making the decision described in the prompt.
5. Do not prioritize demographic attributes unless the task description explicitly requires them.
6. Select the smallest set of clearly relevant attributes.
7. At least one attribute must be returned.

Output ONLY valid JSON in this format:
{
  "required_attributes": {
    "<attribute_name>": "<brief reason>"
  }
}"""
    }
}


def normalize_requirements(obj):
    if not isinstance(obj, dict):
        raise ValueError("Requirements output must be a JSON object.")

    if "required_attributes" in obj:
        required = obj["required_attributes"]
        if not isinstance(required, dict) or not required:
            raise ValueError("'required_attributes' must be a non-empty object.")
        return {"required_attributes": required}

    if obj:
        # backward-compatible fallback if model returns {"income": "...", ...}
        return {"required_attributes": obj}

    raise ValueError("No required attributes found in output.")


def generate_requirements_from_prompts(
    prompts_file_path,
    output_dir,
    temperature,
    style,
    model_name,
    test_start,
    test_end,
):
    for index, json_obj in enumerate(
        islice(read_jsonl_file(prompts_file_path), test_start, test_end),
        start=test_start,
    ):
        print(f"Processing line {index}")
        print("-" * 50)

        task_id = str(json_obj.get("task_id", "default"))
        prompt = json_obj.get("prompt", "")

        if not prompt:
            continue

        qs = f"PROMPT:\n{prompt}\n"
        raw_response = analyst_conversation(style, qs, temperature, model_name)

        try:
            parsed = extract_json_object(raw_response)
            normalized = normalize_requirements(parsed)
        except Exception as e:
            print(f"[WARN] Failed to parse requirement JSON for task {task_id}: {e}")
            normalized = {
                "required_attributes": {
                    "unknown": "fallback because JSON parsing failed"
                }
            }

        out_path = os.path.join(output_dir, f"task_{task_id}_requirements.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(normalized, out_f, ensure_ascii=False, indent=2)

        print(f"[OK] Wrote requirements: {out_path}")
        print(json.dumps(normalized, ensure_ascii=False))
        print("-" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Requirements analyst agent")
    parser.add_argument("--prompts_jsonl_path", required=True)
    parser.add_argument("--output_base_dir", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--prompt_style", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--test_start", type=int, required=True)
    parser.add_argument("--test_end", type=int, required=True)

    args = parser.parse_args()

    print("starting requirements analyst agent")
    print("=" * 50)
    print("prompts_jsonl_path", args.prompts_jsonl_path)
    print("output_base_dir", args.output_base_dir)
    print("TEMPERATURE", args.temperature)
    print("PROMPT_STYLE", args.prompt_style)
    print("MODEL_NAME", args.model_name)
    print("TEST_START", args.test_start)
    print("TEST_END", args.test_end)

    os.makedirs(args.output_base_dir, exist_ok=True)

    generate_requirements_from_prompts(
        prompts_file_path=args.prompts_jsonl_path,
        output_dir=args.output_base_dir,
        temperature=float(args.temperature),
        style=prompt_styles[args.model_name][args.prompt_style],
        model_name=args.model_name,
        test_start=int(args.test_start),
        test_end=int(args.test_end),
    )
