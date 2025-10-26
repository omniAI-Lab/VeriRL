import os
import json
from openai import OpenAI
from tqdm import tqdm



JSONL_FILE = "./data/PyraNet.jsonl"
PROMPT_FILE = "./prompt/generate_testbench_prompt.txt"
OUTPUT_FILE = "output.jsonl"


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  

MODEL_NAME = "gpt-4o-mini"


with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    base_prompt = f.read().strip()


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line.strip())


def generate_question(description, code):
    problem = f"{description}\n\n{code}"
    prompt = base_prompt.replace("{problem}", problem)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a professional Verilog tutor."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    results = []

    data_iter = list(load_jsonl(JSONL_FILE))
    print(f"Loaded {len(data_iter)} samples from {JSONL_FILE}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for item in tqdm(data_iter, desc="Generating HDLBits Questions"):
            desc = item.get("description", "")
            code = item.get("code", "")

            try:
                output_text = generate_question(desc, code)
                result = {
                    "description": desc,
                    "code": code,
                    "output": output_text,
                }
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error on item: {e}")
                continue

    print(f"Done! Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
