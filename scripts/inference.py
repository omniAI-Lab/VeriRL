import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "VeriRL/VeriRL-CodeQwen2.5"


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True
)
model.eval()


question_prompt = (
    "You are an expert in digital logic design and Verilog programming. "
    "Given a Verilog problem statement, your task is to generate a structured explanation "
    "of the approach and a correct Verilog implementation. "
    "Carefully analyze the problem, explain the logic behind the solution, "
    "and provide a well-formatted Verilog implementation. "
    "Ensure the output follows this format: wrap the reasoning inside <REASON> and </REASON>, "
    "and the Verilog solution inside <SOLUTION> and </SOLUTION>.\n\n"
)


task_description = (
    "Design a 4-bit binary counter with synchronous reset and enable signals. "
    "The counter should increment on each rising edge of the clock when enable is high, "
    "and reset to 0 when reset is asserted."
)

full_prompt = question_prompt + task_description


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": full_prompt}
]

chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def parse_response(text):
    reasoning, solution = "", ""
    if "<REASON>" in text and "</REASON>" in text:
        reasoning = text.split("<REASON>")[1].split("</REASON>")[0].strip()
    if "<SOLUTION>" in text and "</SOLUTION>" in text:
        solution = text.split("<SOLUTION>")[1].split("</SOLUTION>")[0].strip()
    return reasoning, solution

reasoning, solution = parse_response(response_text)

print("\n==================== MODEL OUTPUT ====================")
print(response_text)
print("======================================================\n")


