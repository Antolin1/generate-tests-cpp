import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from templates import WIZARD_TEMPLATE_HUMANEVAL, postprocess_markdown, execute_code

set_seed(123)

heval = load_dataset("openai_humaneval")["test"]

prompt = WIZARD_TEMPLATE_HUMANEVAL.format(question=heval[0]["prompt"])

tokenizer = AutoTokenizer.from_pretrained("/data/important_models/wizardcoder-15b-v1")
model = AutoModelForCausalLM.from_pretrained("/data/important_models/wizardcoder-15b-v1", device_map="auto",
                                             torch_dtype=torch.float16)

sample = tokenizer([prompt], return_tensors="pt")
with torch.no_grad():
    generated_sequences = model.generate(
        input_ids=sample["input_ids"].cuda(),
        attention_mask=sample["attention_mask"].cuda(),
        do_sample=True,
        max_new_tokens=1024,
        num_return_sequences=5,
        temperature=0.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_sequences = generated_sequences.cpu().numpy()
    generated_new_tokens = generated_sequences[:, sample["input_ids"].shape[1]:]

for k, new_tokens in enumerate(generated_new_tokens):
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    python_code = postprocess_markdown(generated)
    print(python_code)
    print(execute_code(python_code, heval[0]["test"], heval[0]["entry_point"]))
    print('-' * 100)
