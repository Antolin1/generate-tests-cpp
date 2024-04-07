import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from example_function import FUNC, FUNC_NAME, DOCUMENTATION
from templates import DS_TEMPLATE_DOC

set_seed(123)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", device_map="auto",
                                             torch_dtype=torch.float16)

prompt = DS_TEMPLATE_DOC.format(func=FUNC,
                                func_name=FUNC_NAME,
                                doc=DOCUMENTATION,
                                num=5)

sample = tokenizer([prompt], return_tensors="pt")

with torch.no_grad():
    generated_sequences = model.generate(
        input_ids=sample["input_ids"].cuda(),
        attention_mask=sample["attention_mask"].cuda(),
        do_sample=True,
        max_new_tokens=1024,
        num_return_sequences=1,
        temperature=0.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_sequences = generated_sequences.cpu().numpy()
    generated_new_tokens = generated_sequences[:, sample["input_ids"].shape[1]:]

for k, new_tokens in enumerate(generated_new_tokens):
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(prompt + generated)
    print('-' * 100)
