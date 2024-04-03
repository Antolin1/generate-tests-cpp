import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# dataset = load_dataset("THUDM/humaneval-x", "cpp")["test"]
# print(dataset[0]['canonical_solution'])


func = """bool has_close_elements(vector<float> numbers, float threshold){
    int i,j;
    for (i=0;i<numbers.size();i++)
        for (j=i+1;j<numbers.size();j++)
            if (abs(numbers[i]-numbers[j])<threshold)
                return true;
    return false;
}"""
func_name = "has_close_elements"
doc = "Check if in given vector of numbers, are any two numbers closer to each other than given threshold."

print(func)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", device_map="auto",
                                             torch_dtype=torch.float16)

PROMPT_TEMPLATE = """
{func}
// tests
TEST({func_name},"""

PROMPT_TEMPLATE_DOC = """
// {doc}
{func}
// tests
TEST({func_name},"""

prompt = PROMPT_TEMPLATE_DOC.format(func=func,
                                    func_name=func_name,
                                    doc=doc)
sample = tokenizer([prompt], return_tensors="pt")

with torch.no_grad():
    generated_sequences = model.generate(
        input_ids=sample["input_ids"].cuda(),
        attention_mask=sample["attention_mask"].cuda(),
        do_sample=True,
        max_new_tokens=512,
        num_return_sequences=1,
        temperature=0.4,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_sequences = generated_sequences.cpu().numpy()
    generated_new_tokens = generated_sequences[:, sample["input_ids"].shape[1]:]

for k, new_tokens in enumerate(generated_new_tokens):
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(prompt + generated)
    print('-' * 100)
