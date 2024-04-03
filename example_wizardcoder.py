import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


problem_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nWrite four google tests that this function passes:\n{func}\n\n### Response:\nTEST({func_name},"
        )
prompt = problem_prompt.format(func=func,
                               func_name=func_name)

tokenizer = AutoTokenizer.from_pretrained("WizardLM/WizardCoder-3B-V1.0")
model = AutoModelForCausalLM.from_pretrained("WizardLM/WizardCoder-3B-V1.0", device_map="auto",
                                             torch_dtype=torch.float16)

sample = tokenizer([prompt], return_tensors="pt")
with torch.no_grad():
    generated_sequences = model.generate(
        input_ids=sample["input_ids"].cuda(),
        attention_mask=sample["attention_mask"].cuda(),
        do_sample=True,
        max_new_tokens=1024,
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
