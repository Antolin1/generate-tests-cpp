import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True,
                                             torch_dtype=torch.bfloat16).cuda()
messages = [
    {'role': 'user', 'content': "Write four google tests for the that this function passes\n{func}"}
]
func = """bool has_close_elements(vector<float> numbers, float threshold){
    int i,j;
    for (i=0;i<numbers.size();i++)
        for (j=i+1;j<numbers.size();j++)
            if (abs(numbers[i]-numbers[j])<threshold)
                return true;
    return false;
}"""

messages[0]['content'] = messages[0]['content'].format(func=func)

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
# tokenizer.eos_token_id is the id of <|EOT|> token
outputs = model.generate(inputs, max_new_tokens=512, do_sample=True, temperature=0.2, num_return_sequences=1,
                         eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
