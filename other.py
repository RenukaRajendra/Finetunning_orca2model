from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/Orca-2-7b")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Orca-2-7b", device_map='cuda')

prompt = "How are you?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])