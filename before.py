from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("microsoft/Orca-2-7b").to("cuda:0")

prompt = "How are you?"
tokenizer = AutoTokenizer.from_pretrained("microsoft/Orca-2-7b")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

# Generate
generate_ids = model.generate(**inputs, do_sample=True, num_beams=1, max_new_tokens=100)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])