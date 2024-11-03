from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

config = PeftConfig.from_pretrained("praison/orca-2-7B-v01-fine-tuned-using-ludwig-4bit")
model = AutoModelForCausalLM.from_pretrained("microsoft/Orca-2-7b").to("cuda:0")
model = PeftModel.from_pretrained(model, "praison/orca-2-7B-v01-fine-tuned-using-ludwig-4bit")

prompt = "How are you?"
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("microsoft/Orca-2-7b")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

# Generate
generate_ids = model.generate(**inputs, do_sample=True, num_beams=1, max_new_tokens=100)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
