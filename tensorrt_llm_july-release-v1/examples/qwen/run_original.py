

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("./Qwen-7B", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("./Qwen-7B", device_map="auto", fp16=True, trust_remote_code=True).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained("./Qwen-7B", trust_remote_code=True)

# inputs = tokenizer('蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是', return_tensors='pt')
inputs = tokenizer("Born in north-east France, Soyer trained as a", return_tensors='pt')
inputs = inputs.to(model.device)
# to align with trt version 
pred = model.generate(**inputs, top_k=1, num_beams=1, temperature=1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
