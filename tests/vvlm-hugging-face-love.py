from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import torch
import subprocess
import gc
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def log_gpu_usage():
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"GPU Memory Allocated: {allocated_memory:.4f} GB")
    print(f"GPU Memory Reserved: {reserved_memory:.4f} GB")

def move_model_to_cpu(model):
    for param in model.parameters():
        param.data = param.data.to('cpu')
        if param.grad is not None:
            param.grad.data = param.grad.data.to('cpu')
    model.to('cpu')

torch.set_default_device('cuda')

# Section 1: Load and configure model with LoRA and PPO
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.cuda()

input_text = "What is the capital of France?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()

lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0, bias="none")
model.add_adapter(lora_config, adapter_name="adapter_1")
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

ppo_config = PPOConfig()
ppo_trainer = PPOTrainer(model=model, tokenizer=tokenizer, config=ppo_config)

lora_weights_path = "./lora_weights"
model.save_pretrained(lora_weights_path)

print("For HF:")
log_gpu_usage()
move_model_to_cpu(model)
del model
del ppo_trainer
gc.collect()
torch.cuda.empty_cache()
log_gpu_usage()

# Section 2: Load and run inference with vllm
print("For LLM:")
llm = LLM(model_name, enable_lora=True)
sampling_params = SamplingParams(temperature=0.7)

with torch.no_grad():
    output = llm.generate([input_text], sampling_params=sampling_params, lora_request=LoRARequest("dummy_lora", 1, lora_weights_path))

print(f"Generated text: {output}")

log_gpu_usage()
del llm
gc.collect()
torch.cuda.empty_cache()
log_gpu_usage()

print("DONE")
