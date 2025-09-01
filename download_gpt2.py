import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import ssl

# 禁用SSL验证
ssl._create_default_https_context = ssl._create_unverified_context

print("下载GPT-2模型...")
model_name = "gpt2"
cache_dir = os.path.join(os.getcwd(), "models", "gpt2")
os.makedirs(cache_dir, exist_ok=True)

# 下载tokenizer
print("下载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
tokenizer.save_pretrained(cache_dir)

# 下载模型
print("下载模型...")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
model.save_pretrained(cache_dir)

print(f"模型和tokenizer已保存到: {cache_dir}")
