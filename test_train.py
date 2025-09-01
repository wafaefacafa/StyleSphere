from transformers import LlamaTokenizer
from datasets import Dataset
import yaml
import os
import json
from llama_cpp import Llama
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')

# 设置环境
os.environ['PYTHONWARNINGS'] = 'ignore'

def create_and_prepare_model(config):
    # 使用llama.cpp加载GGUF模型
    model_path = os.path.join(config['model_name_or_path'], "llama-2-7b-chat.Q4_K_S.gguf")
    print(f"加载GGUF模型: {model_path}")
    
    n_gpu_layers = 0 if config.get('use_cpu', False) else -1  # -1表示使用所有可用的GPU层
    model = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=2048,  # 上下文窗口大小
        n_batch=config.get('per_device_train_batch_size', 1)
    )
    
    # llama.cpp自带tokenizer
    tokenizer = model
    
    return model, tokenizer

def prepare_dataset(tokenizer, config, data_path):
    # 加载数据集
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 预处理数据
    processed_data = []
    for item in dataset:
        # 构建提示词训练格式
        prompt_template = """[INST] <<SYS>>
你是一个专业的AI助手，尝试用专业、准确的语言回答用户的问题。
<</SYS>>

{instruction}

{input} [/INST]

{output}

</s>"""
        # 填充模板
        prompt = prompt_template.format(
            instruction=item['instruction'],
            input=item['input'],
            output=item['output']
        )
        processed_data.append(prompt)
    
    return processed_data

def train(config_path="e:/AI/llama2-train/test_config.yaml", data_path="e:/AI/llama2-train/data/test_train.json"):
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建并准备模型
    print("正在加载模型...")
    model, tokenizer = create_and_prepare_model(config)
    
    # 准备数据集
    print("正在准备数据集...")
    data = prepare_dataset(tokenizer, config, data_path)
    
    print("开始训练...")
    
    # 使用llama.cpp进行训练
    for epoch in range(config['max_steps']):
        for i, text in enumerate(data):
            # 使用模型生成
            # 使用模型生成
            response = model.create_completion(
                text,
                max_tokens=config['max_target_length'],
                temperature=0.7,
                top_p=0.95,
                repeat_penalty=1.1,
                echo=False,
                stream=False
            )
            
            # 获取生成的文本
            if isinstance(response, dict) and 'choices' in response and len(response['choices']) > 0:
                generated_text = response['choices'][0]['text']
            else:
                generated_text = "无生成结果"
            
            print(f"Epoch {epoch+1}, Step {i+1}/{len(data)}")
            print(f"Input text: {text}")
            print(f"Generated: {generated_text}\n")
    
    print("训练完成")

if __name__ == "__main__":
    train()
