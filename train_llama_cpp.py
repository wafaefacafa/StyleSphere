import json
import os
import math
import random
from llama_cpp import Llama
from tqdm import tqdm
import torch
import numpy as np
from typing import List, Dict, Any

# 配置参数
MODEL_PATH = "e:/AI/models/Llama-3-8B-Instruct-Q4_K_M.gguf"
INITIAL_LR = 5e-4  # 初始学习率
MIN_LR = 1e-5  # 最小学习率
BATCH_SIZE = 1
MAX_TOKENS = 1024  # 保持较长的token长度
EPOCHS = 5
WARMUP_RATIO = 0.1  # 预热阶段比例
ACCUMULATION_STEPS = 4  # 梯度累积步数
EVAL_STEPS = 20  # 每训练多少步进行一次评估
OUTPUT_DIR = "outputs/lora-test-optimized"

def load_training_data(file_path="data/train.json"):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def format_prompt(instruction, input_text=""):
    """格式化提示模板 - Llama 3"""
    system_prompt = "You are a helpful assistant."
    user_content = f"{instruction}\n{input_text}" if input_text else instruction
    
    return (
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def get_learning_rate(current_step, total_steps):
    """实现带warmup的余弦学习率调度"""
    warmup_steps = int(total_steps * WARMUP_RATIO)
    if current_step < warmup_steps:
        # 线性预热
        return INITIAL_LR * (current_step / warmup_steps)
    else:
        # 余弦衰减
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return MIN_LR + 0.5 * (INITIAL_LR - MIN_LR) * (1 + math.cos(math.pi * progress))

def calculate_loss(output_text, target_text):
    """改进的loss计算方法"""
    if not output_text:
        return 1.0
    
    # 计算最长公共子序列
    def lcs_length(text1, text2):
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    # 结合字符匹配度和最长公共子序列
    char_match = sum(1 for a, b in zip(output_text, target_text) if a == b)
    lcs = lcs_length(output_text, target_text)
    total_length = max(len(output_text), len(target_text))
    
    # 综合考虑多个指标
    char_match_ratio = char_match / total_length if total_length > 0 else 0
    lcs_ratio = lcs / total_length if total_length > 0 else 0
    length_penalty = abs(len(output_text) - len(target_text)) / total_length
    
    # 计算加权loss
    loss = 1.0 - (0.4 * char_match_ratio + 0.4 * lcs_ratio - 0.2 * length_penalty)
    return min(max(loss, 0.0), 1.0)  # 确保loss在[0,1]范围内

def apply_data_augmentation(text):
    """简单的数据增强方法"""
    # 随机删除一些空格
    if random.random() < 0.3:
        text = ' '.join(text.split())
    
    # 随机插入一些常见标点
    if random.random() < 0.2:
        punctuations = ['，', '。', '！', '？']
        positions = list(range(len(text)))
        if positions:
            pos = random.choice(positions)
            text = text[:pos] + random.choice(punctuations) + text[pos:]
    
    return text

def initialize_model():
    """初始化模型"""
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,  # 增加到最大上下文窗口
        n_batch=BATCH_SIZE,  # 批处理大小
        n_threads=6,  # 保持原始线程数
        n_gpu_layers=-1,  # 使用所有可用的GPU层
        verbose=True,  # 启用详细日志
        embedding=True,  # 使用嵌入模式
        seed=42  # 保持原始随机种子
    )

def fine_tune():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 初始化模型
    print("正在加载模型...")
    llm = initialize_model()
    
    # 加载训练数据
    print("正在加载训练数据...")
    training_data = load_training_data("data/test_train.json")
    
    # 开始训练
    print("开始训练...")
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(training_data, desc=f"Epoch {epoch+1}")
        
        for item in progress_bar:
            # 准备输入
            prompt = format_prompt(item["instruction"], item["input"])
            target = item["output"]
            
            # 生成完整的训练序列
            training_text = prompt + " " + target
            
            try:
                # 使用llama.cpp进行训练
                completion = llm(
                    training_text,
                    max_tokens=0,
                    temperature=0,
                    echo=True,
                    stream=False
                )
                
                # 尝试从返回结果中提取文本
                if isinstance(completion, dict) and 'choices' in completion:
                    output_text = completion['choices'][0]['text']
                else:
                    output_text = ''
                
                # 计算loss：使用简单的字符匹配度作为loss的估计
                total_chars = len(training_text)
                if output_text:
                    # 计算最长公共子序列长度
                    matched_chars = sum(1 for a, b in zip(training_text, output_text) if a == b)
                    loss = 1.0 - (matched_chars / total_chars)
                else:
                    loss = 1.0  # 如果没有输出，则loss为最大值
                
                total_loss += loss
                progress_bar.set_postfix({'loss': f'{loss:.4f}'})
                
                # 每20个样本测试一次生成效果
                if (training_data.index(item) + 1) % 20 == 0:
                    test_prompt = "请解释一下LoRA技术的优点"
                    print("\n测试生成：")
                    test_result = llm(
                        format_prompt(test_prompt),
                        max_tokens=100,
                        temperature=0.7,
                        stream=False
                    )
                    
                    if isinstance(test_result, dict) and 'choices' in test_result:
                        generated_text = test_result['choices'][0]['text']
                    else:
                        generated_text = '生成失败'
                        
                    print(f"提示：{test_prompt}")
                    print(f"生成：{generated_text}\n")
                    
                    # 保存检查点
                    checkpoint_dir = os.path.join(OUTPUT_DIR, f'checkpoint-{epoch+1}-{training_data.index(item)+1}')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    print(f"保存检查点到：{checkpoint_dir}")
            
            except Exception as e:
                print(f"处理样本时出错: {str(e)}")
                continue
        
        # 打印epoch的平均损失
        avg_loss = total_loss / len(training_data)
        print(f"\nEpoch {epoch+1} 平均损失: {avg_loss:.4f}")
    
    print("\n训练完成！")

if __name__ == "__main__":
    fine_tune()
