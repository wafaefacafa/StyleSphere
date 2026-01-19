import json
import os
import math
import random
from llama_cpp import Llama
from tqdm import tqdm
import yaml
from typing import List, Dict, Any

def load_config(config_path="train_stylesphere.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# 加载StyleSphere配置
CONFIG = load_config()

def load_training_data(file_path):
    """加载训练数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def format_prompt(instruction, input_text=""):
    """StyleSphere的提示模板"""
    if input_text:
        return f"[INST] {instruction}\n{input_text} [/INST]"
    return f"[INST] {instruction} [/INST]"

def get_learning_rate(current_step, total_steps, config):
    """实现带warmup的余弦学习率调度"""
    warmup_steps = config['warmup_steps']
    max_lr = config['learning_rate']
    min_lr = max_lr * 0.1  # 最低学习率为最大学习率的10%

    if current_step < warmup_steps:
        # 线性预热
        return max_lr * (current_step / warmup_steps)
    else:
        # 余弦衰减
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def calculate_loss(output_text, target_text):
    """计算生成文本与目标文本之间的损失"""
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
    return min(max(loss, 0.0), 1.0)

def apply_data_augmentation(text):
    """StyleSphere专用的数据增强方法"""
    # 添加时尚相关的表情符号
    emojis = ['💄', '👗', '👠', '👜', '💅', '👚', '👒', '💃', '✨', '❤️']
    if random.random() < 0.3:
        text = text + ' ' + random.choice(emojis)
    
    # 添加时尚相关的称谓
    nicknames = ['亲爱的', '宝贝', '美女', '姐妹']
    if random.random() < 0.2 and not any(name in text for name in nicknames):
        text = random.choice(nicknames) + '～' + text
    
    return text

def initialize_model(config):
    """初始化模型"""
    model_params = config.get('model_params', {})
    return Llama(
        model_path=config['model_name_or_path'],
        n_ctx=model_params.get('n_ctx', config.get('max_source_length', 2048)),
        n_batch=config.get('per_device_train_batch_size', 2),
        n_threads=6,
        n_gpu_layers=model_params.get('n_gpu_layers', 0),  # 从 model_params 读取 n_gpu_layers
        verbose=True,
        embedding=True,
        seed=42
    )

def evaluate_model(llm, eval_data):
    """评估模型性能"""
    total_loss = 0
    for item in eval_data:
        prompt = format_prompt(item["instruction"], item["input"])
        target = item["output"]
        
        try:
            completion = llm(
                prompt,
                max_tokens=len(target) + 50,
                temperature=0.1,
                stream=False
            )
            
            if isinstance(completion, dict) and 'choices' in completion:
                output_text = completion['choices'][0]['text']
                loss = calculate_loss(output_text, target)
                total_loss += loss
            else:
                total_loss += 1.0
                
        except Exception as e:
            print(f"评估时出错: {str(e)}")
            total_loss += 1.0
            
    return total_loss / len(eval_data)

def train_stylesphere():
    """StyleSphere专用训练流程"""
    config = CONFIG
    
    # 创建输出和日志目录
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['logging_dir'], exist_ok=True)
    log_file_path = os.path.join(config['logging_dir'], 'training_log.jsonl')
    
    # 初始化模型
    print("正在加载模型...")
    llm = initialize_model(config)
    
    # 加载训练数据
    print("正在加载训练数据...")
    training_data = load_training_data(config['train_data'])
    validation_data = load_training_data(config['val_data'])
    
    # 计算总训练步数
    total_steps = config['max_steps']
    current_step = 0
    accumulated_loss = 0
    best_eval_loss = float('inf')
    
    # 开始训练
    print("开始StyleSphere训练...")
    num_epochs = total_steps // len(training_data) + 1
    
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        for epoch in range(num_epochs):
            total_loss = 0
            progress_bar = tqdm(training_data, desc=f"Epoch {epoch+1}")
            
            for item in progress_bar:
                current_step += 1
                if current_step > total_steps:
                    break
                    
                current_lr = get_learning_rate(current_step, total_steps, config)
                
                # 准备输入，应用StyleSphere的数据增强
                prompt = format_prompt(item["instruction"], item["input"])
                target = apply_data_augmentation(item["output"])
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
                    
                    # 提取输出并计算loss
                    if isinstance(completion, dict) and 'choices' in completion:
                        output_text = completion['choices'][0]['text']
                        loss = calculate_loss(output_text, target)
                    else:
                        output_text = ''
                        loss = 1.0
                    
                    # 梯度累积
                    accumulated_loss += loss / config['gradient_accumulation_steps']
                    
                    if current_step % config['gradient_accumulation_steps'] == 0:
                        total_loss += accumulated_loss
                        accumulated_loss = 0
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'lr': f'{current_lr:.6f}'
                    })
                    
                    # 记录日志
                    log_entry = {
                        'step': current_step,
                        'loss': loss,
                        'lr': current_lr
                    }
                    
                    # 定期评估和保存
                    if current_step % config['logging_steps'] == 0:
                        eval_loss = evaluate_model(llm, validation_data[:10])
                        log_entry['eval_loss'] = eval_loss
                        print(f"\n步骤 {current_step} 评估损失: {eval_loss:.4f}")
                        
                        # 如果是最佳模型，保存检查点
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            checkpoint_dir = os.path.join(config['output_dir'], f'best_checkpoint')
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            print(f"发现更好的模型！保存检查点到：{checkpoint_dir}")
                            
                        # 测试生成效果
                        test_prompt = "请为一个身高165cm的女生推荐夏季穿搭"
                        print("\n测试生成：")
                        test_result = llm(
                            format_prompt(test_prompt),
                            max_tokens=200,
                            temperature=0.7,
                            stream=False
                        )
                        
                        if isinstance(test_result, dict) and 'choices' in test_result:
                            generated_text = test_result['choices'][0]['text']
                            print(f"提示：{test_prompt}")
                            print(f"生成：{generated_text}\n")
                    
                    log_file.write(json.dumps(log_entry) + '\n')
                
                except Exception as e:
                    print(f"处理样本时出错: {str(e)}")
                    continue
            
            # 打印epoch的平均损失
            avg_loss = total_loss / len(training_data)
            print(f"\nEpoch {epoch+1} 平均损失: {avg_loss:.4f}")
            
            # epoch结束时的评估
            epoch_eval_loss = evaluate_model(llm, validation_data)
            print(f"Epoch {epoch+1} 验证集损失: {epoch_eval_loss:.4f}")
            
            if current_step > total_steps:
                break
    
    print(f"\nStyleSphere训练完成！最佳验证集损失: {best_eval_loss:.4f}")

if __name__ == "__main__":
    train_stylesphere()
