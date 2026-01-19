from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import yaml

def create_and_prepare_model(config):
    # 加载基础模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name_or_path'],
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'])
    
    # 准备LoRA配置
    lora_config = LoraConfig(
        r=config['lora_rank'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config['lora_target'].split(',')
    )
    
    # 准备用于训练的模型
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def prepare_dataset(tokenizer, config, data_path):
    # 加载数据集
    dataset = load_dataset("json", data_files=data_path)
    
    def preprocess_function(examples):
        # 将指令和输出组合成完整的提示
        prompts = [
            f"### 指令：{instruction}\n### 输入：{input_text}\n### 响应：{output}"
            for instruction, input_text, output in zip(
                examples["instruction"],
                examples["input"],
                examples["output"]
            )
        ]
        
        # 对文本进行编码
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=config['max_source_length'],
            padding="max_length"
        )
        return tokenized

    # 处理数据集
    tokenized_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_dataset

def train(config_path="e:/AI/llama2-train/test_config.yaml", data_path="e:/AI/llama2-train/data/test_train.json"):
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建并准备模型
    print("正在加载模型...")
    model, tokenizer = create_and_prepare_model(config)
    
    # 准备数据集
    print("正在准备数据集...")
    train_dataset = prepare_dataset(tokenizer, config, data_path)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        max_steps=config['max_steps'],
        warmup_steps=config['warmup_steps'],
        logging_steps=config['logging_steps'],
        save_steps=config['save_steps'],
        save_total_limit=config['save_total_limit'],
        bf16=config['bf16'],
        remove_unused_columns=False
    )
    
    # 创建数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    print("开始训练...")
    # 开始训练
    trainer.train()
    
    # 保存模型
    print(f"保存模型到 {config['output_dir']}")
    trainer.save_model()

if __name__ == "__main__":
    train()
