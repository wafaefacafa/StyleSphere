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

# 配置参数
MODEL_PATH = "E:/R1/TheBloke/Llama-2-7B-Chat-GGUF"
OUTPUT_DIR = "outputs"
BATCH_SIZE = 1
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
TRAIN_STEPS = 1000
OUTPUT_DIR = "outputs"

def create_and_prepare_model():
    # 加载基础模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # 准备LoRA配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    
    # 准备用于训练的模型
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def prepare_dataset(tokenizer):
    # 加载数据集
    dataset = load_dataset("json", data_files="data/train.json")
    
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
            max_length=512,
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

def train():
    # 创建并准备模型
    model, tokenizer = create_and_prepare_model()
    
    # 准备数据集
    train_dataset = prepare_dataset(tokenizer)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        max_steps=TRAIN_STEPS,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
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
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model()

if __name__ == "__main__":
    train()
