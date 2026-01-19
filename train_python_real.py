import json
import os
import torch
import time
import gc
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Set HF Mirror for connection issues
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    from model_config import get_current_config
except ImportError:
    # Fallback if config is missing (safety net)
    def get_current_config():
        return {
            "model_id": "unsloth/llama-3-8b-Instruct", 
            "output_dir": "outputs/llama3_qlora_temp"
        }

# Load configuration from Model Zoo
config = get_current_config()
MODEL_ID = config["model_id"]
OUTPUT_DIR = config["output_dir"]

print(f"🦁 Using Model from Zoo: {config.get('name', MODEL_ID)}")
print(f"📂 Output Directory: {OUTPUT_DIR}")

# ==========================================
# ⚙️ 训练策略设置 (Training Strategy)
# ==========================================
SYSTEM_PROMPT = "你是一个智能助手。请全程使用中文回答用户的问题，保持专业和友善。"
CYCLES = 5          # 循环次数 (总共跑几轮)
START_CYCLE = 2     # 从第几轮开始 (如果之前中断了，可以修改这个数字续借)
EPOCHS_PER_CYCLE = 3 # 每次跑 3 个 Epoch
WAIT_SECONDS = 30    # 每次休息 30 秒 (增加休息时间以缓解过热/显存碎片)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "train.json")
    
    print(f"Loading data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Convert Alpaca to Chat format for formatting
    formatted_data = []
    for item in data:
        instruction = item.get("instruction", "")
        inp = item.get("input", "")
        out = item.get("output", "")
        
        user_msg = instruction
        if inp:
            user_msg += "\n" + inp
            
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": out}
        ]
        formatted_data.append({"messages": messages})
        
    dataset = Dataset.from_list(formatted_data)
    
    print(f"Loading model: {MODEL_ID}...")
    
    # 4-bit quantization (bitsandbytes) enabled
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Using device: cuda (Confirmed: {torch.cuda.is_available()})")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Enable Gradient Checkpointing (handled by prepare_model_for_kbit_training usually, but explicit is good)
    # model.gradient_checkpointing_enable() # prepare_model... does this
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Formatting function
    def format_chat(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return {"text": text}
    
    dataset = dataset.map(format_chat)
    
    # Tokenize
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=512, # 512 is safe for 8GB VRAM. 1024 might push it.
            padding="max_length"
        )
        
    tokenized_dataset = dataset.map(tokenize, batched=True)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=EPOCHS_PER_CYCLE, # Initial Goal
        learning_rate=2e-4,
        fp16=True,                  # Use FP16 for training stability on GPU
        logging_steps=10,
        save_strategy="epoch",      # Save every epoch so we can resume
        optim="paged_adamw_32bit",   # Optimizer optimized for VRAM (requires bitsandbytes)
        gradient_checkpointing=True, # Critical for saving VRAM
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    print(f"🚀 Starting Cyclic Training: {CYCLES} Cycles x {EPOCHS_PER_CYCLE} Epochs")
    print(f"   Total Planned: {CYCLES * EPOCHS_PER_CYCLE} Epochs")
    print(f"   Resuming from Cycle: {START_CYCLE}")

    for i in range(START_CYCLE, CYCLES + 1):
        print(f"\n🔄 CYCLE {i}/{CYCLES} STARTING...")
        
        # Update target epochs for *this* run (cumulative)
        target_epochs = i * EPOCHS_PER_CYCLE
        trainer.args.num_train_epochs = target_epochs
        
        # Resume from checkpoint if not the first cycle or if resuming mid-stream
        # 只要不是第一轮，或者文件夹里已有对应的checkpoint，通常建议开启resume
        resume = True if i > 1 else False
        
        try:
            trainer.train(resume_from_checkpoint=resume)
        except ValueError as e:
            print(f"⚠️ Resume failed or checkpoint mismatch: {e}")
            print("Trying to continue without resume (may restart epoch count if not careful)...")
            trainer.train(resume_from_checkpoint=False)
        
        print(f"✅ Cycle {i} Complete.")
        model.save_pretrained(OUTPUT_DIR)
        
        # Explicit memory cleanup
        print("🧹 Cleaning up GPU memory...")
        torch.cuda.empty_cache()
        gc.collect()
        
        if i < CYCLES:
            print(f"☕ Resting for {WAIT_SECONDS} seconds to cool down GPU...")
            time.sleep(WAIT_SECONDS)
    
    print(f"🎉 All {CYCLES} cycles complete. Final model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
