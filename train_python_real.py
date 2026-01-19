import json
import os
import torch
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

# Configuration
# Switching to Llama 3 8B with 4-bit Quantization (QLoRA)
# using unsloth's ungated version for ease of access
MODEL_ID = "unsloth/llama-3-8b-Instruct"
# MAX_STEPS = 100  <-- Removed fixed steps
NUM_EPOCHS = 3     # Train for 3 full passes over the data
OUTPUT_DIR = "outputs/llama3_qlora_test"

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
        # max_steps=MAX_STEPS,   <-- Replaced with epochs
        num_train_epochs=NUM_EPOCHS,
        learning_rate=2e-4,
        fp16=True,                  # Use FP16 for training stability on GPU
        logging_steps=10,
        save_strategy="no",
        optim="paged_adamw_32bit",   # Optimizer optimized for VRAM (requires bitsandbytes)
        gradient_checkpointing=True, # Critical for saving VRAM
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    print(f"🚀 Starting training ({NUM_EPOCHS} epochs)...")
    trainer.train()
    
    print(f"✅ Training complete. Saved to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
