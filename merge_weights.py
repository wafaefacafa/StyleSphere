import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import shutil

# Configuration
BASE_MODEL_ID = "unsloth/llama-3-8b-Instruct"
ADAPTER_PATH = "outputs/llama3_qlora_test"
MERGED_OUTPUT_PATH = "outputs/llama3_8b_custom_merged"

def main():
    print(f"⚠️  ATTENTION: Merging requires ~16GB of System RAM (CPU) for Llama-3-8B in FP16.")
    print(f"    Current output path: {MERGED_OUTPUT_PATH}")
    
    # 1. Load Base Model in FP16 (CPU)
    # We use CPU because 8GB VRAM is not enough to hold the full FP16 model for merging
    print(f"⏳ Loading base model '{BASE_MODEL_ID}' into CPU RAM...")
    print("   (This might take a while depending on your RAM speed)")
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            return_dict=True
        )
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        print("💡 Suggestion: Ensure you have at least 16GB RAM available.")
        return

    # 2. Load Tokenizer
    print("⏳ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    # 3. Load & Merge Adapter
    print(f"⏳ Loading LoRA adapter from '{ADAPTER_PATH}' and merging...")
    
    try:
        # Resize token embeddings if added tokens (usually not for basic finetune but good practice)
        base_model.resize_token_embeddings(len(tokenizer))
        
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model = model.merge_and_unload() # The magic function
        print("✅ Weights merged successfully!")

    except Exception as e:
        print(f"❌ Merge failed: {e}")
        return

    # 4. Save Final Model
    print(f"💾 Saving standalone model to '{MERGED_OUTPUT_PATH}'...")
    
    if not os.path.exists(MERGED_OUTPUT_PATH):
        os.makedirs(MERGED_OUTPUT_PATH)
        
    model.save_pretrained(MERGED_OUTPUT_PATH)
    tokenizer.save_pretrained(MERGED_OUTPUT_PATH)

    print(f"🎉 Success! The standalone model is ready at: {MERGED_OUTPUT_PATH}")
    print("   You can now copy this folder to any machine and use it without 'peft' or configuration.")

if __name__ == "__main__":
    main()
