import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
import sys

# Paths
BASE_MODEL_ID = "unsloth/llama-3-8b-Instruct"
# Ensure we map the correct path relative to workspace root
ADAPTER_PATH = "outputs/llama3_qlora_test"

def main():
    # Force set CWD to workspace root if running from elsewhere/IDE to find outputs
    # (Optional safety check)
    if not os.path.exists(ADAPTER_PATH):
        print(f"⚠️ Warning: Adapter path '{ADAPTER_PATH}' not found in current directory.")
        print(f"Current Directory: {os.getcwd()}")
        # configure for e:\AI specifically if needed, but relative path is better generally
        
    print("="*50)
    print("🤖 Llama 3 8B QLoRA Chat Test")
    print(f"Base Model: {BASE_MODEL_ID}")
    print(f"Adapter:    {ADAPTER_PATH}")
    print("="*50)

    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    # 2. Config 4-bit loading (Must match training config)
    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. Load Base Model
    print("Loading base model (this may take a minute)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 4. Load LoRA Adapter
    print(f"Loading trained LoRA adapter from {ADAPTER_PATH}...")
    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    except Exception as e:
        print(f"❌ Error loading adapter: {e}")
        print("Did you run the training script successfully?")
        return

    model.eval()

    print("\n✅ System Ready! Chat with your model directly.")
    print("Type 'exit' to quit.\n")

    # 5. Chat Loop
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue
                
            # Prepare message in ChatML format (Llama 3 standard)
            messages = [
                {"role": "user", "content": user_input}
            ]
            
            input_ids = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6, # Low temperature for more focused answers based on training
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode only the new tokens
            response = outputs[0][input_ids.shape[-1]:]
            text = tokenizer.decode(response, skip_special_tokens=True)
            
            print(f"AI:  {text}\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
