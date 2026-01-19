import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
import os
import threading
from model_config import get_current_config

# ==========================================
# ⚙️ Configuration
# ==========================================
# 1. First choice: The fully merged independent model
MERGED_MODEL_PATH = "outputs/llama3_qlora_test_merged_full"

# 2. Second choice: Check model_config.py (Adapter mode)
config = get_current_config()
ADAPTER_PATH = config["output_dir"]
BASE_MODEL_ID = config["model_id"]

# Decide which to load
if os.path.exists(MERGED_MODEL_PATH):
    print(f"🦁 Found Merged Independent Model at: {MERGED_MODEL_PATH}")
    MODEL_PATH = MERGED_MODEL_PATH
    USE_ADAPTER = False
else:
    print(f"🦁 Merged model not found. Using Adapter mode: {BASE_MODEL_ID} + {ADAPTER_PATH}")
    MODEL_PATH = BASE_MODEL_ID
    USE_ADAPTER = True

# ==========================================
# 🚀 Load Model (4-bit for 8GB VRAM)
# ==========================================
print("⏳ Loading model... (This takes a minute)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

if USE_ADAPTER:
    if os.path.exists(ADAPTER_PATH):
        print(f"🔗 Loading Adapter: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    else:
        print("⚠️ Adapter not found. Running Base Model only.")

model.eval()
print("✅ Model Loaded Successfully!")

# ==========================================
# 💬 Chat Logic
# ==========================================
def generate_response(message, history):
    # Prepare prompt
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

# ==========================================
# 🖥️ GUI Setup
# ==========================================
chatbot = gr.ChatInterface(
    generate_response,
    title="🦁 StyleSphere AI Chat (Local)",
    description=f"Running: {MODEL_PATH}",
    # theme="soft", # Removed to fix TypeError
    examples=["介绍一下你自己", "最近有什么时尚趋势？", "你好"]
)

if __name__ == "__main__":
    # Auto-open browser
    chatbot.launch(inbrowser=True, share=False)
