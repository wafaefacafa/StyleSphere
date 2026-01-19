# ==========================================
# 🦁 MODEL ZOO (模型动物园) - 配置文件
# ==========================================
# 在这里定义您想要使用的模型。
# 可以在这随时切换 'SELECTED_MODEL_KEY' 来更换训练对象。

# 当前选择的模型 (修改此处来切换!)
SELECTED_MODEL_KEY = "llama3-8b"
# SELECTED_MODEL_KEY = "qwen2.5-7b"
# SELECTED_MODEL_KEY = "gemma-2-9b"
# SELECTED_MODEL_KEY = "mistral-7b"

# 模型预设库
MODEL_ZOO = {
    "llama3-8b": {
        "name": "Llama 3 8B Instruct",
        "model_id": "unsloth/llama-3-8b-Instruct",
        "output_dir": "outputs/llama3_qlora_test",
        "description": "Meta最新一代模型，智能程度高，通用性强。"
    },
    "qwen2.5-7b": {
        "name": "Qwen 2.5 7B Instruct",
        "model_id": "unsloth/Qwen2.5-7B-Instruct", 
        "output_dir": "outputs/qwen2.5_qlora_v1",
        "description": "阿里通义千问2.5，中文能力极强，数学和编程能力优秀。"
    },
    "gemma-2-9b": {
        "name": "Gemma 2 9B Instruct",
        "model_id": "unsloth/gemma-2-9b-it",
        "output_dir": "outputs/gemma2_qlora_v1",
        "description": "Google最新开源模型，在9B尺寸下性能惊人 (8GB显存4bit刚好能塞下)。"
    },
    "mistral-7b": {
        "name": "Mistral 7B v0.3",
        "model_id": "unsloth/mistral-7b-instruct-v0.3",
        "output_dir": "outputs/mistral_qlora_v1",
        "description": "经典的7B最强基座之一，社区支持极好。"
    }
}

def get_current_config():
    if SELECTED_MODEL_KEY not in MODEL_ZOO:
        raise ValueError(f"Selected model '{SELECTED_MODEL_KEY}' not found in MODEL_ZOO")
    return MODEL_ZOO[SELECTED_MODEL_KEY]
