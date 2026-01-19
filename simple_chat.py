from llama_cpp import Llama
import os

# 配置参数
MODEL_PATH = "E:/R1/TheBloke/Llama-2-7B-Chat-GGUF"  # 替换为你的模型路径

def find_model_file(model_dir):
    """在指定目录中查找.gguf模型文件"""
    if not os.path.exists(model_dir):
        return None
    
    for file in os.listdir(model_dir):
        if file.endswith('.gguf'):
            return os.path.join(model_dir, file)
    return None

def format_prompt(instruction):
    """格式化提示模板"""
    system_prompt = """你是一个专业的中文AI助手。请严格遵守以下原则：

1. 语言要求：
   - 必须全程使用规范的中文
   - 避免使用任何英文句子
   - 不使用表情符号
   - 专业术语需要加中文说明

2. 回答规范：
   - 保持清晰的逻辑结构
   - 语言要简洁准确
   - 重点突出，条理分明
   - 如果不确定，明确说"不确定"

3. 回答格式：
   - 开门见山，直接回答问题
   - 必要时使用分点说明
   - 适当使用标点符号组织语言
   - 保持段落清晰"""
    
    prompt = f"[INST] {system_prompt}\n\n{instruction} [/INST]"
    return prompt

def main():
    # 加载模型
    model_file = find_model_file(MODEL_PATH)
    if not model_file:
        print(f"错误：在目录 {MODEL_PATH} 中未找到.gguf模型文件")
        return
        
    print(f"加载模型: {os.path.basename(model_file)}...")
    llm = Llama(
        model_path=model_file,
        n_ctx=2048,
        n_threads=8,
        n_gpu_layers=32,
        n_batch=128,
        f16_kv=True,
        use_mmap=False,
        use_mlock=True,
        embedding=False,
        verbose=True
    )
    
    print("\nAI助手已就绪！输入 'quit' 退出。")
    
    while True:
        user_input = input("\n请输入您的问题: ").strip()
        
        if user_input.lower() == 'quit':
            break
            
        if not user_input:
            continue
            
        try:
            # 生成提示
            prompt = format_prompt(user_input)
            
            print("\n生成回答中...")
            response = llm(
                prompt,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                repeat_penalty=1.2,
                stop=["[INST]", "[/INST]"],
                echo=False
            )
            
            if isinstance(response, dict) and "choices" in response:
                answer = response["choices"][0]["text"].strip()
                # 分段展示
                paragraphs = answer.split('\n')
                print("\n回答：")
                for para in paragraphs:
                    if para.strip():
                        print(para.strip())
            else:
                print("\n抱歉，我需要重新组织语言。请您换个方式提问，我会更好地回答。")
            
        except Exception as e:
            print(f"\n发生错误: {str(e)}")

if __name__ == "__main__":
    main()
