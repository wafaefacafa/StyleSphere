from llama_cpp import Llama
import os

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
    system_prompt = """你是一位专业的AI技术专家，尤其专注于Llama和LoRA等大模型技术领域。请严格遵守以下原则：

1. 专业性要求：
   - 必须准确区分LoRA(低秩适配技术)和LoRa(无线技术)
   - 准确理解和使用AI技术概念
   - 回答必须体现深入的专业知识
   - 涉及Llama、LoRA等技术时必须准确描述其原理

2. 语言要求：
   - 必须全程使用规范的中文回答
   - 所有专业术语都要加中文解释
   - 严格禁止使用英文句子
   - 使用中文数字和标点符号

3. 知识范围：
   - 专注于AI模型训练和优化技术
   - 重点是Llama系列模型的特点和应用
   - 详细说明LoRA等参数高效微调方法
   - 准确描述模型训练和评估方法

4. 表达规范：
   - 结构要层次分明
   - 解释要深入浅出
   - 适当举例说明
   - 突出技术重点"""
    
    prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n请用中文回答以下问题：\n{instruction} [/INST]"
    return prompt

def main():
    # 基本配置
    MODEL_PATH = "E:/R1/TheBloke/Llama-2-7B-Chat-GGUF"
    LORA_DIR = "outputs/lora-test-optimized/best_checkpoint"  # 指向训练好的LoRA模型
    
    # 查找基础模型
    model_file = find_model_file(MODEL_PATH)
    if not model_file:
        print(f"错误：在目录 {MODEL_PATH} 中未找到.gguf模型文件")
        return

    print(f"加载模型：{os.path.basename(model_file)}")
    llm = Llama(
        model_path=model_file,
        n_ctx=2048,
        n_threads=8,  # 使用8个线程
        verbose=True
    )
    
    # 测试问题
    test_questions = [
        "请详细解释LoRA（低秩适配）技术在大语言模型微调中的优势和原理",
        "从技术角度分析Llama 2相比Llama 1的主要改进和创新",
        "在使用LoRA微调Llama 2时，如何选择和优化关键参数？请给出具体建议",
    ]
    
    print("\n开始测试...")
    for question in test_questions:
        print(f"\n问题：{question}")
        prompt = format_prompt(question)
        
        try:
            response = llm(
                prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repeat_penalty=1.1,
                stop=["[INST]", "[/INST]"],
            )
            
            answer = response["choices"][0]["text"].strip()
            print(f"\n回答：{answer}")
            
        except Exception as e:
            print(f"生成出错：{str(e)}")

if __name__ == "__main__":
    main()
