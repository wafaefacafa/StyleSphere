from llama_cpp import Llama
import json
import os
from tqdm import tqdm
from obsidian_loader import ObsidianLoader

# 配置参数
MODEL_PATH = "E:/R1/TheBloke/Llama-2-7B-Chat-GGUF"  # 稍后会在这里加上具体的文件名
OUTPUT_DIR = "outputs"
USE_CUDA = True  # 启用CUDA支持
OBSIDIAN_VAULT = "E:/kubook"  # Obsidian vault 路径

def find_model_file(model_dir):
    """在指定目录中查找.gguf模型文件"""
    if not os.path.exists(model_dir):
        return None
    
    for file in os.listdir(model_dir):
        if file.endswith('.gguf'):
            return os.path.join(model_dir, file)
    return None

def load_training_data(file_path="train.json"):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_prompt(instruction, context="", input_text=""):
    """格式化提示模板"""
    system_prompt = """你是一个专业的中文AI助手。你只能用中文回答问题。注意以下要求：
1. 保持逻辑清晰，语言连贯，避免前言不搭后语
2. 如果提供了知识库内容，必须基于知识库内容作答
3. 不要编造内容，如果不确定就说"抱歉，我对这方面不太确定"
4. 每个回答都应该有清晰的重点和结构
5. 如果遇到复杂问题，需要分点解释"""
    
    # 创建一个自然的对话格式
    prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
    
    if context:
        prompt += f"根据以下参考资料回答：\n\n{context}\n\n"
    
    prompt += f"{instruction}"
    
    if input_text:
        prompt += f"\n\n补充信息：{input_text}"
    
    prompt += " [/INST]"
    return prompt

def filter_response(text):
    """过滤和格式化输出文本"""
    # 移除可能的特殊标记
    text = (text.replace("[/INST]", "")
               .replace("[INST]", "")
               .replace("<<SYS>>", "")
               .replace("<</SYS>>", "")
               .strip())
    
    # 确保回答是中文
    chinese_char_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    if chinese_char_count < len(text) * 0.5:  # 如果中文字符少于50%
        return "抱歉，我将用中文重新组织答案。"
    
    # 检查回答的完整性
    if len(text) < 20:  # 提高最小长度要求
        return "抱歉，让我重新思考并给出更完整的回答。"
        
    # 检查是否包含多个句子
    sentences = [s for s in text.split('。') if s.strip()]
    if len(sentences) < 2:
        return "抱歉，让我重新组织语言，给出更详细的解释。"
    
    # 删除可能的英文回复
    lines = text.split('\n')
    chinese_lines = [line for line in lines if any('\u4e00' <= char <= '\u9fff' for char in line)]
    text = '\n'.join(chinese_lines)
    
    return text

def search_knowledge_base(query, obsidian_loader):
    """搜索知识库相关内容"""
    results = obsidian_loader.search_documents(query)
    if not results:
        return ""
    
    # 合并相关文档内容
    context = "\n\n".join([
        f"文档：{doc['title']}\n{doc['content'][:500]}..."  # 只使用前500个字符
        for doc in results[:3]  # 最多使用前3个相关文档
    ])
    
    return context

def main():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 查找模型文件
    model_file = find_model_file(MODEL_PATH)
    if not model_file:
        print(f"错误：在目录 {MODEL_PATH} 中未找到.gguf模型文件")
        return
    
    print(f"找到模型文件：{os.path.basename(model_file)}")
    
    # 初始化模型
    print(f"正在加载模型: {os.path.basename(model_file)}...")
    llm = Llama(
        model_path=model_file,
        n_ctx=2048,
        n_threads=6,
        n_gpu_layers=-1,
        verbose=True
    )
    
    # 加载训练数据
    print("正在加载训练数据...")
    training_data = load_training_data()
    
    # 对每个样本进行处理
    print("开始处理训练数据...")
    for item in tqdm(training_data["data"]):
        # 准备输入
        prompt = format_prompt(item["instruction"], item["input"])
        
        try:
            # 生成回复
            completion = llm(
                prompt,
                max_tokens=512,
                temperature=0.3,  # 降低温度值以获得更稳定的输出
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stop=["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"],
                echo=False
            )
            
            # 获取生成的文本
            response = completion['choices'][0]['text'] if completion.get('choices') else ""
            
            # 过滤和格式化输出
            generated_text = filter_response(response.strip())
            
            # 打印结果对比
            print("\n===== 输出结果 =====")
            print(f"问题: {item['instruction']}")
            print(f"模型回答: {generated_text}")
            print("==================\n")
            
            # 保存结果
            result = {
                "prompt": prompt,
                "expected": item["output"],
                "generated": generated_text
            }
            
            # 写入结果
            with open(os.path.join(OUTPUT_DIR, "results.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                
        except Exception as e:
            print(f"生成过程中出错: {e}")
            continue

if __name__ == "__main__":
    main()
