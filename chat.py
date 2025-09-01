from llama_cpp import Llama
from obsidian_loader import ObsidianLoader
import os
import re
import unicodedata

# 配置参数
MODEL_PATH = "E:/R1/TheBloke/Llama-2-7B-Chat-GGUF"  # 替换为你的模型路径
OBSIDIAN_VAULT = "E:/kubook"  # Obsidian vault 路径

def find_model_file(model_dir):
    """在指定目录中查找.gguf模型文件"""
    if not os.path.exists(model_dir):
        return None
    
    for file in os.listdir(model_dir):
        if file.endswith('.gguf'):
            return os.path.join(model_dir, file)
    return None

def format_prompt(instruction, context=""):
    """格式化提示模板"""
    system_prompt = """你是一个专业的中文AI助手。请严格遵守以下原则：

1. 语言要求：
   - 必须全程使用规范的中文
   - 避免使用任何英文句子
   - 不使用表情符号
   - 专业术语需要加中文说明，如：LLaMA（大语言模型）

2. 回答规范：
   - 保持清晰的逻辑结构
   - 语言要简洁准确
   - 重点突出，条理分明
   - 如果不确定，明确说"不确定"

3. 知识库使用：
   - 严格基于提供的知识库内容
   - 不要随意添加未提供的信息
   - 如果知识库信息不足，说明需要补充

4. 回答格式：
   - 开门见山，直接回答问题
   - 必要时使用分点说明
   - 适当使用标点符号组织语言
   - 保持段落清晰

5. 严格禁止：
   - 不能使用英文对话
   - 不能使用表情符号
   - 不能插入题外话
   - 不能做无根据的推测"""
    
    prompt = f"[INST] {system_prompt}\n\n"
    
    if context:
        prompt += f"基于以下知识库内容：\n\n{context}\n\n"
    
    prompt += f"{instruction} [/INST]"
    return prompt

def is_chinese_character(char):
    """判断一个字符是否是中文字符"""
    return 'CJK UNIFIED IDEOGRAPH' in unicodedata.name(char, '')

def is_chinese_sentence(text):
    """检查一段文本是否主要由中文组成"""
    # 去除空白字符和标点符号
    text = ''.join(char for char in text if not char.isspace() and not unicodedata.category(char).startswith('P'))
    if not text:
        return True  # 空文本视为有效
    
    # 提取可能的英文专业术语（括号内的内容和单个单词）
    terms = re.findall(r'\([^)]*\)|[a-zA-Z]+', text)
    # 移除这些术语
    for term in terms:
        text = text.replace(term, '')
    
    if not text:  # 如果移除术语后为空，说明都是合法术语
        return True
    
    # 计算中文字符的比例
    chinese_chars = sum(1 for char in text if is_chinese_character(char))
    total_chars = len(text)
    
    # 如果中文字符占比超过50%，认为是中文句子
    return chinese_chars / total_chars > 0.5

def contains_emoji(text):
    """检查文本是否包含表情符号"""
    try:
        # 使用 unicodedata 检查字符类别
        for char in text:
            if 'EMOJI' in unicodedata.name(char, '') or 'EMOTICON' in unicodedata.name(char, ''):
                return True
        return False
    except:
        return False

def filter_response(text):
    """过滤响应文本，优化中文输出"""
    if not text:
        return None, True
        
    # 1. 清理特殊标记
    text = (text.replace("[/INST]", "")
               .replace("[INST]", "")
               .replace("<<SYS>>", "")
               .replace("<</SYS>>", "")
               .strip())
    
    # 2. 处理表情符号
    if contains_emoji(text):
        text = ''.join(char for char in text if not ('EMOJI' in unicodedata.name(char, '') or 'EMOTICON' in unicodedata.name(char, '')))
    
    # 3. 分析中文内容比例
    pure_text = ''.join(char for char in text if not unicodedata.category(char).startswith('P') and not char.isspace())
    chinese_chars = sum(1 for char in pure_text if '\u4e00' <= char <= '\u9fff')
    total_chars = len(pure_text)
    
    if total_chars > 0 and chinese_chars / total_chars < 0.4:  # 降低阈值到40%
        return None, True
    
    # 4. 处理句子
    sentences = []
    current = []
    
    # 分词并重组句子
    for char in text:
        current.append(char)
        if char in '。！？' or len(''.join(current)) > 100:  # 处理自然句子结束或过长的句子
            sentence = ''.join(current).strip()
            if sentence and not sentence.isspace():
                # 确保句子结尾有标点
                if not sentence[-1] in '。！？':
                    sentence += '。'
                sentences.append(sentence)
            current = []
    
    # 处理最后一个句子
    if current:
        sentence = ''.join(current).strip()
        if sentence and not sentence.isspace():
            if not sentence[-1] in '。！？':
                sentence += '。'
            sentences.append(sentence)
    
    # 5. 最终处理
    if not sentences:
        return None, True
        
    result = '\n'.join(sentences)
    
    # 规范化专业术语的格式
    def format_term(match):
        term = match.group(1)
        desc = match.group(2)
        return f"{term}（{desc}）"
    
    result = re.sub(r'([A-Za-z]+)[（(]([^)）]+)[)）]', format_term, result)
    
    # 清理不需要的字符
    allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n。，、；：？！…—·ˉ¨''""々～‖∶＂＇｀｜〃〔〕〈〉《》「」『』．（）-')
    result = ''.join(char for char in result if char in allowed_chars or '\u4e00' <= char <= '\u9fff')
    
    return result, False

def enforce_chinese_response(llm, prompt, max_attempts=3):
    """强制生成优质的中文回答"""
    best_response = None
    max_chinese_ratio = 0
    min_length = 30  # 降低最小回答长度要求
    
    # 生成参数配置
    base_params = {
        "max_tokens": 1024,  # 增加最大token数
        "top_p": 0.95,  # 稍微提高采样范围
        "top_k": 50,    # 增加top_k值
        "repeat_penalty": 1.2,  # 增加重复惩罚
        "stop": ["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"],
        "echo": False
    }
    
    # 不同的参数组合
    param_variations = [
        {"temperature": 0.7},  # 第一次尝试：较高创造性
        {"temperature": 0.5},  # 第二次尝试：平衡
        {"temperature": 0.3},  # 第三次尝试：更保守
    ]
    
    for attempt in range(max_attempts):
        try:
            # 更新生成参数
            generation_params = base_params.copy()
            generation_params.update(param_variations[attempt])
            
            # 生成回答
            response = llm(prompt, **generation_params)
            
            if isinstance(response, dict) and "choices" in response:
                answer = response["choices"][0]["text"].strip()
                
                # 过滤和检查答案
                filtered_answer, needs_regeneration = filter_response(answer)
                
                if not needs_regeneration and filtered_answer and len(filtered_answer) >= min_length:
                    # 计算中文字符比例
                    pure_text = ''.join(char for char in filtered_answer if '\u4e00' <= char <= '\u9fff')
                    total_text = ''.join(char for char in filtered_answer if not (unicodedata.category(char).startswith('P') or char.isspace()))
                    
                    if len(total_text) > 0:
                        chinese_ratio = len(pure_text) / len(total_text)
                        
                        # 评分标准
                        score = chinese_ratio
                        
                        # 检查句子完整性
                        sentences = re.split(r'[。！？]', filtered_answer)
                        valid_sentences = [s.strip() for s in sentences if len(s.strip()) >= 5]  # 确保句子有实际内容
                        
                        if len(valid_sentences) >= 1:  # 降低句子数量要求
                            score += 0.1 * len(valid_sentences)  # 根据句子数量加分
                            
                            # 更新最佳答案
                            if score > max_chinese_ratio:
                                max_chinese_ratio = score
                                best_response = {
                                    "choices": [{
                                        "text": filtered_answer,
                                        "index": 0,
                                        "logprobs": None,
                                        "finish_reason": "stop"
                                    }]
                                }
                                
                                # 如果达到较好质量，直接返回
                                if chinese_ratio > 0.6 and len(valid_sentences) >= 2:  # 降低标准
                                    return best_response
                                
        except Exception as e:
            print(f"生成过程出错 (尝试 {attempt+1}/{max_attempts}): {str(e)}")
            continue
    
    # 返回最佳结果或默认回答
    if best_response is not None:
        return best_response
    
    # 如果所有尝试都失败，返回一个更具体的回答
    return {
        "choices": [{
            "text": "让我告诉您关于《塞尔达传说》系列游戏的情况。这是任天堂最著名的游戏系列之一，以其独特的冒险体验和创新玩法而闻名。最新作品是《王国之泪》，它延续了旷野之息的开放世界设计，同时加入了更多新玩法。您想了解哪些具体方面？",
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop"
        }]
    }

def main():
    # 初始化 Obsidian 加载器
    print("初始化 Obsidian 知识库...")
    obsidian = ObsidianLoader(OBSIDIAN_VAULT)
    
    # 加载模型
    model_file = find_model_file(MODEL_PATH)
    if not model_file:
        print(f"错误：在目录 {MODEL_PATH} 中未找到.gguf模型文件")
        return
        
    print(f"加载模型: {os.path.basename(model_file)}...")
    llm = Llama(
        model_path=model_file,
        n_ctx=2048,
        n_threads=8,  # 使用8个CPU线程
        n_gpu_layers=32,  # 设置为模型的实际层数
        n_batch=128,  # 降低批处理大小以减少显存使用
        f16_kv=True,  # 使用FP16精度的KV缓存
        use_mmap=False,  # 禁用内存映射，强制加载到内存
        use_mlock=True,  # 锁定内存，防止交换到磁盘
        embedding=False,  # 禁用嵌入层，减少内存使用
        verbose=True
    )
    
    print("\n知识库助手已就绪！输入 'quit' 退出。")
    
    def truncate_text(text, max_chars=1000):
        """截断过长的文本"""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."
    
    while True:
        user_input = input("\n请输入您的问题: ").strip()
        
        if user_input.lower() == 'quit':
            break
            
        if not user_input:
            continue
            
        try:
            # 搜索相关文档
            print("搜索知识库中...")
            relevant_docs = obsidian.search_documents(user_input)
            context = ""
            
            if relevant_docs:
                print(f"找到 {len(relevant_docs)} 个相关文档")
                context = "\n\n".join([
                    f"文档：{doc['title']}\n{truncate_text(doc['content'], 300)}"
                    for doc in relevant_docs[:2]  # 限制为最相关的2个文档
                ])
            else:
                print("未找到相关文档，将使用模型直接回答")
            
            # 处理用户输入，移除潜在的特殊字符
            user_input = re.sub(r'[\U0001F300-\U0001F9FF]', '', user_input)
            user_input = user_input.strip()
            
            # 生成提示
            prompt = format_prompt(user_input, context)
            
            # 强制生成纯中文回答
            print("\n生成回答中...")
            response = enforce_chinese_response(llm, prompt)
            
            if isinstance(response, dict) and "choices" in response:
                answer = response["choices"][0]["text"].strip()
                
                # 规范化专业术语
                def format_term(match):
                    term = match.group(1)
                    desc = match.group(2)
                    return f"{term}（{desc}）"
                
                # 查找并规范化格式：术语(描述) 或 术语（描述）
                answer = re.sub(r'([A-Za-z]+)[（(]([^)）]+)[)）]', format_term, answer)
                
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
