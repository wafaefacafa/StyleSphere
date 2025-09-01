from llama_cpp import Llama

# 加载基础模型
llm = Llama(
    model_path="E:/R1/TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_S.gguf",
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=35  # 启用 GPU 加速
)

# 测试问题列表
test_questions = [
    "请帮我搭配一套适合夏季约会的穿搭",
    "如何选择适合cos的包包？",
    "我是梨形身材，应该如何选择合适的裙装？",
    "今年秋冬流行什么颜色和风格？",
    "如何挑选一双既舒适又时尚的工作鞋？"
]

# 逐个测试问题
for question in test_questions:
    print("\n问题:", question)
    print("\n回答:")
    
    # 构建提示语
    prompt = f"""[INST] <<SYS>>
你是一个专业的AI助手，尝试用专业、准确的语言回答用户的问题。
<</SYS>>

回答以下问题

{question} [/INST]"""
    
    # 生成回答
    output = ""
    response = llm(
        prompt,
        max_tokens=500,
        temperature=0.7,
        top_p=0.9,
        echo=False
    )
    print(response['choices'][0]['text'])
    print("\n" + "="*50)
