# 模型文件位置说明

## 推荐模型（优先使用）
当前默认使用的模型文件位置为:
```
E:/R1/TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_S.gguf
```

这是经过测试的对话模型，适合当前应用场景。文件大小约3.85GB，已优化过性能和内存占用。

## 可用的模型文件

在E:/R1/目录下有多个模型提供商的文件夹：

1. TheBloke （当前使用）
   - Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_S.gguf (3.85GB)

2. 其他模型源目录：
   - city96/
   - MaxedOut/
   - MaziyarPanahi/
   - OuteAI/
   - second-state/
   - tcpipuk/
   - vonjack/

注意：
1. 这些目录中可能包含其他GGUF格式的模型，但并非所有模型都适合当前应用
2. 一些目录中的模型可能是嵌入式模型(Embedding Models)或特殊用途模型，不适合对话场景
3. 建议优先使用上述推荐的模型文件，除非您明确知道其他模型的用途和兼容性

## 如何查找模型

1. 如果遇到模型找不到的错误，请按以下步骤检查：

   - 确认上述路径下是否存在模型文件
   - 确认文件名是否为 `llama-2-7b-chat.Q4_K_S.gguf`
   - 确认文件大小应约为 3.59 GB

2. 如果需要使用其他模型：

   - 可以修改 `test_model.py` 中的模型路径
   - 或创建一个环境变量 `LLAMA_MODEL_PATH` 指向您的模型文件

3. 获取模型的方式：

   - 从 HuggingFace 下载: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
   - 选择 Q4_K_S 版本的模型文件
   - 或使用其他兼容的 GGUF 格式模型

## 支持的模型格式

- 必须是 GGUF 格式的模型
- 推荐使用 Q4_K_S 量化版本以获得较好的性能与内存占用平衡
- 支持 Llama 系列模型
