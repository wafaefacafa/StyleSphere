# Llama 3 8B 训练项目 (Windows + RTX 4060 Ti)

本项目实现了在 Windows 环境下，使用单张 RTX 4060 Ti (8GB) 对 Llama 3 8B 模型进行 4-bit QLoRA 微调。

## ⚠️ 关键环境配置 (踩坑总结)

在 Windows 上进行 QLoRA 训练最核心的难点是 `bitsandbytes` 库的兼容性。请务必遵循以下版本：

*   **OS**: Windows 10/11
*   **GPU**: NVIDIA RTX 4060 Ti (8GB VRAM)
*   **Python**: 3.10+ (推荐 Conda)
*   **CUDA**: 12.4
*   **PyTorch**: `2.6.0+cu124`
*   **BitsAndBytes**: `0.49.1` (必须支持 Windows 且兼容 CUDA 12.x)

### 常用安装命令备忘
如果是全新环境，请按此顺序安装，以避免 DLL 缺失或版本冲突错：

```powershell
# 1. 安装 PyTorch (带 CUDA 12.4 支持)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. 安装 Windows 版本的 BitsAndBytes (0.49.1 或更新)
# 不要直接 pip install bitsandbytes (那个通常是 Linux 版)
# 只要安装了这个正确版本，就不需要手动复制 DLL 了
pip install --upgrade bitsandbytes
```

## 🚀 如何开始 GPU 训练

### 1. 准备数据
训练数据位于 `data/train.json`。格式如下：
```json
[
  {
    "instruction": "用户的问题...",
    "input": "（可选）更多的上下文...",
    "output": "AI的理想回答..."
  }
]
```

### 2. 运行训练脚本
核心训练脚本是 `train_python_real.py`。该脚本已配置为自动使用 GPU + 4bit 量化。

在终端中运行：
```powershell
python e:\AI\llama2-train\train_python_real.py
```

### 3. 修改训练参数
如果需要调整训练时长，请编辑 `train_python_real.py`：

```python
# 修改这里来控制训练轮数
NUM_EPOCHS = 3     # 训练 3 轮 (约 20 分钟)
# MAX_STEPS = 100  # 或者使用固定步数
```

## 📊 训练输出

训练完成后，模型适配器（LoRA Adapters）会保存在：
`outputs/llama3_qlora_test/`

输出包含：
*   `adapter_model.bin` / `adapter_config.json`: 微调后的权重
*   训练日志

## � 如何导出与分享模型 (重要)

您训练好的模型本质上是一个 LoRA 适配器（Adapter），它无法独立运行，必须配合 Llama 3 原版模型使用。

### 方案 A：轻量级分享 (推荐 GitHub / HuggingFace) ✅

这是最标准的做法，因为 LoRA 文件非常小（约 160MB），适合上传。
**直接拷贝此文件夹分享即可**：
> `E:\AI\outputs\llama3_qlora_test`

**接收者如何使用：**
接收者需要在代码中加载 Base Model，然后使用 `PeftModel.from_pretrained(base_model, "你的文件夹路径")`。

---

### 方案 B：生成独立模型 (合并权重)

如果您希望得到一个在这个世界上“独立存在”的模型文件夹（不需要配置 LoRA，直接像用原版 Llama 一样用），可以使用合并脚本。

**注意**：这需要您电脑有 **16GB 以上的内存 (RAM)**。

1. 运行合并脚本：
   ```powershell
   python e:\AI\llama2-train\merge_weights.py
   ```
2. 产出目录：
   > `E:\AI\outputs\llama3_8b_custom_merged`
3. 这个文件夹现在是一个完整的、独立的 Llama 3 模型 (~15GB)，可以被任何支持 Transformers 的工具加载，也可以转换为 GGUF 给 Ollama 使用。

## �🛠️ 常见问题排错

1.  **报错: `ValueError: ... please install bitsandbytes >= 0.43.2`**
    *   **原因**: 安装了旧版 (0.41.1) 或者错误版本的 bitsandbytes。
    *   **解决**: 卸载旧版，安装 0.49.1+ 版本。

2.  **报错: `CUDA out of memory`**
    *   **原因**: 显存不足 (8GB 比较紧张)。
    *   **解决**:
        *   确保脚本中开启了 `load_in_4bit=True`
        *   减小 `per_device_train_batch_size` (目前已设为 1，最小了)
        *   减小 `max_length` (目前设为 512)

3.  **速度慢？**
    *   正常速度约为 **6秒/步** (RTX 4060 Ti)。如果远慢于此，检查是否误用了 CPU。脚本启动时会打印 `Using device: cuda` 以确认。