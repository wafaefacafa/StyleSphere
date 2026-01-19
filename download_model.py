from huggingface_hub import snapshot_download
import os

def main():
    print("开始下载模型...")
    
    # 创建models目录
    os.makedirs("models", exist_ok=True)
    
    # 下载GPTQ量化模型
    model_path = snapshot_download(
        repo_id="TheBloke/Llama-2-7B-Chat-GPTQ",
        local_dir="models/Llama-2-7B-Chat-GPTQ",
        local_dir_use_symlinks=False
    )
    
    print(f"模型下载完成，保存在: {model_path}")
    print("现在可以使用这个模型进行训练了。")

if __name__ == "__main__":
    main()
