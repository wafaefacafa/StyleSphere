from huggingface_hub import snapshot_download
import os
import ssl
import urllib3
import warnings
warnings.filterwarnings('ignore')

def main():
    print("开始下载模型...")
    
    # 禁用SSL验证
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # 创建models目录
    os.makedirs("models", exist_ok=True)
    
    # 下载GPTQ量化模型
    model_path = snapshot_download(
        repo_id="TheBloke/Llama-2-7B-Chat-GPTQ",
        local_dir="models/Llama-2-7B-Chat-GPTQ",
        local_dir_use_symlinks=False,
        verify=False
    )
    
    print(f"模型下载完成，保存在: {model_path}")
    print("现在可以使用这个模型进行训练了。")

if __name__ == "__main__":
    main()
