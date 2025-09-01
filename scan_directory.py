import os
from pathlib import Path

def scan_directory(root_path):
    """扫描目录结构"""
    root_path = Path(root_path)
    print(f"\n开始扫描目录: {root_path}")
    print("="*50)
    
    def print_directory(path, level=0):
        indent = "  " * level
        try:
            # 打印当前目录下的所有文件和文件夹
            for item in sorted(path.iterdir()):
                if item.is_dir():
                    print(f"{indent}📁 {item.name}/")
                    print_directory(item, level + 1)
                else:
                    print(f"{indent}📄 {item.name}")
        except Exception as e:
            print(f"{indent}❌ 访问错误: {e}")
    
    print_directory(root_path)

if __name__ == "__main__":
    VAULT_PATH = r'E:\kubook'
    scan_directory(VAULT_PATH)
