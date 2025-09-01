import sys
import json

def print_file_content(file_path):
    """打印文件的前500个字符"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"\n文件: {file_path}")
            print("="*50)
            print("前500个字符:")
            print(content[:500])
            print("="*50)
            if file_path.endswith('.json'):
                try:
                    json_data = json.loads(content)
                    print("\nJSON结构:")
                    print_json_structure(json_data)
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
    except Exception as e:
        print(f"读取文件错误: {e}")

def print_json_structure(obj, level=0):
    """递归打印JSON结构"""
    indent = "  " * level
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                print(f"{indent}{key}:")
                print_json_structure(value, level + 1)
            else:
                print(f"{indent}{key}: {type(value).__name__}")
    elif isinstance(obj, list):
        if obj:
            print(f"{indent}Array[{len(obj)}]:")
            print_json_structure(obj[0], level + 1)
        else:
            print(f"{indent}Empty Array")

if __name__ == "__main__":
    files = [
        r'E:\kubook\塞尔达传说王国之泪\塞尔达.md',
        r'E:\kubook\塞尔达传说王国之泪\林克.md',
        r'E:\kubook\bilibili\json\search_contents_2025-08-23.json'
    ]
    
    for file_path in files:
        print_file_content(file_path)
