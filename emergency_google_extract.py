import re
import json
import os

def emergency_extract(input_file, output_file):
    print(f"⚡ 正在强力扫描文件: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # 1.3M 直接读入内存作为字符串，毫无压力
            raw_text = f.read()
    except Exception as e:
        print(f"读取失败: {e}")
        return

    # Google 消息特有的结构：["消息正文", null, ..., "角色"]
    # 我们用正则抓取：双引号开头，中间是一堆null，最后是 user 或 model
    # 这样可以绕过所有嵌套陷阱
    pattern = r'\["([\s\S]*?)",null,null,null,null,null,null,null,"(user|model)"'
    
    final_messages = []
    seen = set()

    print("🔎 开始特征匹配...")
    matches = re.finditer(pattern, raw_text)
    
    for match in matches:
        try:
            # group(1) captures the content inside the quotes
            raw_content = match.group(1)
            # JSON escape sequences need to be handled. 
            # Since we are regex matching the raw file content which includes escaped quotes like \"
            # We need to be careful. The regex `[\s\S]*?` is non-greedy, stopping at the first `,null...`
            # But what if the string contains `,null...`? Unlikely for this specific pattern.
            
            # 尝试使用 json.loads 解析字符串内容
            # 我们把匹配到的内容（不含两边的引号）放回引号中，构建一个合法的 JSON 字符串
            # 这样可以利用 json 标准库处理所有的转义字符（\u, \n, \" 等）
            json_str = f'"{match.group(1)}"'
            content = json.loads(json_str) 
        except Exception as e:
            # 如果构建 JSON 失败（比如内容里有未转义的换行等极端情况），尝试回退
            # 但通常 google 的响应里转义是标准的
            print(f"JSON解析警告: {e} - 尝试直接使用原始内容")
            content = match.group(1)
             
        # 去掉 Google 内部的一些特殊标签（如 thoughts 提示）
        content = content.replace("Expand to view model thoughts", "").strip()
        
        role_label = match.group(2)
        role = "user" if role_label == "user" else "assistant"
        
        # 简单过滤：内容不为空且不重复
        if content and content not in seen:
            final_messages.append({
                "role": role,
                "content": content
            })
            seen.add(content)

    print(f"✅ 提取完成！找到 {len(final_messages)} 条对话。")

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(final_messages, f_out, ensure_ascii=False, indent=4)
    print(f"💾 结果已保存至: {output_file}")

if __name__ == "__main__":
    # 确保文件名和你保存的一致
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, 'google_raw.json')
    output_path = os.path.join(base_dir, 'data', 'chat_safe_clean.json')
    
    emergency_extract(input_path, output_path)
