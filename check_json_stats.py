import json
import os

file_path = 'E:/AI/result.json'

if not os.path.exists(file_path):
    print("result.json not found!")
    exit()

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total files processed: {len(data)}")
print("-" * 50)

for item in data:
    filename = item.get('file', 'Unknown')
    messages = item.get('messages', [])
    total_chars = sum(len(m.get('content', '')) for m in messages)
    # Rough token estimation: 1 token approx 3-4 chars for English, 1-2 chars for Chinese.
    # Let's take a conservative average of 2.5 chars per token for mixed.
    estimated_tokens = int(total_chars / 2.5) 
    
    print(f"File: {filename}")
    print(f"  - Conversation Turns: {len(messages)}")
    print(f"  - Total Characters: {total_chars}")
    print(f"  - Estimated Tokens (Rough): ~{estimated_tokens}")
    
    if "Roleplay" in filename:
        print("\n  [Preview of extracted content from Roleplay...]")
        # Find a long message to preview
        long_msgs = sorted(messages, key=lambda x: len(x['content']), reverse=True)
        if long_msgs:
            preview = long_msgs[0]['content'][:500].replace('\n', ' ')
            print(f"  Sample (Assistant/User): {preview}...")
    print("-" * 50)
