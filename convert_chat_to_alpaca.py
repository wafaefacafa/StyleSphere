import json
import os

def convert_to_alpaca():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, 'data', 'chat_final_clean.json')
    output_path = os.path.join(base_dir, 'data', 'train.json')
    
    print(f"Reading from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    alpaca_data = []
    
    # Simple pairing: User -> Assistant
    # We iterate and look for a user message, then take the immediate next assistant message.
    
    i = 0
    while i < len(data) - 1:
        current_msg = data[i]
        next_msg = data[i+1]
        
        if current_msg['role'] == 'user' and next_msg['role'] == 'assistant':
            entry = {
                "instruction": current_msg['content'],
                "input": "",
                "output": next_msg['content']
            }
            alpaca_data.append(entry)
            i += 2
        else:
            # Skip validly or handle weirdness
            # If we have User -> User, we might skip the first user text.
            # If Assistant -> User, we just move past the assistant.
            i += 1

    print(f"Generated {len(alpaca_data)} training pairs.")
    
    print(f"Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=4)
        
    print(f"✅ Saved to {output_path}")

if __name__ == "__main__":
    convert_to_alpaca()
