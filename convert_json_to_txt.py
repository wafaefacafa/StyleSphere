import json
import os

def format_prompt_llama3(instruction, input_text, output):
    """Format the data using Llama 3 style."""
    system_prompt = "You are a helpful assistant."
    user_content = f"{instruction}\n{input_text}" if input_text else instruction
    
    prompt = (
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
    )
    return prompt

def convert():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, 'data', 'train.json')
    output_path = os.path.join(base_dir, 'data', 'train.txt')
    
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            formatted_text = format_prompt_llama3(
                item.get('instruction', ''),
                item.get('input', ''),
                item.get('output', '')
            )
            # Replace dictionary newlines with escaped versions if needed, 
            # but usually 'finetune' takes raw text. 
            # However, for separation, we often append a newline or separator.
            # Llama.cpp's finetune example often trains on a raw text stream.
            f.write(formatted_text + "\n")
            
    print(f"✅ Converted {len(data)} items to {output_path}")

if __name__ == "__main__":
    convert()
