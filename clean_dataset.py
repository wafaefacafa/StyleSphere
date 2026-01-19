import json
import os
import re

def clean_dataset():
    # Use the correct path structure
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "data", "chat_safe_clean.json")
    output_path = os.path.join(base_dir, "data", "chat_final_clean.json")
    
    print(f"Reading from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_count = 0
    # Pattern to extract text validation:
    # Look for the structure: ],["(TargetText)",null,[
    # Use DOTALL so . matches newlines
    pattern = re.compile(r"\],\[\"(.*?)\",null,\[", re.DOTALL) 
    
    for item in data:
        original = item["content"]
        # This check is good to keep
        if "prompts/" in original or "Enquirer Wang" in original:
            match = pattern.search(original)
            if match:
                clean_text = match.group(1)
                
                # Check for missed escapes. If the string contains literal \"
                # it means the regex stopped at ",null,[ but the text had valid quotes.
                # Since we match up to ",null,[, any internal quotes must be part of content.
                # However, if the text WAS a JSON string, quotes would be escaped as \" inside the extracted text?
                # No, json.load turns \" into " in memory.
                # So we have raw " in clean_text. This is fine.
                
                if clean_text != original:
                    item["content"] = clean_text
                    cleaned_count += 1
                    if cleaned_count <= 5:
                        print(f" Cleaned: {clean_text[:50]}...")
            else:
                 pass
        
        # Additional check: If it starts with "prompts/" but regex failed
        elif original.strip().startswith("prompts/"):
            print(f" Warning: Item starts with prompts/ but regex failed: {original[:50]}...")

    print(f"Writing to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print(f"\n Cleaned {cleaned_count} items.")
    print(f" Saved to {output_path}")

if __name__ == "__main__":
    clean_dataset()

