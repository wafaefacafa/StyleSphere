import json
import os
import sys

def find_conversation_data(data):
    """
    Recursively search specifically for a 'link' key or common chat structures.
    The user mentioned 'identifying the long string of conversation in the LINK'.
    """
    if isinstance(data, dict):
        # Specific check for 'link' key as per user hint
        if 'link' in data:
            return data['link']
        
        # Check for other common conversation keys
        for key in ['conversations', 'messages', 'history', 'chat']:
            if key in data:
                return data[key]
        
        # Recursive search
        for key, value in data.items():
            result = find_conversation_data(value)
            if result:
                return result
    
    elif isinstance(data, list):
        # If it's a list, check if elements look like messages
        if len(data) > 0:
            if isinstance(data[0], dict) and ('role' in data[0] or 'from' in data[0]):
                return data
            # Or if it's a list of strings (raw text conversation)
            if isinstance(data[0], str):
                return data
            
        for item in data:
            result = find_conversation_data(item)
            if result:
                return result
                
    return None

def parse_text_transcript(text):
    """
    Parse a long string containing a transcript (e.g. User: ... AI: ...)
    Returns a list of dicts with role/content.
    """
    messages = []
    lines = text.split('\n')
    current_role = None
    current_content = []
    
    # Heuristics for role markers
    role_markers = {
        "User:": "user",
        "Human:": "user",
        "AI:": "assistant",
        "Assistant:": "assistant",
        "System:": "system",
        "Input:": "user",         # Google AI Studio common label
        "input:": "user",
        "Output:": "assistant",   # Google AI Studio common label
        "output:": "assistant",
        "Model:": "assistant",    # Google AI Studio common label
        "model:": "assistant"
    }
    
    for line in lines:
        matched_role = None
        for marker, role in role_markers.items():
            if line.strip().startswith(marker):
                matched_role = role
                # Remove marker from content
                line = line.strip()[len(marker):].strip()
                break
        
        if matched_role:
            if current_role:
                messages.append({"role": current_role, "content": "\n".join(current_content)})
            current_role = matched_role
            current_content = [line]
        else:
            if current_role:
                current_content.append(line)
                
    if current_role and current_content:
        messages.append({"role": current_role, "content": "\n".join(current_content)})
        
    return messages

def process_messages(messages, output_file, max_context=2048):
    """
    Convert a list of messages into training pairs (Instruction/Input/Output).
    Handles sliding window for context if needed, or simple pairs.
    """
    processed_data = []

    # Handle if messages is a single long string
    if isinstance(messages, str):
        if messages.startswith('http'):
            print(f"WARNING: The found data is a URL: {messages}")
            print("Please download the content of this URL to a file first, or ensure the JSON contains the conversation text.")
            return
        
        print("Detected raw string data. Attempting to parse as transcript...")
        messages = parse_text_transcript(messages)
    
    # Check format of messages
    if not messages:
        return
        
    is_dict = isinstance(messages[0], dict)
    is_str = isinstance(messages[0], str)
    
    history = []
    
    for i, msg in enumerate(messages):
        role = ""
        content = ""
        
        if is_dict:
            # Handle standard formats
            role = msg.get('role', msg.get('from', ''))
            content = msg.get('content', msg.get('value', ''))
        elif is_str:
            # Handle list of strings (assuming alternating User/AI or explicit prefixes)
            content = msg
            # Heuristic: Odd/Even or check prefixes like "User:", "AI:"
            if i % 2 == 0:
                role = "user"
            else:
                role = "assistant"
        
        # Normalize role
        if role.lower() in ['user', 'human', 'input']:
            role = 'user'
        elif role.lower() in ['assistant', 'ai', 'model', 'gpt']:
            role = 'assistant'
            
        if role == 'user':
            history.append({"role": "user", "content": content})
        elif role == 'assistant':
            # Create a training example: 
            # Instruction: System/Context
            # Input: The last user message (and history?)
            # Output: This assistant message
            
            if history:
                last_user = history[-1]
                if last_user['role'] == 'user':
                    # Simple pair
                    entry = {
                        "instruction": "You are a helpful AI assistant. Continue the conversation.",
                        "input": last_user['content'],
                        "output": content,
                        "history": history[:-1] # Optional: keep history for sophisticated loaders
                    }
                    
                    # If we want to burn history into Input (Context Window method)
                    # This constructs a multi-turn promtp matching Llama 2 format if possible
                    context_str = ""
                    for h in history[:-1]:
                        context_str += f"{h['role']}: {h['content']}\n"
                    
                    if context_str:
                        entry['instruction'] += f"\n\nContext:\n{context_str}"
                    
                    processed_data.append(entry)
                
            history.append({"role": "assistant", "content": content})

    # Save to output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(messages)} messages into {len(processed_data)} training examples.")
    print(f"Saved to {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_conversation.py <input_file>")
        return

    input_file = sys.argv[1]
    output_file = "data/converted_raw_chat.json"
    
    print(f"Reading {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # First try parsing as JSON
            try:
                data = json.load(f)
                conversation = find_conversation_data(data)
                if conversation:
                    print(f"Found structure conversation with {len(conversation)} items.")
                    process_messages(conversation, output_file)
                    return
            except json.JSONDecodeError:
                f.seek(0)
                # If not JSON, read as whole string
                data = f.read()
                print("Input is not valid JSON. Treating as raw text transcript...")
                messages = parse_text_transcript(data)
                if messages:
                    print(f"Parsed {len(messages)} messages from text.")
                    process_messages(messages, output_file)
                else:
                    print("Could not parse any messages from text. Make sure lines start with 'User:', 'AI:', 'Input:', 'Model:' etc.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
