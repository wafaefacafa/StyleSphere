import json
import os
import sys
import requests
from bs4 import BeautifulSoup
import re
import argparse

def extract_url_from_json(file_path):
    """
    Try to find a URL in a JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Recursive search for a string that looks like a URL
        def find_url(obj):
            if isinstance(obj, str):
                if obj.strip().startswith('http://') or obj.strip().startswith('https://'):
                    return obj.strip()
            elif isinstance(obj, dict):
                # Prioritize keys named 'link', 'url'
                for key in ['link', 'url', 'share_link', 'source']:
                    if key in obj:
                        res = find_url(obj[key])
                        if res: return res
                
                # Search all values
                for v in obj.values():
                    res = find_url(v)
                    if res: return res
            elif isinstance(obj, list):
                for item in obj:
                    res = find_url(item)
                    if res: return res
            return None

        return find_url(data)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None

def fetch_content_from_url(url):
    """
    Fetch the content of the URL.
    Returns the parsed text or structured data if found.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    print(f"Fetching URL: {url}...")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return None

def parse_chatgpt_next_data(html_content):
    """
    Attempt to find __NEXT_DATA__ script tag which often contains the structured chat data for Next.js apps (like ChatGPT/ShareGPT).
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    script = soup.find('script', id='__NEXT_DATA__')
    if script:
        try:
            data = json.loads(script.string)
            # Traverse to find meaningful conversation structure
            # This is highly specific to the site structure, but we can try generic keys
            
            # Helper to find list of messages
            def find_messages(obj):
                if isinstance(obj, dict):
                    if 'mapping' in obj and isinstance(obj['mapping'], dict):
                        # ChatGPT structure often uses 'mapping'
                        return list(obj['mapping'].values())
                    if 'messages' in obj and isinstance(obj['messages'], list):
                        return obj['messages']
                    for v in obj.values():
                        res = find_messages(v)
                        if res: return res
                elif isinstance(obj, list):
                    for item in obj:
                        res = find_messages(item)
                        if res: return res
                return None
                
            messages_raw = find_messages(data)
            if messages_raw:
                extracted = []
                for m in messages_raw:
                    # Generic extraction for ChatGPT-like structure
                    if isinstance(m, dict):
                        msg_obj = m.get('message', {})
                        if not msg_obj: continue # Skip if no message content
                        
                        author = msg_obj.get('author', {}).get('role', 'unknown')
                        content_parts = msg_obj.get('content', {}).get('parts', [])
                        text = "".join([str(p) for p in content_parts if isinstance(p, str)])
                        
                        if text:
                            extracted.append({'role': author, 'content': text})
                return extracted
                
        except Exception as e:
            print(f"Error parsing __NEXT_DATA__: {e}")
            
    return None

def parse_html_text_heuristics(html_content):
    """
    Fallback: Extract text and try to split by roles.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()
        
    text = soup.get_text(separator='\n')
    
    # Try to clean up empty lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    messages = []
    current_role = None
    current_content = []
    
    # Heuristics for common chat interfaces
    role_patterns = {
        r'^(User|Human|You):?': 'user',
        r'^(ChatGPT|Assistant|AI|Model):?': 'assistant'
    }
    
    for line in lines:
        matched = False
        for pattern, role in role_patterns.items():
            if re.match(pattern, line, re.IGNORECASE):
                if current_role:
                    messages.append({"role": current_role, "content": "\n".join(current_content)})
                current_role = role
                # Remove the label from the line
                content_start = re.match(pattern, line, re.IGNORECASE).end()
                line_content = line[content_start:].strip()
                current_content = [line_content] if line_content else []
                matched = True
                break
        
        if not matched:
            if current_role:
                current_content.append(line)
    
    if current_role and current_content:
        messages.append({"role": current_role, "content": "\n".join(current_content)})
        
    return messages

def convert_to_alpaca(messages):
    """
    Convert list of {role, content} to Alpaca format.
    """
    training_data = []
    history = []
    
    for msg in messages:
        role = msg.get('role', '').lower()
        content = msg.get('content', '')
        
        if role in ['user', 'human']:
            history.append(content)
        elif role in ['assistant', 'ai', 'model', 'system'] and role != 'system':
            # Create training pair
            if history:
                last_input = history[-1]
                # Simple single-turn context for now, or could include history
                entry = {
                    "instruction": "You are a helpful AI assistant.",
                    "input": last_input,
                    "output": content
                }
                training_data.append(entry)
                # Clear history or keep it? Standard Alpaca is usually 1-turn or explicit history in input.
                # Let's clean the user history after use to assume pairs. 
                # If there were multiple user messages in a row, we only use the last one?
                # A robust converter handles this better, but this is a start.
            
    return training_data

def main():
    parser = argparse.ArgumentParser(description="Extract conversation from a Link/JSON and convert to training data.")
    parser.add_argument("input", help="URL or Path to JSON file containing a link")
    parser.add_argument("--output", default="data/url_train.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    url = args.input
    
    # Check if input is a file
    if os.path.exists(args.input):
        print(f"Reading input file: {args.input}")
        url = extract_url_from_json(args.input)
        if not url:
            print("No URL found in the provided JSON file.")
            sys.exit(1)
        print(f"Found URL: {url}")
    
    if not url.startswith('http'):
        print("Input must be a valid URL starting with http/https")
        sys.exit(1)
        
    html = fetch_content_from_url(url)
    if not html:
        sys.exit(1)
        
    print("Parsing content...")
    # Try structured extraction first
    messages = parse_chatgpt_next_data(html)
    
    if not messages:
        print("Could not find structured data (__NEXT_DATA__). Attempting text heuristic parsing...")
        messages = parse_html_text_heuristics(html)
        
    if not messages:
        print("Failed to extract any conversation messages.")
        sys.exit(1)
        
    print(f"Extracted {len(messages)} messages.")
    
    training_data = convert_to_alpaca(messages)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully saved {len(training_data)} training examples to {args.output}")

if __name__ == "__main__":
    main()
