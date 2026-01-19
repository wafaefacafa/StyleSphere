import os
import json
import re
from bs4 import BeautifulSoup
import html2text

# Configuration
INPUT_DIR = "web" # Folder containing the HTML files
OUTPUT_FILE = "result.json"

def clean_content(text):
    """
    Clean up UI artifacts from the markdown.
    """
    lines = text.split('\n')
    cleaned = []
    skip_phrases = [
        "expand to view model thoughts",
        "chevron_right",
        "edit",
        "more_vert",
        "sharecompare_arrowsadd",
        "auto",
        "thoughts", 
        "![Thinking]", 
        "![Enquirer Wang]"
    ]
    
    for line in lines:
        l_lower = line.strip().lower()
        if not l_lower:
            cleaned.append(line) # Keep empty lines for structure
            continue
            
        is_artifact = False
        for phrase in skip_phrases:
            if l_lower == phrase:
                is_artifact = True
                break
        
        # Remove lines that are just icons or UI distinct markers
        if l_lower in ["edit", "copy", "share", "good", "bad", "refresh"]:
             is_artifact = True

        if not is_artifact:
            # Remove image links if they are just icons
            if re.match(r'^!\[.*\]\(.*\)$', line.strip()):
                if "watermark.png" in line or "unnamed.png" in line:
                    continue
            cleaned.append(line)
            
    return "\n".join(cleaned).strip()

def parse_html_file(file_path):
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()
    
    # Use BeautifulSoup to verify it's valid HTML and maybe get the title
    soup = BeautifulSoup(html_content, 'html.parser')
    title = soup.title.string if soup.title else "Unknown"
    
    # Use html2text to convert to Markdown (preserving structure)
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.body_width = 0 # No wrapping
    markdown_text = h.handle(html_content)
    
    # Parse the markdown to identify roles
    # Heuristic: Split by "\nUser" and "\nModel" lines
    
    # Normalize newlines
    markdown_text = markdown_text.replace("\r\n", "\n")
    
    # We look for the pattern:
    # (Start) -> Content -> Model -> Content -> User -> Content -> ...
    
    # First, split by likely headers.
    # The simple regex `\n(User|Model)\n` might miss markers if the HTML conversion didn't put newlines exactly right.
    # Let's try a more robust split that allows for some whitespace and optional bolding.
    
    # Use re.split with a capture group to keep the delimiters
    # Matches: Newline, optional markdown bold, "User" or "Model", optional markdown bold, Newline
    # We use (?i) for case insensitivity just in case
    # We use [ \t]* to allow spaces/tabs on the same line
    
    pattern = r'\n\s*(?:\*\*)?(User|Model)(?:\*\*)?\s*\n'
    tokens = re.split(pattern, "\n" + markdown_text, flags=re.IGNORECASE)
    
     # tokens[0] is initial content
    # tokens[1] is Role, tokens[2] is Content, tokens[3] is Role, tokens[4] is Content...
    
    messages = []
    
    # Heuristic: If the first token isn't a role, it's likely the First User Prompt (Title/Context)
    # But our regex `\n(User|Model)\n` splits strictly.
    # If the file starts with text then "Model", the first chunk goes to tokens[0].
    
    first_chunk = tokens[0].strip()
    if first_chunk:
        # Check if we should ignore navbar stuff
        # AI Studio exports often have a navbar at the top.
        # Clean up UI noise: Look for the Title line (# Title)
        lines = first_chunk.split('\n')
        start_idx = 0
        for idx, line in enumerate(lines):
            # The title usually starts with #
            if line.strip().startswith('# '):
                start_idx = idx
                break
            # Or if we see the user account, the next line might be content
            if '@' in line and '.com' in line:
                start_idx = idx + 1
        
        cleaned_first = clean_content("\n".join(lines[start_idx:]))
        # Remove common noise if still present
        cleaned_first = re.sub(r'\d+,\d+\s+tokens', '', cleaned_first)
        
        if cleaned_first and len(cleaned_first) < 20000: # Sanity check length
             messages.append({"role": "user", "content": cleaned_first})
    
    for i in range(1, len(tokens), 2):
        role_marker = tokens[i].strip()
        content = tokens[i+1] if i+1 < len(tokens) else ""
        
        role = "assistant" if role_marker == "Model" else "user"
        cleaned_text = clean_content(content)
        
        if cleaned_text:
            messages.append({"role": role, "content": cleaned_text})
            
    return {"file": os.path.basename(file_path), "title": title, "messages": messages}

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Directory {INPUT_DIR} not found.")
        return

    all_data = []
    
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".html"):
                full_path = os.path.join(root, file)
                result = parse_html_file(full_path)
                if result['messages']:
                    all_data.append(result)
    
    # Save proper JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully processed {len(all_data)} files.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
