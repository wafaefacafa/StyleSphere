import asyncio
import json
import re
from playwright.async_api import async_playwright

URLS = [
    "https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221hgfmILPIOJJ-lfVgylGy54fPmfeU8hNN%22%5D,%22action%22:%22open%22,%22userId%22:%22107397029029250892273%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing",
    "https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221znKN3uEghDt-f4WZFzPIkyj8c7OBy6Ja%22%5D,%22action%22:%22open%22,%22userId%22:%22107397029029250892273%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing",
    "https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221V45IuH4tn99Z348KZkQzLlD4JlsHOr1m%22%5D,%22action%22:%22open%22,%22userId%22:%22107397029029250892273%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing"
]

def extract_dialogue(text):
    """
    Heuristic to parse the text dump from the page into dialogue.
    Google AI Studio usually displays prompts as "Input" / "Output" or specific role headers.
    """
    messages = []
    lines = text.split('\n')
    current_role = None
    current_content = []
    
    # Heuristics for roles in the UI text dump
    # This might need adjustment based on valid output
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Check for role markers
        if re.match(r'^(user|input|prompt):', line, re.IGNORECASE):
            if current_role:
                messages.append({"role": current_role, "content": "\n".join(current_content)})
            current_role = "user"
            content = re.sub(r'^(user|input|prompt):\s*', '', line, flags=re.IGNORECASE)
            current_content = [content] if content else []
            continue
            
        if re.match(r'^(model|output|response):', line, re.IGNORECASE):
            if current_role:
                messages.append({"role": current_role, "content": "\n".join(current_content)})
            current_role = "assistant"
            content = re.sub(r'^(model|output|response):\s*', '', line, flags=re.IGNORECASE)
            current_content = [content] if content else []
            continue
            
        if current_role:
            current_content.append(line)
            
    if current_role and current_content:
        messages.append({"role": current_role, "content": "\n".join(current_content)})
        
    return messages

async def scrape_url(page, url):
    print(f"Navigating to {url}...")
    await page.goto(url)
    
    # Wait for the main content to load. 
    # AI Studio might take a while to hydrate.
    try:
        # Wait for some text to appear that indicates content
        # We'll wait for network idle to be safe
        await page.wait_for_load_state('networkidle', timeout=15000)
        # Give it a bit more time for JS rendering
        await asyncio.sleep(5)
    except Exception as e:
        print(f"Time out waiting for load: {e}")
        
    # Get all text
    # This is a crude method, but effective for a first pass
    # Better would be finding the specific container for the chat
    # Try to find the prompt-container or similar
    content = await page.evaluate("() => document.body.innerText")
    
    return content

async def main():
    results = []
    
    # Check for proxy in arguments
    proxy_server = None
    import sys
    for arg in sys.argv:
        if arg.startswith("--proxy="):
            proxy_server = arg.split("=")[1]
    
    launch_args = {"headless": True}
    if proxy_server:
        print(f"Using proxy: {proxy_server}")
        launch_args["proxy"] = {"server": proxy_server}

    async with async_playwright() as p:
        # Launch browser. Headless=False helps debugging if user watches, but True is faster.
        # User is in VS Code terminal, so True is safer unless they want to watch.
        # Let's use headless=True by default.
        browser = await p.chromium.launch(**launch_args)
        context = await browser.new_context()
        page = await context.new_page()
        
        for url in URLS:
            try:
                raw_text = await scrape_url(page, url)
                if raw_text:
                    # Depending on the raw text structure, we might need a parser
                    # For now, let's just save the raw text to see what we got
                    # And try to parse it
                    
                    # Try parsing
                    dialogue = extract_dialogue(raw_text)
                    
                    if dialogue:
                         # Convert to Alpaca
                        for i in range(len(dialogue)):
                            msg = dialogue[i]
                            if msg['role'] == 'assistant':
                                # Look back for user
                                input_text = ""
                                if i > 0 and dialogue[i-1]['role'] == 'user':
                                    input_text = dialogue[i-1]['content']
                                
                                results.append({
                                    "instruction": "You are a helpful AI assistant.",
                                    "input": input_text,
                                    "output": msg['content'],
                                    "source": url
                                })
                    else:
                        print(f"Warning: Could not extract dialogue structure from {url[-20:]}...")
                        # Fallback: Save raw for manual inspection if needed
                        results.append({
                            "raw_dump": raw_text,
                            "source": url,
                            "note": "Failed to parse dialogue structure"
                        })
                        
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                
        await browser.close()
        
    output_file = "data/scraped_aistudio.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"Done. Saved {len(results)} items to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
