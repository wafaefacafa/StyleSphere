from bs4 import BeautifulSoup
import os

file_path = 'E:/AI/web/CD、TS、女装、男娘区别 _ Google AI Studio.html'

with open(file_path, 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f, 'html.parser')

# Find all strings
texts = soup.body.stripped_strings
count = 0
for t in texts:
    if len(t) > 20:
        print(f"--- Text found ({len(t)} chars) ---")
        print(t[:100])
        parent = soup.find(string=t).parent
        print(f"Parent: {parent.name}, Class: {parent.get('class')}")
        count += 1
        if count > 20: break
