import html2text
import os

file_path = 'E:/AI/web/CD、TS、女装、男娘区别 _ Google AI Studio.html'
with open(file_path, 'r', encoding='utf-8') as f:
    html = f.read()

markdown = html2text.html2text(html)
with open('E:/AI/llama2-train/debug_markdown.md', 'w', encoding='utf-8') as f:
    f.write(markdown)
