import markdown
import sys
import re

def convert():
    with open('paper_draft.md', 'r') as f:
        text = f.read()
    
    # Protect math blocks from markdown processing
    math_blocks = []
    
    def math_repl(match):
        math_blocks.append(match.group(0))
        return f"@@MATH_BLOCK_{len(math_blocks)-1}@@"
        
    text = re.sub(r'\$\$.*?\$\$', math_repl, text, flags=re.DOTALL)
    text = re.sub(r'\$.*?\$', math_repl, text)
    
    html = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    
    # Restore math blocks
    for i, block in enumerate(math_blocks):
        html = html.replace(f"@@MATH_BLOCK_{i}@@", block)
    
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Paper Draft</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; line-height: 1.6; padding: 2em; max-width: 800px; margin: 0 auto; }}
            h1, h2, h3 {{ border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }}
            th, td {{ border: 1px solid #dfe2e5; padding: 6px 13px; }}
            th {{ font-weight: 600; background-color: #f6f8fa; }}
            tr:nth-child(2n) {{ background-color: #f6f8fa; }}
            img {{ max-width: 100%; height: auto; display: block; margin: 1em auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        </style>
        <script>
            MathJax = {{
              tex: {{inlineMath: [['$', '$'], ['\\\\(', '\\\\)']]}}
            }};
        </script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
    </head>
    <body>
        {html}
    </body>
    </html>
    """
    
    with open('paper_draft.html', 'w') as f:
        f.write(full_html)
    
    print("Generated paper_draft.html")

if __name__ == '__main__':
    convert()
