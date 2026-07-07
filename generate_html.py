import markdown
import sys

def convert():
    with open('paper_draft.md', 'r') as f:
        text = f.read()
    
    html = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    
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
