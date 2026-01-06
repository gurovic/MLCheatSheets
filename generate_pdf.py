#!/usr/bin/env python3
"""
Script to generate a single PDF file containing all ML cheatsheets.
Uses Chrome headless to convert HTML to PDF, then merges all PDFs.
"""

import os
import glob
import subprocess
import tempfile
import shutil
from pathlib import Path
from PyPDF2 import PdfMerger

def find_chrome():
    """Find Chrome/Chromium executable."""
    chrome_paths = [
        '/usr/bin/google-chrome',
        '/usr/bin/chromium-browser',
        '/usr/bin/chromium',
        'google-chrome',
        'chromium-browser',
        'chromium'
    ]
    
    for path in chrome_paths:
        if shutil.which(path):
            return path
    
    return None

def create_cover_page_html(num_sheets):
    """Create a cover page HTML."""
    return f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>ML Cheatsheets - Cover</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
                color: #333;
                background: white;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }}
            
            .cover {{
                text-align: center;
                padding: 50px;
            }}
            
            .cover h1 {{
                font-size: 48px;
                color: #1a5fb4;
                margin-bottom: 20px;
            }}
            
            .cover h2 {{
                font-size: 24px;
                color: #666;
                margin-bottom: 10px;
            }}
            
            .cover p {{
                font-size: 16px;
                color: #888;
                margin: 5px 0;
            }}
            
            .count {{
                margin-top: 50px;
                font-size: 18px;
                font-weight: 600;
                color: #1a5fb4;
            }}
        </style>
    </head>
    <body>
        <div class="cover">
            <h1>ML Cheatsheets</h1>
            <h2>Полная коллекция шпаргалок по машинному обучению</h2>
            <p>Владимир Гуровиц (школа "Летово")</p>
            <p>DeepSeek, Github Copilot, Perplexity Comet</p>
            <p class="count">Всего шпаргалок: {num_sheets}</p>
        </div>
    </body>
    </html>
    """

def html_to_pdf_chrome(html_file, pdf_file, chrome_path):
    """Convert a single HTML file to PDF using Chrome headless."""
    try:
        cmd = [
            chrome_path,
            '--headless',
            '--disable-gpu',
            '--no-sandbox',
            '--print-to-pdf=' + str(pdf_file),
            '--print-to-pdf-no-header',
            str(html_file)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        return Path(pdf_file).exists()
        
    except Exception as e:
        print(f"    Error: {e}")
        return False

def create_combined_pdf():
    """Combine all HTML cheatsheets into a single PDF file."""
    
    # Get the directory paths
    script_dir = Path(__file__).parent
    cheatsheets_dir = script_dir / "cheatsheets"
    output_pdf = script_dir / "ML_Cheatsheets_Complete.pdf"
    
    # Find Chrome
    chrome_path = find_chrome()
    if not chrome_path:
        print("❌ Chrome/Chromium not found. Please install Chrome or Chromium.")
        return False
    
    print(f"Using Chrome: {chrome_path}")
    
    # Get all HTML files except template
    html_files = sorted(glob.glob(str(cheatsheets_dir / "*.html")))
    html_files = [f for f in html_files if not f.endswith("template.html")]
    
    print(f"Found {len(html_files)} cheatsheet files")
    
    # Create a temporary directory for individual PDFs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pdf_files = []
        
        # Create cover page
        print("\nCreating cover page...")
        cover_html = temp_path / "00_cover.html"
        cover_pdf = temp_path / "00_cover.pdf"
        cover_html.write_text(create_cover_page_html(len(html_files)), encoding='utf-8')
        
        if html_to_pdf_chrome(cover_html, cover_pdf, chrome_path):
            pdf_files.append(str(cover_pdf))
            print("  ✓ Cover page created")
        else:
            print("  ✗ Failed to create cover page")
        
        # Convert each HTML file to PDF
        print("\nConverting HTML files to PDF...")
        success_count = 0
        failed_files = []
        
        for i, html_file in enumerate(html_files, 1):
            try:
                filename = Path(html_file).name
                pdf_filename = filename.replace('.html', '.pdf')
                pdf_file = temp_path / pdf_filename
                
                print(f"  [{i}/{len(html_files)}] {filename}...", end='', flush=True)
                
                if html_to_pdf_chrome(html_file, pdf_file, chrome_path):
                    pdf_files.append(str(pdf_file))
                    success_count += 1
                    print(" ✓")
                else:
                    failed_files.append(filename)
                    print(" ✗")
                    
            except Exception as e:
                print(f" ✗ Error: {e}")
                failed_files.append(filename)
        
        print(f"\nSuccessfully converted: {success_count}/{len(html_files)}")
        if failed_files:
            print(f"Failed files ({len(failed_files)}): {', '.join(failed_files[:5])}{'...' if len(failed_files) > 5 else ''}")
        
        # Merge all PDFs
        print(f"\nMerging {len(pdf_files)} PDF files...")
        try:
            merger = PdfMerger()
            
            for pdf_file in pdf_files:
                if Path(pdf_file).exists():
                    merger.append(pdf_file)
            
            merger.write(str(output_pdf))
            merger.close()
            
            # Get file size
            file_size_mb = output_pdf.stat().st_size / (1024 * 1024)
            
            print(f"\n✅ PDF generated successfully!")
            print(f"   File: {output_pdf}")
            print(f"   Size: {file_size_mb:.2f} MB")
            print(f"   Total pages: {len(pdf_files)}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error merging PDFs: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = create_combined_pdf()
    exit(0 if success else 1)
