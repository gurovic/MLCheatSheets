#!/usr/bin/env python3
"""
Fix HTML cheatsheet issues:
1. Remove consecutive duplicate images
2. Remove duplicate images across files
3. Fix misplaced closing div tags
"""

import os
import re
import glob
from collections import defaultdict
import hashlib


def fix_consecutive_duplicate_images(html_content):
    """Remove images that appear twice in a row."""
    # Find all img tags with their positions
    img_pattern = r'<img[^>]*>'
    matches = list(re.finditer(img_pattern, html_content))
    
    if not matches:
        return html_content
    
    # Identify positions of duplicates to remove
    positions_to_remove = []
    prev_img = None
    prev_end = 0
    
    for i, match in enumerate(matches):
        current_img = match.group()
        current_start = match.start()
        
        # Check if there's only whitespace between previous and current
        if prev_img is not None:
            between = html_content[prev_end:current_start].strip()
            if between == '' and current_img == prev_img:
                # This is a consecutive duplicate
                positions_to_remove.append((prev_end, match.end()))
                print(f"  Removing consecutive duplicate image: {current_img[:80]}...")
        
        prev_img = current_img
        prev_end = match.end()
    
    # Remove duplicates in reverse order to maintain positions
    for start, end in reversed(positions_to_remove):
        html_content = html_content[:start] + html_content[end:]
    
    return html_content


def fix_misplaced_closing_divs(html_content):
    """
    Fix closing divs that appear at the end of the document instead of
    before the next .block section.
    
    The issue is that closing </div> for container appears right before </body></html>
    instead of after the last .block div.
    """
    lines = html_content.split('\n')
    
    # Find the pattern: closing div near the end, right before </body>
    # We need to move it to after the last .block closing div
    
    # Find the last .block closing div
    last_block_div_idx = -1
    container_close_idx = -1
    
    for i in range(len(lines) - 1, -1, -1):
        # Look for standalone </div> near the end
        if '</div>' == lines[i].strip() and container_close_idx == -1:
            # Check if this is isolated (not part of a block)
            if i < len(lines) - 5:  # Not at the very end
                continue
            # Check if previous lines have </body> or </html>
            context = ''.join(lines[max(0, i-2):min(len(lines), i+3)])
            if '</body>' in context or '</html>' in context:
                container_close_idx = i
                
        # Find the last </div> that closes a .block
        if '</div>' in lines[i] and 'block' in ''.join(lines[max(0, i-20):i]):
            if last_block_div_idx == -1:
                last_block_div_idx = i
                
        if container_close_idx != -1 and last_block_div_idx != -1:
            break
    
    # If we found a misplaced closing div
    if container_close_idx != -1 and last_block_div_idx != -1 and container_close_idx > last_block_div_idx + 5:
        print(f"  Moving closing div from line {container_close_idx} to after line {last_block_div_idx}")
        
        # Remove the misplaced closing div
        closing_div_line = lines[container_close_idx]
        lines.pop(container_close_idx)
        
        # Insert it after the last block div
        # Find the right indentation
        indent = '  '  # Default indent
        if last_block_div_idx < len(lines):
            # Match the indentation of the line after last block
            match = re.match(r'^(\s*)', lines[last_block_div_idx])
            if match:
                indent = match.group(1)
        
        # Insert with a blank line before
        lines.insert(last_block_div_idx + 1, '')
        lines.insert(last_block_div_idx + 2, closing_div_line.strip())
        
        return '\n'.join(lines)
    
    return html_content


def process_file(filepath):
    """Process a single HTML file to fix all issues."""
    print(f"\nProcessing: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix consecutive duplicate images
    content = fix_consecutive_duplicate_images(content)
    
    # Fix misplaced closing divs
    content = fix_misplaced_closing_divs(content)
    
    # Only write if changes were made
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ File updated")
        return True
    else:
        print(f"  No changes needed")
        return False


def main():
    """Process all HTML files in the cheatsheets directory."""
    print("=" * 80)
    print("Fixing HTML cheatsheet issues")
    print("=" * 80)
    
    html_files = sorted(glob.glob('cheatsheets/*.html'))
    print(f"\nFound {len(html_files)} HTML files to process")
    
    files_modified = 0
    for filepath in html_files:
        if process_file(filepath):
            files_modified += 1
    
    print("\n" + "=" * 80)
    print(f"Summary: Modified {files_modified} out of {len(html_files)} files")
    print("=" * 80)


if __name__ == '__main__':
    main()
