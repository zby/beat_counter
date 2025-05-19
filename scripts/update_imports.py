#!/usr/bin/env python3
"""
Script to automatically update imports from beat_detection to beat_counter
and from web_app to beat_counter.web_app.
"""
import os
import re
from pathlib import Path

def update_file(file_path):
    """Update imports in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace beat_detection with beat_counter
    updated = re.sub(r'from beat_detection', 'from beat_counter', content)
    updated = re.sub(r'import beat_detection', 'import beat_counter', updated)
    
    # Replace root-level web_app imports
    updated = re.sub(r'from web_app', 'from beat_counter.web_app', updated)
    updated = re.sub(r'import web_app', 'import beat_counter.web_app', updated)
    
    # Better handling for patch statements with mixed quotes
    updated = re.sub(r"patch\(['\"]beat_detection([^'\"]*)['\"]", r"patch('beat_counter\1'", updated)
    updated = re.sub(r"patch\(['\"]web_app([^'\"]*)['\"]", r"patch('beat_counter.web_app\1'", updated)
    
    # Also update environment variable references
    updated = re.sub(r'BEAT_DETECTION_FORCE_CPU', 'BEAT_COUNTER_FORCE_CPU', updated)
    
    # Skip if no changes were made
    if content == updated:
        return False
    
    with open(file_path, 'w') as f:
        f.write(updated)
    
    return True

def find_py_files(directory):
    """Recursively find all Python files in a directory."""
    py_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files

def main():
    """Run the import updater."""
    base_dir = Path(__file__).parent.parent
    
    # Find all Python files in beat_counter directory
    py_files = find_py_files(base_dir / 'beat_counter')
    
    # Add tests directory
    if (base_dir / 'tests').exists():
        py_files.extend(find_py_files(base_dir / 'tests'))
    
    # Also update root Python files
    for root_py in base_dir.glob('*.py'):
        py_files.append(str(root_py))
    
    updated_count = 0
    for file_path in py_files:
        if update_file(file_path):
            print(f"Updated: {file_path}")
            updated_count += 1
    
    print(f"Updated {updated_count} files.")

if __name__ == "__main__":
    main() 