#!/usr/bin/env python3
"""
Script to identify and fix bare except Exception clauses.

This script scans Python files for bare `except Exception:` clauses and
helps replace them with more specific exception handling.

Usage:
    python scripts/fix_bare_except.py [--dry-run] [--file PATH]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple


class ExceptLocation(NamedTuple):
    file: str
    line: int
    column: int
    context: str


def find_bare_except_files(root_dir: Path) -> list[Path]:
    """Find all Python files with bare except clauses."""
    files_with_bare_except = []
    
    for py_file in root_dir.rglob("*.py"):
        # Skip certain directories
        if any(part in str(py_file) for part in ('venv', '__pycache__', '.git', 'node_modules')):
            continue
        
        try:
            with open(py_file, encoding='utf-8') as f:
                content = f.read()
            
            # Quick regex check for bare except Exception
            if re.search(r'except\s+Exception\s*:', content):
                files_with_bare_except.append(py_file)
        except (OSError, UnicodeDecodeError):
            continue
    
    return files_with_bare_except


def analyze_file(file_path: Path) -> list[ExceptLocation]:
    """Analyze a file for bare except clauses."""
    locations = []
    
    try:
        with open(file_path, encoding='utf-8') as f:
            lines = f.readlines()
        
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check for bare except Exception
            if re.match(r'except\s+Exception\s*:', stripped):
                # Get context (next few lines)
                context_lines = lines[i:min(i+5, len(lines))]
                context = ''.join(context_lines).strip()
                
                # Check if it's just logging or has proper handling
                if 'pass' in context or ('log' not in context.lower() and 'raise' not in context):
                    locations.append(ExceptLocation(
                        file=str(file_path),
                        line=i,
                        column=0,
                        context=context[:200]
                    ))
    except (OSError, SyntaxError, UnicodeDecodeError):
        pass
    
    return locations


def generate_fix_suggestions(locations: list[ExceptLocation]) -> str:
    """Generate fix suggestions for found issues."""
    if not locations:
        return "No critical bare except clauses found!"
    
    output = []
    output.append(f"Found {len(locations)} bare except clauses that need attention:\n")
    
    # Group by file
    by_file: dict[str, list[ExceptLocation]] = {}
    for loc in locations:
        if loc.file not in by_file:
            by_file[loc.file] = []
        by_file[loc.file].append(loc)
    
    for file_path, file_locations in sorted(by_file.items()):
        output.append(f"\n{file_path}:")
        for loc in file_locations[:10]:  # Show first 10 per file
            output.append(f"  Line {loc.line}:")
            output.append(f"    Context: {loc.context[:100]}...")
            output.append("    Fix: Replace with specific exceptions + logging + re-raise")
        if len(file_locations) > 10:
            output.append(f"  ... and {len(file_locations) - 10} more")
    
    return '\n'.join(output)


def fix_file(file_path: Path, dry_run: bool = True) -> int:
    """Fix bare except clauses in a file.
    
    Returns number of fixes applied.
    """
    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()
        
        fixes_applied = 0
        
        # Pattern 1: except Exception: pass -> log and continue
        pattern1 = re.compile(
            r'(except\s+Exception\s*:\s*\n\s*)pass\b',
            re.MULTILINE
        )
        
        def replace_pass(match):
            nonlocal fixes_applied
            fixes_applied += 1
            indent = match.group(1)
            return f"{indent}log.warning(\"Operation failed silently\")\n{indent}continue  # or return/raise as appropriate"
        
        content = pattern1.sub(replace_pass, content)
        
        # Pattern 2: except Exception: without proper handling
        # This is more complex and requires manual review
        
        if not dry_run and fixes_applied > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return fixes_applied
    
    except (OSError, SyntaxError, UnicodeDecodeError):
        return 0


def main():
    root = Path(__file__).parent.parent
    
    print("Scanning for bare except Exception clauses...")
    files = find_bare_except_files(root)
    print(f"Found {len(files)} files with bare except clauses")
    
    # Analyze critical issues
    all_locations = []
    for file_path in files[:20]:  # Analyze first 20 files
        locations = analyze_file(file_path)
        all_locations.extend(locations)
    
    print("\n" + generate_fix_suggestions(all_locations))
    
    # Show fix guidance
    print("\n" + "="*70)
    print("FIX GUIDANCE:")
    print("="*70)
    print("""
For each bare `except Exception:`, replace with:

1. SPECIFIC EXCEPTIONS:
   Instead of:
       except Exception:
           pass
   
   Use:
       except (OSError, ValueError) as e:
           log.warning("Operation failed: %s", e)
           return None

2. LOGGING + RE-RAISE:
   Instead of:
       except Exception:
           log.error("Failed")
   
   Use:
       except Exception as e:
           log.error("Operation failed: %s", e)
           raise  # Re-raise to caller

3. CONTEXT-SPECIFIC:
   For data loading:
       except (OSError, json.JSONDecodeError) as e:
           log.warning("Failed to load %s: %s", path, e)
           return default_value
   
   For network operations:
       except (requests.RequestException, socket.error) as e:
           log.error("Network error: %s", e)
           raise ConnectionError(f"Failed to fetch: {e}") from e

CRITICAL FILES TO FIX FIRST:
1. utils/security.py - Security-critical operations
2. trading/oms.py - Order management
3. trading/risk.py - Risk controls
4. data/fetcher*.py - Data integrity
5. config/settings.py - Configuration
""")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
