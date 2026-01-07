#!/usr/bin/env python3
"""
Prepare David Lawrence Coins for Mint Mark Classification Training

Processes all davidlawrence_coins* folders and organizes images by mint mark:
- Extracts mint mark from description (e.g., "1914-D" → D, "1886" → None)
- Handles all US mint marks including historical mints:
  - None: Philadelphia (no mint mark on coin)
  - P: Philadelphia (explicit mark, used in some years)
  - D: Denver (1906-present)
  - DL: Dahlonega (1838-1861) - automatically detected by year
  - S: San Francisco
  - O: New Orleans
  - CC: Carson City
  - C: Charlotte (1838-1861)
  - W: West Point
- Year is included in filenames for training/disambiguation

Organizes into:
  - mintmark_dataset/<mintmark>/obverse/
  - mintmark_dataset/<mintmark>/reverse/

Usage:
  # Test with first 10 coins from each folder (dry run)
  python prepare_mintmark_dataset.py --test 10 --dry-run
  
  # Process all coins from all davidlawrence_coins* folders
  python prepare_mintmark_dataset.py --all
"""

import json
import shutil
import argparse
import glob
import re
from pathlib import Path
from collections import defaultdict


# Valid US Mint marks
# Note: D before 1862 is Dahlonega (DL), D from 1906+ is Denver
# C (1838-1861) is Charlotte
VALID_MINT_MARKS = {'P', 'D', 'DL', 'S', 'O', 'CC', 'C', 'W', 'None'}

# Year ranges for mint disambiguation
DAHLONEGA_YEARS = range(1838, 1862)  # 1838-1861
DENVER_START_YEAR = 1906
CHARLOTTE_YEARS = range(1838, 1862)  # 1838-1861


def extract_mint_mark(description, year=None):
    """
    Extract mint mark from coin description.
    
    Patterns:
      - "1914-D 5c PCGS MS62" → D (Denver, post-1906)
      - "1850-D $1 PCGS AU55" → DL (Dahlonega, 1838-1861)
      - "1849-C $1 PCGS VF35" → C (Charlotte, 1838-1861)
      - "1883-CC $1 PCGS MS65+" → CC (Carson City)
      - "1881-S $1 PCGS MS63DMPL" → S (San Francisco)
      - "1886 $1 PCGS MS64" → None (Philadelphia, no mark)
      - "1942-P 5c NGC MS66" → P (explicit Philadelphia)
    
    Args:
        description: The coin description string
        year: Optional year string to help parse
        
    Returns:
        tuple: (mint_mark, extracted_year) where mint_mark is P, D, DL, S, O, CC, C, W, 
               or 'None' for no mint mark. Returns (None, None) if cannot parse.
    """
    if not description:
        return None, None
    
    # Clean up description
    desc = description.strip()
    
    # Common pattern: YEAR-MINTMARK at the start (or after "Hit List VAM:" prefix)
    # Examples: "1914-D 5c", "1883-CC $1", "1881-S $1", "1942-P 5c"
    
    # Remove common prefixes
    desc = re.sub(r'^Hit List VAM:\s*', '', desc)
    
    # Pattern to match year with optional mint mark
    # Handles: 1914-D, 1883-CC, 1880/9-S (overdate with mintmark), 1878 7/8TF
    # The mint mark comes after the year and a hyphen
    # Added C for Charlotte
    
    extracted_year = None
    raw_mint_mark = None
    
    # First try to match patterns with overdates like "1880/9-S"
    overdate_pattern = r'^(\d{4})/\d+-([PDSOWTC]|CC)\b'
    match = re.search(overdate_pattern, desc)
    if match:
        extracted_year = int(match.group(1))
        raw_mint_mark = match.group(2)
    
    # Standard pattern: YEAR-MINTMARK
    if not raw_mint_mark:
        standard_pattern = r'^(\d{4})-([PDSOWTC]|CC)\b'
        match = re.search(standard_pattern, desc)
        if match:
            extracted_year = int(match.group(1))
            raw_mint_mark = match.group(2)
    
    # Check if year is at start without mint mark (Philadelphia)
    if not raw_mint_mark:
        # Pattern: starts with 4-digit year followed by space or other pattern (not hyphen-letter)
        no_mint_pattern = r'^(\d{4})(?:\s|/|\s+\d)'
        match = re.search(no_mint_pattern, desc)
        if match:
            extracted_year = int(match.group(1))
            # Verify there's no mint mark right after
            year_end = match.end(1)
            rest = desc[year_end:]
            # If next char is hyphen followed by mint mark letter, we should have caught it above
            # Otherwise it's Philadelphia (no mint mark)
            if not rest.startswith('-'):
                raw_mint_mark = 'None'
            # Check if it's just a hyphen followed by denomination or other non-mintmark
            elif rest.startswith('-') and len(rest) > 1:
                next_char = rest[1]
                if next_char not in 'PDSOWTC':
                    raw_mint_mark = 'None'
    
    # Also check for year at start with immediate space (like "1886 $1")
    if not raw_mint_mark:
        simple_pattern = r'^(\d{4})\s+(?:\$|[0-9]|[a-z])'
        match = re.search(simple_pattern, desc, re.IGNORECASE)
        if match:
            extracted_year = int(match.group(1))
            raw_mint_mark = 'None'
    
    # If we got here with a year provided but no mint mark extracted, check again
    if not raw_mint_mark and year:
        year_str = str(year)
        if desc.startswith(year_str):
            extracted_year = int(year)
            # Check what comes after the year
            rest = desc[len(year_str):]
            if rest.startswith('-'):
                # Extract potential mint mark
                mm_match = re.match(r'-([PDSOWTC]|CC)', rest)
                if mm_match:
                    raw_mint_mark = mm_match.group(1)
                else:
                    raw_mint_mark = 'None'
            else:
                raw_mint_mark = 'None'
    
    # Use provided year if we couldn't extract one
    if extracted_year is None and year:
        try:
            extracted_year = int(year)
        except (ValueError, TypeError):
            pass
    
    # Disambiguate D → DL for Dahlonega (1838-1861)
    if raw_mint_mark == 'D' and extracted_year:
        if extracted_year in DAHLONEGA_YEARS:
            raw_mint_mark = 'DL'  # Dahlonega
        # else it's Denver (1906+)
    
    return raw_mint_mark, extracted_year


def find_source_directories(pattern='davidlawrence_coins*'):
    """
    Find all directories matching the pattern.
    
    Args:
        pattern: Glob pattern to match directories
        
    Returns:
        list: List of Path objects for matching directories
    """
    matches = glob.glob(pattern)
    # Filter to only directories that have data/ and images/ subdirs
    valid_dirs = []
    for match in matches:
        path = Path(match)
        if path.is_dir() and (path / 'data').exists() and (path / 'images').exists():
            valid_dirs.append(path)
    return sorted(valid_dirs)


def process_mintmark_dataset(source_pattern='davidlawrence_coins*', 
                              output_dir='mintmark_dataset',
                              max_coins_per_folder=None,
                              dry_run=False):
    """
    Process David Lawrence coins into mint mark classification format.
    
    Args:
        source_pattern: Glob pattern to match source directories
        output_dir: Output directory for organized dataset
        max_coins_per_folder: Maximum number of coins to process per folder (None = all)
        dry_run: If True, show what would be done without copying
    """
    output_path = Path(output_dir)
    
    # Find all matching source directories
    source_dirs = find_source_directories(source_pattern)
    
    if not source_dirs:
        print(f"❌ Error: No directories found matching '{source_pattern}' with data/ and images/ subdirectories")
        return
    
    print("="*60)
    print("MINT MARK DATASET PREPARATION")
    print("="*60)
    print(f"Source pattern: {source_pattern}")
    print(f"Found {len(source_dirs)} source directories:")
    for sd in source_dirs:
        json_count = len(list((sd / 'data').glob('*.json')))
        print(f"  - {sd.name}: {json_count} coins")
    print(f"Output: {output_dir}")
    if max_coins_per_folder:
        print(f"Processing: First {max_coins_per_folder} coins per folder (TEST MODE)")
    else:
        print(f"Processing: ALL coins from ALL folders")
    if dry_run:
        print(f"Mode: DRY RUN (no files will be copied)")
    print("="*60)
    
    # Statistics
    stats = {
        'total': 0,
        'success': 0,
        'no_mint_mark_found': 0,
        'missing_obverse': 0,
        'missing_reverse': 0,
        'details_skipped': 0,
        'by_mint_mark': defaultdict(int),
        'by_source': defaultdict(int),
        'sample_descriptions': defaultdict(list)  # Store sample descriptions for each mint mark
    }
    
    # Helper function for sanitizing filenames
    def sanitize(val, max_len=20):
        """Sanitize a value for use in filename."""
        s = str(val).replace('/', '-').replace('\\', '-').replace(' ', '')
        s = ''.join(c for c in s if c.isalnum() or c in '-_$.')
        return s[:max_len]
    
    # Process each source directory
    for source_path in source_dirs:
        data_dir = source_path / 'data'
        images_dir = source_path / 'images'
        
        # Get all JSON files from this source
        json_files = sorted(list(data_dir.glob('*.json')))
        
        if max_coins_per_folder:
            json_files = json_files[:max_coins_per_folder]
        
        print(f"\n--- Processing: {source_path.name} ({len(json_files)} coins) ---")
        
        # Process each coin in this source
        for json_file in json_files:
            stats['total'] += 1
            inventory_id = json_file.stem
            
            # Read JSON data
            try:
                with open(json_file, 'r') as f:
                    coin_data = json.load(f)
            except Exception as e:
                print(f"⚠️  Error reading {json_file.name}: {e}")
                continue
            
            # Get description and year
            description = coin_data.get('description') or ''
            year = coin_data.get('year')
            
            # Skip "Details" grades (problem coins: cleaned, damaged, etc.)
            if 'detail' in description.lower():
                stats['details_skipped'] += 1
                continue
            
            # Extract mint mark from description
            mint_mark, extracted_year = extract_mint_mark(description, year)
            
            # Use extracted year if available, otherwise fall back to JSON year
            coin_year = extracted_year if extracted_year else year
            
            if mint_mark is None:
                stats['no_mint_mark_found'] += 1
                if stats['no_mint_mark_found'] <= 10:  # Only print first 10
                    print(f"⚠️  {inventory_id}: Could not extract mint mark from: {description}")
                continue
            
            # Store sample descriptions for verification
            if len(stats['sample_descriptions'][mint_mark]) < 5:
                stats['sample_descriptions'][mint_mark].append(description)
            
            # Find obverse (_2) and reverse (_3) images
            obverse_img = None
            reverse_img = None
            
            for img_path_str in coin_data.get('images', []):
                img_path = Path(img_path_str)
                
                # Check if it's _2 (obverse) or _3 (reverse)
                if img_path.stem.endswith('_2'):
                    if img_path.exists():
                        obverse_img = img_path
                    else:
                        local_img = images_dir / img_path.name
                        if local_img.exists():
                            obverse_img = local_img
                elif img_path.stem.endswith('_3'):
                    if img_path.exists():
                        reverse_img = img_path
                    else:
                        local_img = images_dir / img_path.name
                        if local_img.exists():
                            reverse_img = local_img
            
            # Check if we have both images
            if not obverse_img:
                stats['missing_obverse'] += 1
                continue
            
            if not reverse_img:
                stats['missing_reverse'] += 1
                continue
            
            # Create output directories for this mint mark
            mint_mark_dir = output_path / mint_mark
            obverse_dir = mint_mark_dir / 'obverse'
            reverse_dir = mint_mark_dir / 'reverse'
            
            if not dry_run:
                obverse_dir.mkdir(parents=True, exist_ok=True)
                reverse_dir.mkdir(parents=True, exist_ok=True)
            
            # Build filename with relevant info
            # Year is first to help with training/debugging mint mark vs year disambiguation
            cert_num = coin_data.get('cert_number', 'unknown')
            grade = coin_data.get('grade', 'unk')
            denomination = coin_data.get('denomination', 'unknown')
            grading_service = coin_data.get('grading_service', 'UNK')
            
            safe_cert = sanitize(cert_num, 30)
            safe_year = str(coin_year) if coin_year else 'nodate'
            safe_denom = sanitize(denomination, 10)
            safe_service = str(grading_service).upper()[:4]
            safe_grade = sanitize(grade, 10)
            
            # Format: YEAR_MINTMARK_... so year is prominent for training
            filename = f"{safe_year}_{mint_mark}_{safe_denom}_{safe_grade}_{safe_service}_{safe_cert}.jpg"
            
            obverse_dest = obverse_dir / filename
            reverse_dest = reverse_dir / filename
            
            if dry_run:
                if stats['success'] < 20:  # Only print first 20 in dry run
                    print(f"✓ {inventory_id}: {mint_mark} ← \"{description[:50]}...\"")
            else:
                shutil.copy2(obverse_img, obverse_dest)
                shutil.copy2(reverse_img, reverse_dest)
            
            stats['success'] += 1
            stats['by_mint_mark'][mint_mark] += 1
            stats['by_source'][source_path.name] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total coins processed: {stats['total']}")
    print(f"Successfully organized: {stats['success']}")
    print(f"Could not extract mint mark: {stats['no_mint_mark_found']}")
    print(f"Missing obverse: {stats['missing_obverse']}")
    print(f"Missing reverse: {stats['missing_reverse']}")
    print(f"Details skipped: {stats['details_skipped']} (problem coins)")
    
    print("\n" + "-"*60)
    print("COINS BY MINT MARK:")
    print("-"*60)
    for mm, count in sorted(stats['by_mint_mark'].items(), key=lambda x: x[1], reverse=True):
        pct = (count / stats['success'] * 100) if stats['success'] > 0 else 0
        print(f"  {mm:5s}: {count:6d} pairs ({pct:5.1f}%)")
    
    print("\n" + "-"*60)
    print("SAMPLE DESCRIPTIONS BY MINT MARK (for verification):")
    print("-"*60)
    for mm, samples in sorted(stats['sample_descriptions'].items()):
        print(f"\n  [{mm}]:")
        for sample in samples[:3]:
            print(f"    - {sample}")
    
    print("\n" + "-"*60)
    print("COINS BY SOURCE FOLDER:")
    print("-"*60)
    for source, count in sorted(stats['by_source'].items()):
        print(f"  {source}: {count} pairs")
    
    print("="*60)
    
    if not dry_run:
        print(f"\n✓ Dataset ready at: {output_dir}/")
        print(f"\nDirectory structure:")
        print(f"  {output_dir}/")
        for mm in sorted(stats['by_mint_mark'].keys()):
            print(f"    {mm}/")
            print(f"      obverse/")
            print(f"      reverse/")
    else:
        print(f"\nThis was a DRY RUN. No files were copied.")
        print(f"To actually process files, remove --dry-run flag")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare David Lawrence coins for mint mark classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with first 10 coins per folder (dry run)
  %(prog)s --test 10 --dry-run
  
  # Process first 10 coins per folder
  %(prog)s --test 10
  
  # Process all coins from all davidlawrence_coins* folders
  %(prog)s --all
  
  # Custom output directory
  %(prog)s --all --output my_mintmark_dataset
  
  # Use a specific source pattern
  %(prog)s --all --source "davidlawrence_coins_morgans"

Mint Marks:
  None  - Philadelphia (no mint mark on coin)
  P     - Philadelphia (explicit P mark, used in some years)
  D     - Denver (1906-present)
  DL    - Dahlonega (1838-1861) - auto-detected when D appears before 1862
  S     - San Francisco
  O     - New Orleans
  CC    - Carson City
  C     - Charlotte (1838-1861)
  W     - West Point
        """
    )
    
    parser.add_argument('--test', type=int, metavar='N', 
                       help='Test mode: process only first N coins per folder')
    parser.add_argument('--all', action='store_true',
                       help='Process all coins from all matching folders')
    parser.add_argument('--output', '-o', default='mintmark_dataset',
                       help='Output directory (default: mintmark_dataset)')
    parser.add_argument('--source', '-s', default='davidlawrence_coins*',
                       help='Source directory pattern (default: davidlawrence_coins*)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without copying files')
    
    args = parser.parse_args()
    
    # Determine max_coins
    if args.test:
        max_coins = args.test
    elif args.all:
        max_coins = None
    else:
        print("❌ Error: Must specify either --test N or --all")
        print("Run with --help for usage information")
        return
    
    process_mintmark_dataset(
        source_pattern=args.source,
        output_dir=args.output,
        max_coins_per_folder=max_coins,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()

