#!/usr/bin/env python3
"""
Prepare David Lawrence Morgan Dollars for Training

Processes ONLY the davidlawrence_coins_morgans folder and organizes images into training format:
- Reads grade from JSON files
- Identifies obverse (_2) and reverse (_3) images
- Separates Proof and Circulation coins based on description
- Organizes into:
  - morgans_dataset/Proof/<grade>/obverse/ and morgans_dataset/Proof/<grade>/reverse/
  - morgans_dataset/Circulation/<grade>/obverse/ and morgans_dataset/Circulation/<grade>/reverse/

Usage:
  # Test with first 10 coins
  python prepare_morgans_dataset.py --test 10
  
  # Process all Morgan dollars
  python prepare_morgans_dataset.py --all
"""

import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


# Fixed source directory - only Morgans
SOURCE_DIR = Path('davidlawrence_coins_morgans')
DEFAULT_OUTPUT_DIR = 'morgans_dataset'


def normalize_grade(grade_str):
    """
    Normalize grade string for folder names using standard Sheldon scale.
    
    Grade Ranges:
      P01-P02: Poor
      FR02: Fair  
      AG03: About Good
      G04-G06: Good
      VG08-VG10: Very Good
      F12-F15: Fine
      VF20-VF35: Very Fine
      XF40-XF45: Extremely Fine
      AU50-AU58: About Uncirculated
      MS60-MS70: Mint State
    
    Examples:
      - '66' → 'ms66'
      - '50' → 'au50'
      - 'MS66' → 'ms66'
    """
    if not grade_str:
        return 'unknown'
    
    grade_str = str(grade_str).strip().upper()
    
    # If already has prefix, just lowercase it
    if not grade_str[0].isdigit():
        return grade_str.lower()
    
    # Convert number to proper grade designation
    try:
        num = int(grade_str)
        
        if num <= 1:
            return f'p{num:02d}'
        elif num == 2:
            return 'fr02'
        elif num == 3:
            return 'ag03'
        elif num in [4, 6]:
            return f'g{num:02d}'
        elif num in [8, 10]:
            return f'vg{num:02d}'
        elif num in [12, 15]:
            return f'f{num:02d}'
        elif num in [20, 25, 30, 35]:
            return f'vf{num:02d}'
        elif num in [40, 45]:
            return f'xf{num:02d}'
        elif num in [50, 53, 55, 58]:
            return f'au{num:02d}'
        elif 60 <= num <= 70:
            return f'ms{num:02d}'
        else:
            # Fallback: return as-is with 'grade' prefix
            return f'grade{num:02d}'
    except ValueError:
        return grade_str.lower()


def process_morgans(output_dir=DEFAULT_OUTPUT_DIR,
                    max_coins=None,
                    dry_run=False):
    """
    Process Morgan dollars into training format.
    
    Args:
        output_dir: Output directory for organized dataset
        max_coins: Maximum number of coins to process (None = all)
        dry_run: If True, show what would be done without copying
    """
    output_path = Path(output_dir)
    data_dir = SOURCE_DIR / 'data'
    images_dir = SOURCE_DIR / 'images'
    
    # Verify source exists
    if not SOURCE_DIR.exists():
        print(f"❌ Error: Source directory not found: {SOURCE_DIR}")
        return
    
    if not data_dir.exists() or not images_dir.exists():
        print(f"❌ Error: Missing data/ or images/ subdirectory in {SOURCE_DIR}")
        return
    
    # Get all JSON files
    json_files = sorted(list(data_dir.glob('*.json')))
    total_available = len(json_files)
    
    if max_coins:
        json_files = json_files[:max_coins]
    
    print("="*60)
    print("MORGAN DOLLARS DATASET PREPARATION")
    print("="*60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Available coins: {total_available}")
    print(f"Output: {output_dir}")
    if max_coins:
        print(f"Processing: First {max_coins} coins (TEST MODE)")
    else:
        print(f"Processing: ALL {total_available} coins")
    if dry_run:
        print(f"Mode: DRY RUN (no files will be copied)")
    print("="*60)
    
    # Statistics
    stats = {
        'total': 0,
        'success': 0,
        'missing_grade': 0,
        'missing_obverse': 0,
        'missing_reverse': 0,
        'details_skipped': 0,
        'proof_count': 0,
        'circulation_count': 0,
        'by_grade': defaultdict(int)
    }
    
    # Helper function for sanitizing filenames
    def sanitize(val, max_len=20):
        """Sanitize a value for use in filename."""
        s = str(val).replace('/', '-').replace('\\', '-').replace(' ', '')
        s = ''.join(c for c in s if c.isalnum() or c in '-_$.')
        return s[:max_len]
    
    # Process each coin
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
        
        # Extract grade
        grade = coin_data.get('grade')
        if not grade:
            stats['missing_grade'] += 1
            print(f"⚠️  {inventory_id}: No grade found")
            continue
        
        grade_folder = normalize_grade(grade)
        
        # Determine if coin is Proof or Circulation based on description
        description = coin_data.get('description', '')
        
        # Skip "Details" grades (problem coins: cleaned, damaged, etc.)
        if 'detail' in description.lower():
            stats['details_skipped'] += 1
            continue
        
        is_proof = 'proof' in description.lower()
        
        # Set the root category folder
        category_folder = 'Proof' if is_proof else 'Circulation'
        
        if is_proof:
            stats['proof_count'] += 1
        else:
            stats['circulation_count'] += 1
        
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
            print(f"⚠️  {inventory_id}: Missing obverse image (_2)")
            continue
        
        if not reverse_img:
            stats['missing_reverse'] += 1
            print(f"⚠️  {inventory_id}: Missing reverse image (_3)")
            continue
        
        # Create output directories with category folder
        grade_dir = output_path / category_folder / grade_folder
        obverse_dir = grade_dir / 'obverse'
        reverse_dir = grade_dir / 'reverse'
        
        if not dry_run:
            obverse_dir.mkdir(parents=True, exist_ok=True)
            reverse_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images with consistent naming
        cert_num = coin_data.get('cert_number', 'unknown')
        year = coin_data.get('year', 'nodate')
        denomination = coin_data.get('denomination', 'unknown')
        grading_service = coin_data.get('grading_service', 'UNK')
        
        safe_cert = sanitize(cert_num, 30)
        safe_year = sanitize(year, 10)
        safe_denom = sanitize(denomination, 10)
        safe_service = str(grading_service).upper()[:4]
        
        filename = f"{grade_folder}-{safe_service}-{safe_year}-{safe_denom}-{safe_cert}.jpg"
        
        obverse_dest = obverse_dir / filename
        reverse_dest = reverse_dir / filename
        
        if dry_run:
            print(f"✓ {inventory_id} (Grade: {grade_folder})")
            print(f"  Obverse: {obverse_img.name} → {obverse_dest}")
            print(f"  Reverse: {reverse_img.name} → {reverse_dest}")
        else:
            shutil.copy2(obverse_img, obverse_dest)
            shutil.copy2(reverse_img, reverse_dest)
        
        stats['success'] += 1
        stats['by_grade'][grade_folder] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total coins processed: {stats['total']}")
    print(f"Successfully organized: {stats['success']}")
    print(f"  - Proof coins: {stats['proof_count']}")
    print(f"  - Circulation coins: {stats['circulation_count']}")
    print(f"Missing grade: {stats['missing_grade']}")
    print(f"Missing obverse: {stats['missing_obverse']}")
    print(f"Missing reverse: {stats['missing_reverse']}")
    print(f"Details skipped: {stats['details_skipped']} (problem coins)")
    
    print("\nCoins by grade:")
    for grade, count in sorted(stats['by_grade'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {grade}: {count} pairs")
    print("="*60)
    
    if not dry_run:
        print(f"\n✓ Dataset ready at: {output_dir}/")
        print(f"\nTo train, use:")
        print(f"  python coin_classifier_ordinal_regression.py")
        print(f"  # Or edit DATA_DIR in the script to point to '{output_dir}'")
    else:
        print(f"\nThis was a DRY RUN. No files were copied.")
        print(f"To actually process files, remove --dry-run flag")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Morgan dollars for training (from davidlawrence_coins_morgans)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with first 10 coins (dry run)
  %(prog)s --test 10 --dry-run
  
  # Process first 10 coins
  %(prog)s --test 10
  
  # Process ALL Morgan dollars
  %(prog)s --all
  
  # Custom output directory
  %(prog)s --all --output my_morgans_dataset
        """
    )
    
    parser.add_argument('--test', type=int, metavar='N', 
                       help='Test mode: process only first N coins')
    parser.add_argument('--all', action='store_true',
                       help='Process all Morgan dollars')
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
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
    
    process_morgans(
        output_dir=args.output,
        max_coins=max_coins,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()








