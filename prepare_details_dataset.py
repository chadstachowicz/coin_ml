#!/usr/bin/env python3
"""
Prepare Details vs Straight Coin Classification Dataset

Processes all davidlawrence_coins* folders and organizes images into:
- details_dataset/Details/   - Coins with "detail" in description (problem coins)
- details_dataset/Straight/  - Coins without "detail" in description (normal grades)

The dataset is balanced 1:1 between Details and Straight coins.

Usage:
  # Test with first 10 coins from each folder (dry run)
  python prepare_details_dataset.py --test 10 --dry-run
  
  # Process all coins, balanced dataset
  python prepare_details_dataset.py --all
  
  # Process all coins without balancing
  python prepare_details_dataset.py --all --no-balance
"""

import json
import shutil
import argparse
import glob
import random
from pathlib import Path
from collections import defaultdict


def normalize_grade(grade_str):
    """
    Normalize grade string for folder names using standard Sheldon scale.
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
            return f'grade{num:02d}'
    except ValueError:
        return grade_str.lower()


def find_source_directories(pattern='davidlawrence_coins*'):
    """
    Find all directories matching the pattern.
    """
    matches = glob.glob(pattern)
    valid_dirs = []
    for match in matches:
        path = Path(match)
        if path.is_dir() and (path / 'data').exists() and (path / 'images').exists():
            valid_dirs.append(path)
    return sorted(valid_dirs)


def collect_all_coins(source_pattern='davidlawrence_coins*', max_coins_per_folder=None):
    """
    Collect all coins from source directories and categorize as Details or Straight.
    
    Returns:
        tuple: (details_coins, straight_coins) - lists of coin data dicts
    """
    source_dirs = find_source_directories(source_pattern)
    
    if not source_dirs:
        print(f"❌ Error: No directories found matching '{source_pattern}'")
        return [], []
    
    print(f"Found {len(source_dirs)} source directories:")
    for sd in source_dirs:
        json_count = len(list((sd / 'data').glob('*.json')))
        print(f"  - {sd.name}: {json_count} coins")
    
    details_coins = []
    straight_coins = []
    
    stats = {
        'total': 0,
        'missing_grade': 0,
        'missing_obverse': 0,
        'missing_reverse': 0,
    }
    
    for source_path in source_dirs:
        data_dir = source_path / 'data'
        images_dir = source_path / 'images'
        
        json_files = sorted(list(data_dir.glob('*.json')))
        
        if max_coins_per_folder:
            json_files = json_files[:max_coins_per_folder]
        
        for json_file in json_files:
            stats['total'] += 1
            inventory_id = json_file.stem
            
            try:
                with open(json_file, 'r') as f:
                    coin_data = json.load(f)
            except Exception as e:
                continue
            
            # Extract grade
            grade = coin_data.get('grade')
            if not grade:
                stats['missing_grade'] += 1
                continue
            
            # Get description
            description = coin_data.get('description', '')
            
            # Find obverse (_2) and reverse (_3) images
            obverse_img = None
            reverse_img = None
            
            for img_path_str in coin_data.get('images', []):
                img_path = Path(img_path_str)
                
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
            
            if not obverse_img:
                stats['missing_obverse'] += 1
                continue
            
            if not reverse_img:
                stats['missing_reverse'] += 1
                continue
            
            # Build coin info
            coin_info = {
                'inventory_id': inventory_id,
                'grade': grade,
                'grade_folder': normalize_grade(grade),
                'description': description,
                'obverse_img': obverse_img,
                'reverse_img': reverse_img,
                'cert_number': coin_data.get('cert_number', 'unknown'),
                'year': coin_data.get('year', 'nodate'),
                'denomination': coin_data.get('denomination', 'unknown'),
                'grading_service': coin_data.get('grading_service', 'UNK'),
                'source': source_path.name
            }
            
            # Categorize based on "detail" in description
            if 'detail' in description.lower():
                details_coins.append(coin_info)
            else:
                straight_coins.append(coin_info)
    
    print(f"\nCollection stats:")
    print(f"  Total processed: {stats['total']}")
    print(f"  Missing grade: {stats['missing_grade']}")
    print(f"  Missing obverse: {stats['missing_obverse']}")
    print(f"  Missing reverse: {stats['missing_reverse']}")
    
    return details_coins, straight_coins


def process_details_dataset(source_pattern='davidlawrence_coins*',
                            output_dir='details_dataset',
                            max_coins_per_folder=None,
                            balance=True,
                            dry_run=False,
                            seed=42):
    """
    Process coins into Details vs Straight classification dataset.
    """
    output_path = Path(output_dir)
    
    print("="*60)
    print("DETAILS VS STRAIGHT DATASET PREPARATION")
    print("="*60)
    print(f"Source pattern: {source_pattern}")
    print(f"Output: {output_dir}")
    print(f"Balanced: {balance}")
    if max_coins_per_folder:
        print(f"Processing: First {max_coins_per_folder} coins per folder (TEST MODE)")
    else:
        print(f"Processing: ALL coins from ALL folders")
    if dry_run:
        print(f"Mode: DRY RUN (no files will be copied)")
    print("="*60)
    
    # Collect all coins
    details_coins, straight_coins = collect_all_coins(source_pattern, max_coins_per_folder)
    
    print(f"\nBefore balancing:")
    print(f"  Details coins: {len(details_coins)}")
    print(f"  Straight coins: {len(straight_coins)}")
    
    # Balance the dataset if requested
    if balance:
        random.seed(seed)
        min_count = min(len(details_coins), len(straight_coins))
        
        if min_count == 0:
            print("❌ Error: One category has no coins!")
            return
        
        # Randomly sample to balance
        if len(details_coins) > min_count:
            details_coins = random.sample(details_coins, min_count)
        if len(straight_coins) > min_count:
            straight_coins = random.sample(straight_coins, min_count)
        
        print(f"\nAfter balancing (1:1):")
        print(f"  Details coins: {len(details_coins)}")
        print(f"  Straight coins: {len(straight_coins)}")
        print(f"  Total pairs: {len(details_coins) + len(straight_coins)}")
    
    # Helper function for sanitizing filenames
    def sanitize(val, max_len=20):
        s = str(val).replace('/', '-').replace('\\', '-').replace(' ', '')
        s = ''.join(c for c in s if c.isalnum() or c in '-_$.')
        return s[:max_len]
    
    # Process and copy files
    stats = {
        'details': 0,
        'straight': 0,
        'by_grade': defaultdict(lambda: {'details': 0, 'straight': 0})
    }
    
    def copy_coin(coin_info, category):
        """Copy a single coin's images to the output directory."""
        category_dir = output_path / category
        obverse_dir = category_dir / 'obverse'
        reverse_dir = category_dir / 'reverse'
        
        if not dry_run:
            obverse_dir.mkdir(parents=True, exist_ok=True)
            reverse_dir.mkdir(parents=True, exist_ok=True)
        
        safe_cert = sanitize(coin_info['cert_number'], 30)
        safe_year = sanitize(coin_info['year'], 10)
        safe_denom = sanitize(coin_info['denomination'], 10)
        safe_service = str(coin_info['grading_service']).upper()[:4]
        grade_folder = coin_info['grade_folder']
        
        filename = f"{grade_folder}-{safe_service}-{safe_year}-{safe_denom}-{safe_cert}.jpg"
        
        obverse_dest = obverse_dir / filename
        reverse_dest = reverse_dir / filename
        
        if not dry_run:
            shutil.copy2(coin_info['obverse_img'], obverse_dest)
            shutil.copy2(coin_info['reverse_img'], reverse_dest)
        
        return filename
    
    print(f"\nProcessing Details coins...")
    for coin in details_coins:
        copy_coin(coin, 'Details')
        stats['details'] += 1
        stats['by_grade'][coin['grade_folder']]['details'] += 1
    
    print(f"Processing Straight coins...")
    for coin in straight_coins:
        copy_coin(coin, 'Straight')
        stats['straight'] += 1
        stats['by_grade'][coin['grade_folder']]['straight'] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Details coins: {stats['details']}")
    print(f"Straight coins: {stats['straight']}")
    print(f"Total coin pairs: {stats['details'] + stats['straight']}")
    
    print("\nBreakdown by grade:")
    for grade in sorted(stats['by_grade'].keys()):
        counts = stats['by_grade'][grade]
        print(f"  {grade}: Details={counts['details']}, Straight={counts['straight']}")
    
    print("="*60)
    
    if not dry_run:
        print(f"\n✓ Dataset ready at: {output_dir}/")
        print(f"\nStructure:")
        print(f"  {output_dir}/")
        print(f"    Details/")
        print(f"      obverse/  ({stats['details']} images)")
        print(f"      reverse/  ({stats['details']} images)")
        print(f"    Straight/")
        print(f"      obverse/  ({stats['straight']} images)")
        print(f"      reverse/  ({stats['straight']} images)")
    else:
        print(f"\nThis was a DRY RUN. No files were copied.")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Details vs Straight coin classification dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with first 10 coins per folder (dry run)
  %(prog)s --test 10 --dry-run
  
  # Process first 10 coins per folder
  %(prog)s --test 10
  
  # Process all coins, balanced 1:1
  %(prog)s --all
  
  # Process all coins without balancing
  %(prog)s --all --no-balance
  
  # Custom output directory
  %(prog)s --all --output my_details_dataset
        """
    )
    
    parser.add_argument('--test', type=int, metavar='N',
                       help='Test mode: process only first N coins per folder')
    parser.add_argument('--all', action='store_true',
                       help='Process all coins from all matching folders')
    parser.add_argument('--output', '-o', default='details_dataset',
                       help='Output directory (default: details_dataset)')
    parser.add_argument('--source', '-s', default='davidlawrence_coins*',
                       help='Source directory pattern (default: davidlawrence_coins*)')
    parser.add_argument('--no-balance', action='store_true',
                       help='Do not balance the dataset (keep all samples)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without copying files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for balanced sampling (default: 42)')
    
    args = parser.parse_args()
    
    if args.test:
        max_coins = args.test
    elif args.all:
        max_coins = None
    else:
        print("❌ Error: Must specify either --test N or --all")
        print("Run with --help for usage information")
        return
    
    process_details_dataset(
        source_pattern=args.source,
        output_dir=args.output,
        max_coins_per_folder=max_coins,
        balance=not args.no_balance,
        dry_run=args.dry_run,
        seed=args.seed
    )


if __name__ == '__main__':
    main()








