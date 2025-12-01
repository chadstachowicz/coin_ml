#!/usr/bin/env python3
"""
Quick dataset checker to help decide which training method to use.
"""
from pathlib import Path
from collections import defaultdict


def normalize_grade_name(grade):
    """
    Normalize grade name by keeping only up to the last digit.
    Examples:
        g06bn -> g06
        au58+ -> au58
        ms65 -> ms65
    """
    last_digit_pos = -1
    for i, char in enumerate(grade):
        if char.isdigit():
            last_digit_pos = i
    
    if last_digit_pos >= 0:
        return grade[:last_digit_pos + 1]
    
    return grade


def check_dataset():
    """Check current dataset and provide training recommendations."""
    
    print("="*70)
    print("PCGS COIN DATASET CHECKER")
    print("="*70)
    print()
    
    # Check images directory
    images_dir = Path('images')
    if not images_dir.exists():
        print("❌ ERROR: 'images/' directory not found!")
        print("   Run the web scraper first to collect coins.")
        return
    
    # Collect statistics with grade normalization
    images_by_grade = defaultdict(int)
    grade_sources = defaultdict(list)  # Track which folders contribute to each normalized grade
    total_images = 0
    
    for grade_folder in images_dir.iterdir():
        if not grade_folder.is_dir():
            continue
        
        # Skip non-grade folders
        if grade_folder.name in ['obverse', 'reverse']:
            continue
        
        # Check for 6000x3000 resolution folder
        high_res_folder = grade_folder / '6000x3000'
        if not high_res_folder.exists():
            continue
        
        # Count images
        count = len([f for f in high_res_folder.iterdir() 
                    if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        if count > 0:
            original_grade = grade_folder.name
            normalized_grade = normalize_grade_name(original_grade)
            images_by_grade[normalized_grade] += count
            grade_sources[normalized_grade].append((original_grade, count))
            total_images += count
    
    # Display results
    if not images_by_grade:
        print("❌ No images found in images/<grade>/6000x3000/ folders")
        print("\n💡 TIP: Use the web UI to scrape coins first:")
        print("   python app.py")
        print("   Then open http://localhost:8000")
        return
    
    print(f"📊 DATASET OVERVIEW")
    print("-"*70)
    print(f"{'Grade (Class)':<20} {'Images':<10} {'Status':<15} {'Bar':<25}")
    print("-"*70)
    
    max_count = max(images_by_grade.values())
    
    for grade, count in sorted(images_by_grade.items(), key=lambda x: x[1], reverse=True):
        # Status
        if count < 20:
            status = "⚠️  Too few"
            status_color = ""
        elif count < 50:
            status = "⚡ OK"
            status_color = ""
        elif count < 100:
            status = "✅ Good"
            status_color = ""
        else:
            status = "🌟 Excellent"
            status_color = ""
        
        # Bar chart
        bar_length = int((count / max_count) * 20)
        bar = "█" * bar_length
        
        print(f"{grade:<20} {count:<10} {status:<15} {bar}")
        
        # Show source folders if grade was combined from multiple
        sources = grade_sources[grade]
        if len(sources) > 1:
            source_str = " + ".join([f"{src}({cnt})" for src, cnt in sources])
            print(f"{'':20} ← {source_str}")
    
    print("-"*70)
    print(f"{'TOTAL':<20} {total_images:<10} {len(images_by_grade)} classes")
    print("="*70)
    print()
    
    # Statistics
    min_images = min(images_by_grade.values())
    max_images = max(images_by_grade.values())
    avg_images = total_images / len(images_by_grade)
    
    print("📈 STATISTICS")
    print("-"*70)
    print(f"Total Classes:        {len(images_by_grade)}")
    print(f"Total Images:         {total_images}")
    print(f"Avg Images/Class:     {avg_images:.1f}")
    print(f"Min Images/Class:     {min_images} ({min(images_by_grade, key=images_by_grade.get)})")
    print(f"Max Images/Class:     {max_images} ({max(images_by_grade, key=images_by_grade.get)})")
    
    # Check balance
    imbalance_ratio = max_images / min_images if min_images > 0 else float('inf')
    if imbalance_ratio > 3:
        print(f"\n⚠️  WARNING: Dataset is imbalanced (ratio: {imbalance_ratio:.1f}:1)")
        print("   Consider collecting more images for underrepresented classes.")
    else:
        print(f"\n✅ Dataset balance is good (ratio: {imbalance_ratio:.1f}:1)")
    
    print("="*70)
    print()
    
    # Recommendations
    print("🎯 TRAINING RECOMMENDATIONS")
    print("-"*70)
    
    if min_images < 20:
        print("\n⚠️  INSUFFICIENT DATA")
        print(f"   You have only {min_images} images in the smallest class.")
        print("   Minimum recommended: 20 images per class")
        print()
        print("   📥 ACTION: Collect more coins using bulk upload")
        print("      python app.py → http://localhost:8000")
        print()
    
    elif min_images < 50:
        print("\n⚡ LIMITED DATA - Use ResNet-50 (Transfer Learning)")
        print(f"   You have {min_images}-{max_images} images per class.")
        print()
        print("   ✅ RECOMMENDED: ResNet-50 Fine-tuning")
        print("      jupyter notebook coin_classifier_full.ipynb")
        print()
        print("   Why? Pre-trained weights work better with limited data")
        print()
        print("   📈 To improve: Collect 50+ images per class for better results")
        print()
    
    elif min_images < 100:
        print("\n✅ GOOD DATA - Either Method Works")
        print(f"   You have {min_images}-{max_images} images per class.")
        print()
        print("   OPTION 1 (Recommended): Custom CNN - Preserves rectangular shape")
        print("      python coin_classifier_custom.py")
        print("      + Full coin detail preserved (2:1 aspect ratio)")
        print("      + Higher accuracy potential")
        print("      - Longer training time")
        print()
        print("   OPTION 2 (Faster): ResNet-50 - Square images")
        print("      jupyter notebook coin_classifier_full.ipynb")
        print("      + Faster training (30-60 min)")
        print("      + Less memory required")
        print("      - Crops/distorts coin shape")
        print()
        print("   💡 TIP: Try both and compare results!")
        print()
    
    else:
        print("\n🌟 EXCELLENT DATA - Use Custom CNN")
        print(f"   You have {min_images}+ images per class. Perfect!")
        print()
        print("   ✅ RECOMMENDED: Custom CNN - Train from Scratch")
        print("      python coin_classifier_custom.py")
        print("      + Preserves rectangular coin shape (2:1 ratio)")
        print("      + Learns coin-specific features")
        print("      + Highest accuracy potential")
        print()
        print("   Expected Results:")
        print(f"      - Training time: 2-4 hours ({50} epochs)")
        print("      - Expected accuracy: 90-98%")
        print()
    
    print("="*70)
    print()
    
    # Next steps
    print("📋 NEXT STEPS")
    print("-"*70)
    print("1. Prepare dataset:")
    print("   python prepare_dataset.py")
    print()
    print("2. Choose training method (see recommendations above)")
    print()
    print("3. Monitor training:")
    print("   tensorboard --logdir=runs")
    print()
    print("4. For detailed comparison:")
    print("   Read TRAINING_COMPARISON.md")
    print("="*70)


if __name__ == '__main__':
    check_dataset()

