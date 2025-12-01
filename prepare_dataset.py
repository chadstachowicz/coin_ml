"""
Prepare coin images dataset for training.
Creates train/test/validation splits (70/20/10) from images folder.
Preserves rectangular aspect ratio for custom CNN training.
"""
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Configuration
SOURCE_DIR = 'images'
OUTPUT_DIR = 'coin_dataset'
RESOLUTION = '6000x3000'  # The resolution folder to use (rectangular - preserves 2:1 aspect ratio)
TRAIN_RATIO = 0.7
TEST_RATIO = 0.2
VAL_RATIO = 0.1
RANDOM_SEED = 42

print("NOTE: Using rectangular images (6000x3000) - 2:1 aspect ratio preserved")
print("Training with custom CNN architecture designed for rectangular coins\n")


def normalize_grade_name(grade):
    """
    Normalize grade name by keeping only up to the last digit.
    Examples:
        g06bn -> g06
        au58+ -> au58
        ms65 -> ms65
        g04 -> g04
    """
    # Find the last digit position
    last_digit_pos = -1
    for i, char in enumerate(grade):
        if char.isdigit():
            last_digit_pos = i
    
    # If we found a digit, return everything up to and including it
    if last_digit_pos >= 0:
        return grade[:last_digit_pos + 1]
    
    # If no digit found, return the original (shouldn't happen with coin grades)
    return grade


def collect_images_by_grade():
    """
    Collect all images from images/<grade>/6000x3000/ folders.
    Returns dict: {grade: [list of image paths]}
    """
    images_by_grade = defaultdict(list)
    
    source_path = Path(SOURCE_DIR)
    if not source_path.exists():
        print(f"Error: {SOURCE_DIR} directory not found!")
        return images_by_grade
    
    # Track which original folders map to which normalized grades
    grade_mapping = defaultdict(list)
    
    # Scan all grade folders
    for grade_folder in source_path.iterdir():
        if not grade_folder.is_dir():
            continue
        
        # Skip obverse/reverse folders
        if grade_folder.name in ['obverse', 'reverse']:
            continue
        
        original_grade = grade_folder.name
        normalized_grade = normalize_grade_name(original_grade)
        
        resolution_folder = grade_folder / RESOLUTION
        
        if not resolution_folder.exists():
            print(f"Warning: No {RESOLUTION} folder found for grade {original_grade}")
            continue
        
        # Collect all image files under the normalized grade
        img_count = 0
        for img_file in resolution_folder.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                images_by_grade[normalized_grade].append(img_file)
                img_count += 1
        
        if img_count > 0:
            grade_mapping[normalized_grade].append(original_grade)
            if original_grade != normalized_grade:
                print(f"Found {img_count} images for grade: {original_grade} -> normalized to {normalized_grade}")
            else:
                print(f"Found {img_count} images for grade: {original_grade}")
    
    # Print summary of combined grades
    print("\n" + "=" * 60)
    print("Grade Normalization Summary:")
    print("=" * 60)
    for normalized, originals in sorted(grade_mapping.items()):
        if len(originals) > 1:
            print(f"{normalized:15} ← Combined from: {', '.join(originals)}")
            print(f"{'':15}    Total images: {len(images_by_grade[normalized])}")
        else:
            print(f"{normalized:15}    {len(images_by_grade[normalized])} images")
    print("=" * 60)
    
    return images_by_grade


def create_dataset_structure():
    """Create the dataset directory structure."""
    output_path = Path(OUTPUT_DIR)
    
    # Remove existing dataset if it exists
    if output_path.exists():
        response = input(f"{OUTPUT_DIR} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return False
        shutil.rmtree(output_path)
    
    # Create structure
    for split in ['train', 'test', 'val']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    return True


def split_and_copy_images(images_by_grade):
    """
    Split images into train/test/val and copy to dataset structure.
    """
    random.seed(RANDOM_SEED)
    output_path = Path(OUTPUT_DIR)
    
    stats = {
        'train': 0,
        'test': 0,
        'val': 0
    }
    
    for grade, image_paths in images_by_grade.items():
        if len(image_paths) == 0:
            continue
        
        # Shuffle images
        random.shuffle(image_paths)
        
        # Calculate split indices
        total = len(image_paths)
        train_end = int(total * TRAIN_RATIO)
        test_end = train_end + int(total * TEST_RATIO)
        
        # Split images
        train_images = image_paths[:train_end]
        test_images = image_paths[train_end:test_end]
        val_images = image_paths[test_end:]
        
        print(f"\nGrade '{grade}':")
        print(f"  Total: {total} | Train: {len(train_images)} | Test: {len(test_images)} | Val: {len(val_images)}")
        
        # Create grade folders in each split
        for split in ['train', 'test', 'val']:
            (output_path / split / grade).mkdir(parents=True, exist_ok=True)
        
        # Copy images to appropriate splits
        for img_path in train_images:
            dest = output_path / 'train' / grade / img_path.name
            shutil.copy2(img_path, dest)
            stats['train'] += 1
        
        for img_path in test_images:
            dest = output_path / 'test' / grade / img_path.name
            shutil.copy2(img_path, dest)
            stats['test'] += 1
        
        for img_path in val_images:
            dest = output_path / 'val' / grade / img_path.name
            shutil.copy2(img_path, dest)
            stats['val'] += 1
    
    return stats


def main():
    """Main function to prepare the dataset."""
    print("=" * 60)
    print("PCGS Coin Dataset Preparation")
    print("=" * 60)
    print(f"Source: {SOURCE_DIR}/<grade>/{RESOLUTION}/")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"Split Ratio: Train={TRAIN_RATIO*100}% | Test={TEST_RATIO*100}% | Val={VAL_RATIO*100}%")
    print("=" * 60)
    
    # Step 1: Collect all images by grade
    print("\n[1/3] Collecting images from source directory...")
    images_by_grade = collect_images_by_grade()
    
    if not images_by_grade:
        print("Error: No images found! Make sure you have images in the correct structure.")
        print(f"Expected: {SOURCE_DIR}/<grade>/{RESOLUTION}/*.jpg")
        return
    
    total_images = sum(len(imgs) for imgs in images_by_grade.values())
    print(f"\nTotal images found: {total_images}")
    print(f"Total classes (grades): {len(images_by_grade)}")
    
    # Check if we have enough images
    min_images = min(len(imgs) for imgs in images_by_grade.values())
    if min_images < 10:
        print(f"\nWarning: Some grades have very few images (minimum: {min_images})")
        print("This may not be enough for proper training.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Step 2: Create dataset structure
    print("\n[2/3] Creating dataset directory structure...")
    if not create_dataset_structure():
        return
    
    # Step 3: Split and copy images
    print("\n[3/3] Splitting and copying images...")
    stats = split_and_copy_images(images_by_grade)
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"Train images: {stats['train']}")
    print(f"Test images: {stats['test']}")
    print(f"Validation images: {stats['val']}")
    print(f"\nDataset location: {OUTPUT_DIR}/")
    print("\nYou can now use this dataset with PyTorch's ImageFolder:")
    print(f"  training_dataset = datasets.ImageFolder('{OUTPUT_DIR}/train', transform=transform)")
    print(f"  testing_dataset = datasets.ImageFolder('{OUTPUT_DIR}/test', transform=transform)")
    print(f"  validation_dataset = datasets.ImageFolder('{OUTPUT_DIR}/val', transform=transform)")
    print("=" * 60)


if __name__ == '__main__':
    main()

