#!/usr/bin/env python3
"""
Preprocess Dataset for Training

Applies Hough circle detection and white background to all coin images,
saving them to a new directory. This avoids repeating expensive preprocessing
during every training batch.

Usage:
    python preprocess_dataset.py --input davidlawrence_dataset/Circulation --output davidlawrence_preprocessed/Circulation
    python preprocess_dataset.py --input davidlawrence_dataset --output davidlawrence_preprocessed --all
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Default configuration
DEFAULT_OUTPUT_SIZE = 768


def preprocess_coin_image(img_bgr, output_size=768):
    """
    Preprocess coin image with Hough circle detection and white background.
    
    Args:
        img_bgr: OpenCV BGR image
        output_size: Output size (square)
    
    Returns:
        Preprocessed BGR image
    """
    height, width = img_bgr.shape[:2]
    min_dim = min(height, width)
    
    # Convert to grayscale for circle detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dim // 2,
        param1=50, param2=30,
        minRadius=int(min_dim * 0.2), maxRadius=int(min_dim * 0.5)
    )
    
    # Try with more lenient parameters if first attempt fails
    if circles is None:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dim // 3,
            param1=100, param2=20,
            minRadius=int(min_dim * 0.15), maxRadius=int(min_dim * 0.55)
        )
    
    # Default to center crop if no circle detected
    if circles is None:
        cx, cy = width // 2, height // 2
        radius = min_dim // 2 - 10
    else:
        circles = np.uint16(np.around(circles))
        cx, cy, radius = circles[0][0]
    
    # Create circular mask on original image
    mask_radius = int(radius * 1.02)  # Slight padding
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), mask_radius, 255, -1)
    
    # Feather mask edges for smooth transition
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    
    # Create white background and blend
    white_img = np.ones_like(img_bgr) * 255
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    blended = (img_bgr.astype(float) * mask_3ch + white_img.astype(float) * (1 - mask_3ch)).astype(np.uint8)
    
    # Crop to bounding box of the coin
    padding = int(radius * 0.05)
    crop_radius = radius + padding
    x1 = max(0, int(cx - crop_radius))
    y1 = max(0, int(cy - crop_radius))
    x2 = min(width, int(cx + crop_radius))
    y2 = min(height, int(cy + crop_radius))
    
    cropped = blended[y1:y2, x1:x2]
    
    # Resize to output size
    white_bg = np.ones((output_size, output_size, 3), dtype=np.uint8) * 255
    
    crop_h, crop_w = cropped.shape[:2]
    if crop_h > 0 and crop_w > 0:
        scale = (output_size * 0.92) / max(crop_h, crop_w)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        x_offset = (output_size - new_w) // 2
        y_offset = (output_size - new_h) // 2
        white_bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return white_bg


def process_single_image(args):
    """Process a single image (for parallel processing)."""
    input_path, output_path, output_size = args
    
    try:
        # Read image
        img = cv2.imread(str(input_path))
        if img is None:
            return input_path, False, "Could not read image"
        
        # Preprocess
        processed = preprocess_coin_image(img, output_size)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), processed, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return input_path, True, None
    except Exception as e:
        return input_path, False, str(e)


def preprocess_dataset(input_dir, output_dir, output_size=768, num_workers=None, skip_existing=True):
    """
    Preprocess all images in a dataset directory.
    
    Args:
        input_dir: Input directory (e.g., davidlawrence_dataset/Circulation)
        output_dir: Output directory for preprocessed images
        output_size: Output image size (square)
        num_workers: Number of parallel workers (default: CPU count - 1)
        skip_existing: Skip files that already exist in output directory
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print("=" * 60)
    print("DATASET PREPROCESSING")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Output size: {output_size}x{output_size}")
    print(f"Workers: {num_workers}")
    print(f"Skip existing: {skip_existing}")
    print("=" * 60)
    
    # Collect all image paths
    tasks = []
    skipped_count = 0
    
    # Walk through directory structure
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_file = Path(root) / file
                
                # Maintain directory structure
                relative_path = input_file.relative_to(input_path)
                output_file = output_path / relative_path
                
                # Skip if output already exists
                if skip_existing and output_file.exists():
                    skipped_count += 1
                    continue
                
                tasks.append((input_file, output_file, output_size))
    
    print(f"Found {len(tasks) + skipped_count} total images")
    if skip_existing:
        print(f"  Skipping {skipped_count} already processed")
        print(f"  Processing {len(tasks)} new images")
    
    if not tasks:
        print("No images found!")
        return
    
    # Process images in parallel
    success_count = 0
    error_count = 0
    errors = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_image, task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc="Preprocessing") as pbar:
            for future in as_completed(futures):
                input_file, success, error = future.result()
                
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    errors.append((input_file, error))
                
                pbar.update(1)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"✓ Processed: {success_count}")
    print(f"✗ Errors: {error_count}")
    print(f"Output directory: {output_dir}")
    
    if errors:
        print(f"\nFirst 10 errors:")
        for path, error in errors[:10]:
            print(f"  {path}: {error}")
    
    # Create a marker file to indicate preprocessing is complete
    marker_file = output_path / "_preprocessed.txt"
    with open(marker_file, 'w') as f:
        f.write(f"Preprocessed with output_size={output_size}\n")
        f.write(f"Total images: {success_count}\n")
        f.write(f"Errors: {error_count}\n")
    
    print(f"\n✓ Ready to use! Update your training script:")
    print(f"  DATA_DIR = '{output_dir}'")
    print(f"  USE_PREPROCESSING = False  # Already preprocessed!")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess coin dataset with Hough circle detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess Circulation folder (skips existing by default)
  python preprocess_dataset.py -i davidlawrence_dataset/Circulation -o davidlawrence_preprocessed/Circulation
  
  # Preprocess entire dataset (Proof + Circulation)
  python preprocess_dataset.py -i davidlawrence_dataset -o davidlawrence_preprocessed --all
  
  # Custom output size
  python preprocess_dataset.py -i davidlawrence_dataset/Circulation -o preprocessed_512 --size 512
  
  # Force reprocess all (ignore existing)
  python preprocess_dataset.py -i davidlawrence_dataset/Circulation -o davidlawrence_preprocessed/Circulation --force
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory containing coin images')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory for preprocessed images')
    parser.add_argument('--size', type=int, default=DEFAULT_OUTPUT_SIZE,
                        help=f'Output image size (default: {DEFAULT_OUTPUT_SIZE})')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--all', action='store_true',
                        help='Process all subdirectories')
    parser.add_argument('--force', action='store_true',
                        help='Reprocess all images even if output exists')
    
    args = parser.parse_args()
    
    preprocess_dataset(
        input_dir=args.input,
        output_dir=args.output,
        output_size=args.size,
        num_workers=args.workers,
        skip_existing=not args.force
    )


if __name__ == '__main__':
    main()

