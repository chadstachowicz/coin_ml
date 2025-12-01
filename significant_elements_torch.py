#!/usr/bin/env python3
"""
Significant Element Detection for Coin Images (PyTorch version)

Implements the pipeline:
1. Gaussian blur (7x7 kernel, sigma=1.17)
2. Sobel edge detection for gradient magnitudes  
3. Otsu's thresholding for binary significant element mask
"""

import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import argparse
import math


def gaussian_kernel_2d(kernel_size=7, sigma=1.17):
    """Create a 2D Gaussian kernel."""
    # Create 1D Gaussian
    x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    
    # Create 2D kernel
    gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
    
    return gauss_2d


def gaussian_blur(image_tensor, kernel_size=7, sigma=1.17):
    """Apply Gaussian blur using PyTorch."""
    # Create Gaussian kernel
    kernel = gaussian_kernel_2d(kernel_size, sigma)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, kernel_size, kernel_size]
    
    # Apply convolution
    padding = kernel_size // 2
    blurred = F.conv2d(image_tensor, kernel, padding=padding)
    
    return blurred


def sobel_gradients(image_tensor):
    """Compute Sobel gradients."""
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
    
    # Apply Sobel filters
    grad_x = F.conv2d(image_tensor, sobel_x, padding=1)
    grad_y = F.conv2d(image_tensor, sobel_y, padding=1)
    
    # Compute magnitude
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    
    return magnitude, grad_x, grad_y


def otsu_threshold(image_tensor):
    """
    Compute Otsu's threshold using PyTorch.
    
    Returns:
        threshold_value (float), binary_tensor
    """
    # Flatten and normalize to 0-255
    img_flat = image_tensor.flatten()
    img_min, img_max = img_flat.min(), img_flat.max()
    
    if img_max > img_min:
        normalized = ((img_flat - img_min) / (img_max - img_min) * 255)
    else:
        return 0, torch.zeros_like(image_tensor)
    
    # Compute histogram
    hist = torch.histc(normalized, bins=256, min=0, max=255)
    
    # Normalize histogram
    hist = hist / hist.sum()
    
    # Compute cumulative sums
    bin_centers = torch.arange(256, dtype=torch.float32)
    cum_sum = torch.cumsum(hist, dim=0)
    cum_mean = torch.cumsum(hist * bin_centers, dim=0)
    
    # Global mean
    global_mean = cum_mean[-1]
    
    # Between-class variance for each threshold
    between_var = torch.zeros(256)
    for t in range(256):
        w0 = cum_sum[t]
        w1 = 1 - w0
        
        if w0 == 0 or w1 == 0:
            continue
        
        mu0 = cum_mean[t] / w0
        mu1 = (global_mean - cum_mean[t]) / w1
        
        between_var[t] = w0 * w1 * (mu0 - mu1) ** 2
    
    # Find optimal threshold
    threshold = torch.argmax(between_var).item()
    
    # Scale back to original range
    threshold_scaled = threshold / 255.0 * (img_max - img_min) + img_min
    
    # Create binary image
    binary = (image_tensor > threshold_scaled).float()
    
    return threshold, binary


def detect_significant_elements(image_path, save_output=True, detect_tiny=True, output_root='davidlawrence_elements'):
    """
    Detect significant and tiny elements in a coin image using PyTorch.
    
    Args:
        image_path: Path to the coin image
        save_output: Whether to save the output images
        detect_tiny: Whether to also detect tiny elements
        output_root: Root directory for element outputs (default: 'davidlawrence_elements')
    
    Returns:
        dict: Dictionary containing all results
    """
    # Read image
    image = Image.open(image_path)
    
    # Convert to grayscale if needed
    if image.mode != 'L':
        gray = image.convert('L')
    else:
        gray = image
    
    # Convert to tensor [1, 1, H, W]
    import torchvision.transforms as transforms
    to_tensor = transforms.ToTensor()
    gray_tensor = to_tensor(gray).unsqueeze(0)  # [1, 1, H, W]
    
    print(f"Processing: {Path(image_path).name}")
    print(f"Image size: {gray_tensor.shape[2:]} (H x W)")
    
    # ========================================================================
    # SIGNIFICANT ELEMENTS (Large features)
    # ========================================================================
    print("\n[SIGNIFICANT ELEMENTS]")
    
    # Step 1: Gaussian blur with larger kernel
    sigma_large = 1.17
    blurred_large = gaussian_blur(gray_tensor, kernel_size=7, sigma=sigma_large)
    print(f"✓ Applied Gaussian blur (kernel=7x7, sigma={sigma_large})")
    
    # Step 2: Sobel edge detection
    gradient_large, grad_x_large, grad_y_large = sobel_gradients(blurred_large)
    print(f"✓ Applied Sobel edge detection")
    print(f"  Gradient range: [{gradient_large.min():.4f}, {gradient_large.max():.4f}]")
    
    # Step 3: Otsu's thresholding
    threshold_sig, I_SE = otsu_threshold(gradient_large)
    print(f"✓ Applied Otsu's thresholding")
    print(f"  Otsu threshold: {threshold_sig}")
    
    sig_pixels = (I_SE > 0).sum().item()
    total_pixels = I_SE.numel()
    print(f"  Significant pixels: {sig_pixels:,} ({100 * sig_pixels / total_pixels:.2f}%)")
    
    # ========================================================================
    # TINY ELEMENTS (Fine features)
    # ========================================================================
    I_TE = None
    I_R = None
    blurred_small = None
    gradient_small = None
    
    if detect_tiny:
        print("\n[TINY ELEMENTS]")
        
        # Step 1: Gaussian blur with smaller kernel (captures finer details)
        sigma_small = 0.83
        blurred_small = gaussian_blur(gray_tensor, kernel_size=5, sigma=sigma_small)
        print(f"✓ Applied Gaussian blur (kernel=5x5, sigma={sigma_small})")
        
        # Step 2: Sobel edge detection
        gradient_small, grad_x_small, grad_y_small = sobel_gradients(blurred_small)
        print(f"✓ Applied Sobel edge detection")
        print(f"  Gradient range: [{gradient_small.min():.4f}, {gradient_small.max():.4f}]")
        
        # Step 3: Otsu's thresholding to create I_R
        threshold_tiny, I_R = otsu_threshold(gradient_small)
        print(f"✓ Applied Otsu's thresholding")
        print(f"  Otsu threshold: {threshold_tiny}")
        
        # Step 4: Subtract significant elements to get tiny elements
        # I_TE = I_R - I_SE (only keep pixels that are in I_R but not in I_SE)
        I_TE = torch.where((I_R > 0) & (I_SE == 0), torch.ones_like(I_R), torch.zeros_like(I_R))
        
        tiny_pixels = (I_TE > 0).sum().item()
        print(f"✓ Computed tiny elements (I_R - I_SE)")
        print(f"  Tiny pixels: {tiny_pixels:,} ({100 * tiny_pixels / total_pixels:.2f}%)")
    
    # Prepare results
    results = {
        'original': image,
        'grayscale': gray_tensor,
        'blurred_large': blurred_large,
        'gradient_large': gradient_large,
        'significant_elements': I_SE,
        'otsu_threshold_sig': threshold_sig,
        'image_path': image_path
    }
    
    if detect_tiny:
        results.update({
            'blurred_small': blurred_small,
            'gradient_small': gradient_small,
            'I_R': I_R,
            'tiny_elements': I_TE,
            'otsu_threshold_tiny': threshold_tiny
        })
    
    # Save outputs
    if save_output:
        # Mirror the directory structure in the output root
        # e.g., davidlawrence_dataset/Proof/ms65/obverse/coin.jpg
        #    -> davidlawrence_elements/Proof/ms65/obverse/coin.jpg
        
        image_path_obj = Path(image_path)
        
        # Find the relative path from davidlawrence_dataset
        # This handles both "Proof" and "Circulation" categories
        parts = image_path_obj.parts
        
        # Find where 'davidlawrence_dataset' is in the path
        try:
            dataset_idx = parts.index('davidlawrence_dataset')
            # Get everything after 'davidlawrence_dataset' (e.g., Proof/ms65/obverse)
            relative_parts = parts[dataset_idx + 1:]
            
            # Reconstruct the path under the new root
            output_dir = Path(output_root) / Path(*relative_parts[:-1])  # Exclude filename
        except ValueError:
            # If 'davidlawrence_dataset' not in path, just use a subdirectory
            output_dir = Path(output_root) / "uncategorized"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = image_path_obj.stem
        
        # Convert tensors back to PIL Images
        to_pil = transforms.ToPILImage()
        
        print(f"\n[SAVING OUTPUTS]")
        
        # ---- Significant Elements ----
        # Save blurred (large kernel)
        blur_large_img = to_pil(blurred_large.squeeze(0))
        blur_large_img.save(output_dir / f"{base_name}_blur_large.jpg")
        
        # Save gradient (large)
        grad_norm_large = (gradient_large - gradient_large.min()) / (gradient_large.max() - gradient_large.min())
        grad_large_img = to_pil(grad_norm_large.squeeze(0))
        grad_large_img.save(output_dir / f"{base_name}_gradient_large.jpg")
        
        # Save significant elements binary mask
        sig_img = to_pil(I_SE.squeeze(0))
        sig_img.save(output_dir / f"{base_name}_significant.jpg")
        
        # Create significant elements overlay (red)
        original_rgb = image.convert('RGB')
        overlay_sig = transforms.ToTensor()(original_rgb)  # [3, H, W]
        mask_sig = I_SE.squeeze(0).squeeze(0)  # [H, W]
        overlay_sig[:, mask_sig > 0] = torch.tensor([1.0, 0, 0]).unsqueeze(1)
        
        original_tensor = transforms.ToTensor()(original_rgb)
        blended_sig = 0.7 * original_tensor + 0.3 * overlay_sig
        blended_sig_img = to_pil(blended_sig)
        blended_sig_img.save(output_dir / f"{base_name}_significant_overlay.jpg")
        
        print(f"✓ Saved significant element outputs")
        
        # ---- Tiny Elements ----
        if detect_tiny:
            # Save blurred (small kernel)
            blur_small_img = to_pil(blurred_small.squeeze(0))
            blur_small_img.save(output_dir / f"{base_name}_blur_small.jpg")
            
            # Save gradient (small)
            grad_norm_small = (gradient_small - gradient_small.min()) / (gradient_small.max() - gradient_small.min())
            grad_small_img = to_pil(grad_norm_small.squeeze(0))
            grad_small_img.save(output_dir / f"{base_name}_gradient_small.jpg")
            
            # Save I_R (raw thresholded from small blur)
            ir_img = to_pil(I_R.squeeze(0))
            ir_img.save(output_dir / f"{base_name}_I_R.jpg")
            
            # Save tiny elements binary mask
            tiny_img = to_pil(I_TE.squeeze(0))
            tiny_img.save(output_dir / f"{base_name}_tiny.jpg")
            
            # Create tiny elements overlay (blue)
            overlay_tiny = transforms.ToTensor()(original_rgb)  # [3, H, W]
            mask_tiny = I_TE.squeeze(0).squeeze(0)  # [H, W]
            overlay_tiny[:, mask_tiny > 0] = torch.tensor([0, 0.5, 1.0]).unsqueeze(1)  # Light blue
            
            blended_tiny = 0.7 * original_tensor + 0.3 * overlay_tiny
            blended_tiny_img = to_pil(blended_tiny)
            blended_tiny_img.save(output_dir / f"{base_name}_tiny_overlay.jpg")
            
            # Create combined overlay (red = significant, blue = tiny)
            overlay_combined = transforms.ToTensor()(original_rgb)
            overlay_combined[:, mask_sig > 0] = torch.tensor([1.0, 0, 0]).unsqueeze(1)  # Red
            overlay_combined[:, mask_tiny > 0] = torch.tensor([0, 0.5, 1.0]).unsqueeze(1)  # Blue
            
            blended_combined = 0.7 * original_tensor + 0.3 * overlay_combined
            blended_combined_img = to_pil(blended_combined)
            blended_combined_img.save(output_dir / f"{base_name}_combined_overlay.jpg")
            
            print(f"✓ Saved tiny element outputs")
            print(f"✓ Saved combined overlay (red=significant, blue=tiny)")
        
        print(f"\n✓ All outputs saved to: {output_dir}/")
    
    return results


def process_batch(data_dir, max_images=None, category='Proof', detect_tiny=True, output_root='davidlawrence_elements'):
    """Process multiple images from the dataset."""
    data_path = Path(data_dir) / category
    
    if not data_path.exists():
        print(f"❌ Error: {data_path} does not exist")
        return
    
    # Find all obverse images
    image_files = []
    for grade_dir in data_path.iterdir():
        if grade_dir.is_dir():
            obverse_dir = grade_dir / 'obverse'
            if obverse_dir.exists():
                image_files.extend(list(obverse_dir.glob('*.jpg')))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print("=" * 60)
    print(f"ELEMENT DETECTION - {category}")
    print("=" * 60)
    print(f"Total images to process: {len(image_files)}")
    print(f"Detecting: Significant{'+ Tiny' if detect_tiny else ''} elements")
    print("=" * 60)
    
    results = []
    for i, image_file in enumerate(image_files, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(image_files)}]")
        print('=' * 60)
        try:
            result = detect_significant_elements(image_file, save_output=True, detect_tiny=detect_tiny, output_root=output_root)
            results.append(result)
        except Exception as e:
            print(f"❌ Error processing {image_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Successfully processed: {len(results)}/{len(image_files)} images")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Detect significant elements in coin images (PyTorch version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image
  %(prog)s --image davidlawrence_dataset/Proof/ms65/obverse/image.jpg
  
  # Process first 10 Proof coins
  %(prog)s --dataset davidlawrence_dataset --category Proof --max 10
  
  # Process all Circulation coins
  %(prog)s --dataset davidlawrence_dataset --category Circulation
        """
    )
    
    parser.add_argument('--image', type=str,
                       help='Path to a single coin image')
    parser.add_argument('--dataset', type=str,
                       help='Path to dataset directory (contains Proof/Circulation)')
    parser.add_argument('--category', type=str, default='Proof',
                       choices=['Proof', 'Circulation'],
                       help='Category to process (default: Proof)')
    parser.add_argument('--max', type=int,
                       help='Maximum number of images to process')
    parser.add_argument('--no-tiny', action='store_true',
                       help='Skip tiny element detection (only detect significant elements)')
    parser.add_argument('--output', '-o', type=str, default='davidlawrence_elements',
                       help='Output root directory (default: davidlawrence_elements)')
    
    args = parser.parse_args()
    
    detect_tiny = not args.no_tiny
    
    if args.image:
        # Process single image
        detect_significant_elements(args.image, save_output=True, detect_tiny=detect_tiny, output_root=args.output)
    elif args.dataset:
        # Process dataset
        process_batch(args.dataset, max_images=args.max, category=args.category, detect_tiny=detect_tiny, output_root=args.output)
    else:
        print("❌ Error: Must specify either --image or --dataset")
        parser.print_help()


if __name__ == '__main__':
    main()

