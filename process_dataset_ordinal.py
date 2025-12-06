"""
Process Entire David Lawrence Dataset with Ordinal Regression

Loads a trained ordinal regression model and evaluates it on
the entire davidlawrence_dataset directory (both Proof and Circulation).

Provides:
- Per-coin predictions with confidence
- MAE, ±1, ±2 accuracy metrics
- Per-grade and per-company statistics
- CSV/JSON output of all predictions

Usage:
    python process_dataset_ordinal.py --model models/coin_ordinal_best.pth --output results.csv
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

import os
import json
import csv
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import re


# ============================================================================
# HELPER FUNCTIONS (Step-based encoding aware)
# ============================================================================

# Defaults (will be overridden by model checkpoint)
VALID_GRADES = [2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45,
                50, 53, 55, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68]
NUM_STEPS = len(VALID_GRADES)
GRADE_TO_STEP = {grade: step for step, grade in enumerate(VALID_GRADES)}
STEP_TO_GRADE = {step: grade for step, grade in enumerate(VALID_GRADES)}
ENCODING_TYPE = 'sheldon'
GRADE_MIN = 1.0
GRADE_MAX = 70.0


def parse_sheldon_grade(grade_str):
    """Convert grade string to numeric Sheldon scale."""
    match = re.search(r'(\d+)', grade_str.lower())
    if match:
        return int(match.group(1))
    
    grade_map = {
        'poor': 1, 'fr': 2, 'ag': 3, 'g': 4, 'vg': 8,
        'f': 12, 'vf': 20, 'xf': 40, 'au': 50, 'ms': 60
    }
    
    grade_lower = grade_str.lower()
    for key, val in grade_map.items():
        if key in grade_lower:
            return val
    return 50


def round_to_valid_grade(grade):
    """Round to nearest valid Sheldon grade."""
    return min(VALID_GRADES, key=lambda x: abs(x - grade))


def step_to_sheldon(step):
    """Convert step position back to Sheldon grade."""
    step = int(round(step))
    step = max(0, min(step, NUM_STEPS - 1))
    return STEP_TO_GRADE[step]


def denormalize_grade(normalized_grade):
    """Convert normalized grade back to Sheldon scale (encoding-aware)."""
    if ENCODING_TYPE == 'step_based':
        if isinstance(normalized_grade, torch.Tensor):
            steps = normalized_grade * (NUM_STEPS - 1)
            result = torch.zeros_like(steps)
            for i, s in enumerate(steps):
                result[i] = step_to_sheldon(s.item())
            return result
        else:
            step = normalized_grade * (NUM_STEPS - 1)
            return step_to_sheldon(step)
    else:
        if isinstance(normalized_grade, torch.Tensor):
            return normalized_grade * (GRADE_MAX - GRADE_MIN) + GRADE_MIN
        return normalized_grade * (GRADE_MAX - GRADE_MIN) + GRADE_MIN


def format_grade_name(sheldon_grade):
    """Convert Sheldon number to grade name."""
    if sheldon_grade >= 60:
        return f"MS{sheldon_grade}"
    elif sheldon_grade >= 50:
        return f"AU{sheldon_grade}"
    elif sheldon_grade >= 40:
        return f"XF{sheldon_grade}"
    elif sheldon_grade >= 20:
        return f"VF{sheldon_grade}"
    elif sheldon_grade >= 8:
        return f"VG{sheldon_grade:02d}"
    else:
        return f"G{sheldon_grade:02d}"


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class OrdinalRegressionResNet(nn.Module):
    """ResNet for ordinal regression (must match training)."""
    
    def __init__(self, num_companies=None, company_embedding_dim=32, freeze_backbone=False):
        super(OrdinalRegressionResNet, self).__init__()
        
        self.use_company = num_companies is not None
        
        if self.use_company:
            self.company_embedding = nn.Embedding(num_companies, company_embedding_dim)
        
        weights = ResNet50_Weights.IMAGENET1K_V2
        obverse_resnet = resnet50(weights=weights)
        reverse_resnet = resnet50(weights=weights)
        
        self.obverse_encoder = nn.Sequential(*list(obverse_resnet.children())[:-1])
        self.reverse_encoder = nn.Sequential(*list(reverse_resnet.children())[:-1])
        
        self.feature_dim = 2048
        
        fusion_input_dim = self.feature_dim * 2
        if self.use_company:
            fusion_input_dim += company_embedding_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, obverse, reverse, company_idx=None):
        obverse_feat = self.obverse_encoder(obverse).view(obverse.size(0), -1)
        reverse_feat = self.reverse_encoder(reverse).view(reverse.size(0), -1)
        
        combined = torch.cat([obverse_feat, reverse_feat], dim=1)
        
        if self.use_company and company_idx is not None:
            company_emb = self.company_embedding(company_idx)
            combined = torch.cat([combined, company_emb], dim=1)
        
        fused = self.fusion(combined)
        output = self.regression_head(fused).squeeze(-1)
        
        return output


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def load_model(model_path, device):
    """Load trained ordinal regression model."""
    print(f"\nLoading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    use_company = checkpoint.get('use_company', False)
    company_to_idx = checkpoint.get('company_to_idx')
    idx_to_company = checkpoint.get('idx_to_company')
    company_embedding_dim = checkpoint.get('company_embedding_dim', 32)
    
    # Detect encoding type and set global variables
    global VALID_GRADES, NUM_STEPS, GRADE_TO_STEP, STEP_TO_GRADE, ENCODING_TYPE, GRADE_MIN, GRADE_MAX
    
    ENCODING_TYPE = checkpoint.get('encoding', 'sheldon')
    
    if ENCODING_TYPE == 'step_based':
        VALID_GRADES = checkpoint.get('valid_grades', VALID_GRADES)
        NUM_STEPS = checkpoint.get('num_steps', len(VALID_GRADES))
        GRADE_TO_STEP = {grade: step for step, grade in enumerate(VALID_GRADES)}
        STEP_TO_GRADE = {step: grade for step, grade in enumerate(VALID_GRADES)}
        print(f"  Encoding: STEP-BASED ({NUM_STEPS} steps)")
    else:
        GRADE_MIN = checkpoint.get('grade_min', 1.0)
        GRADE_MAX = checkpoint.get('grade_max', 70.0)
        print(f"  Encoding: SHELDON-BASED ({GRADE_MIN}-{GRADE_MAX})")
    
    num_companies = len(company_to_idx) if use_company else None
    
    model = OrdinalRegressionResNet(
        num_companies=num_companies,
        company_embedding_dim=company_embedding_dim,
        freeze_backbone=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    mae_unit = "steps" if ENCODING_TYPE == 'step_based' else "grades"
    print(f"✓ Model loaded (Epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"  Best val MAE: {checkpoint.get('val_mae', 'unknown'):.2f} {mae_unit}")
    print(f"  Company conditioning: {use_company}")
    if use_company:
        print(f"  Companies: {list(company_to_idx.keys())}")
    
    return model, use_company, company_to_idx, idx_to_company


def process_dataset(dataset_root, model, device, use_company, company_to_idx, transform):
    """Process entire dataset and make predictions."""
    dataset_path = Path(dataset_root)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")
    
    # Find all subdirectories (Proof, Circulation, etc.)
    root_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    print(f"\nProcessing dataset: {dataset_root}")
    print(f"Found {len(root_dirs)} root directories: {[d.name for d in root_dirs]}")
    
    all_results = []
    
    for root_dir in root_dirs:
        print(f"\n{'='*70}")
        print(f"Processing: {root_dir.name}")
        print(f"{'='*70}")
        
        # Find all grade folders
        grade_folders = sorted([d for d in root_dir.iterdir() if d.is_dir()])
        
        for grade_folder in tqdm(grade_folders, desc=f"Grades in {root_dir.name}"):
            true_grade_name = grade_folder.name
            true_grade_value = parse_sheldon_grade(true_grade_name)
            
            obverse_dir = grade_folder / 'obverse'
            reverse_dir = grade_folder / 'reverse'
            
            if not obverse_dir.exists() or not reverse_dir.exists():
                continue
            
            obverse_images = sorted([f for f in obverse_dir.glob('*.jpg') if f.is_file()])
            
            for obverse_img in obverse_images:
                reverse_img = reverse_dir / obverse_img.name
                
                if not reverse_img.exists():
                    continue
                
                # Parse metadata from filename
                # Format: <grade>-<company>-<year>-<denom>-<cert>.jpg
                parts = obverse_img.stem.split('-')
                company = parts[1] if len(parts) >= 2 else 'UNKNOWN'
                year = parts[2] if len(parts) >= 3 else 'unknown'
                cert = parts[4] if len(parts) >= 5 else 'unknown'
                
                # Skip unwanted companies
                if company in ['OTHE', 'THAT']:
                    continue
                
                # Load images
                try:
                    obv_image = Image.open(obverse_img).convert('RGB')
                    rev_image = Image.open(reverse_img).convert('RGB')
                    
                    obv_tensor = transform(obv_image).unsqueeze(0).to(device)
                    rev_tensor = transform(rev_image).unsqueeze(0).to(device)
                    
                    # Prepare company index
                    company_idx = None
                    if use_company and company in company_to_idx:
                        company_idx = torch.tensor([company_to_idx[company]], device=device)
                    
                    # Predict
                    with torch.no_grad():
                        normalized_pred = model(obv_tensor, rev_tensor, company_idx)
                        continuous_grade = denormalize_grade(normalized_pred.float().cpu()).item()
                        rounded_grade = round_to_valid_grade(continuous_grade)
                        predicted_grade_name = format_grade_name(rounded_grade)
                    
                    # Calculate error
                    grade_error = abs(continuous_grade - true_grade_value)
                    
                    # Store result
                    result = {
                        'dataset': root_dir.name,
                        'true_grade': true_grade_name,
                        'true_sheldon': true_grade_value,
                        'predicted_grade': predicted_grade_name,
                        'predicted_sheldon': rounded_grade,
                        'continuous_prediction': continuous_grade,
                        'error': grade_error,
                        'company': company,
                        'year': year,
                        'cert': cert,
                        'filename': obverse_img.stem,
                        'obverse_path': str(obverse_img.relative_to(dataset_path)),
                        'reverse_path': str(reverse_img.relative_to(dataset_path))
                    }
                    
                    all_results.append(result)
                
                except Exception as e:
                    print(f"Error processing {obverse_img.name}: {e}")
                    continue
    
    return all_results


def compute_metrics(results):
    """Compute evaluation metrics from results."""
    if not results:
        return {}
    
    # Overall metrics
    errors = [r['error'] for r in results]
    mae = np.mean(errors)
    
    # Create sorted list of unique grades for step counting
    all_grades = set([r['true_sheldon'] for r in results] + [r['predicted_sheldon'] for r in results])
    grade_order = sorted(all_grades)
    grade_to_position = {grade: pos for pos, grade in enumerate(grade_order)}
    
    # Count within ±1, ±2 steps
    within_1 = 0
    within_2 = 0
    exact = 0
    
    for r in results:
        if r['true_sheldon'] == r['predicted_sheldon']:
            exact += 1
        
        if r['true_sheldon'] in grade_to_position and r['predicted_sheldon'] in grade_to_position:
            true_pos = grade_to_position[r['true_sheldon']]
            pred_pos = grade_to_position[r['predicted_sheldon']]
            step_diff = abs(pred_pos - true_pos)
            
            if step_diff <= 1:
                within_1 += 1
            if step_diff <= 2:
                within_2 += 1
    
    total = len(results)
    
    # Per-grade metrics
    per_grade = defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': []})
    for r in results:
        grade = r['true_grade']
        per_grade[grade]['total'] += 1
        per_grade[grade]['errors'].append(r['error'])
        if r['true_sheldon'] == r['predicted_sheldon']:
            per_grade[grade]['correct'] += 1
    
    # Per-company metrics
    per_company = defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': []})
    for r in results:
        company = r['company']
        per_company[company]['total'] += 1
        per_company[company]['errors'].append(r['error'])
        if r['true_sheldon'] == r['predicted_sheldon']:
            per_company[company]['correct'] += 1
    
    return {
        'total_samples': total,
        'mae': mae,
        'exact_accuracy': 100.0 * exact / total,
        'within_1_step': 100.0 * within_1 / total,
        'within_2_steps': 100.0 * within_2 / total,
        'exact_count': exact,
        'within_1_count': within_1,
        'within_2_count': within_2,
        'per_grade': dict(per_grade),
        'per_company': dict(per_company)
    }


def print_metrics(metrics):
    """Print evaluation metrics."""
    print("\n" + "="*70)
    print("OVERALL METRICS")
    print("="*70)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"MAE: {metrics['mae']:.2f} Sheldon grades")
    print(f"Exact match: {metrics['exact_accuracy']:.2f}% ({metrics['exact_count']}/{metrics['total_samples']})")
    print(f"Within ±1 step: {metrics['within_1_step']:.2f}% ({metrics['within_1_count']}/{metrics['total_samples']})")
    print(f"Within ±2 steps: {metrics['within_2_steps']:.2f}% ({metrics['within_2_count']}/{metrics['total_samples']})")
    
    # Per-grade breakdown
    print("\n" + "="*70)
    print("PER-GRADE ACCURACY")
    print("="*70)
    print(f"{'Grade':<10} {'Exact':<8} {'MAE':<8} {'Total':<8}")
    print("-"*70)
    
    for grade in sorted(metrics['per_grade'].keys(), key=lambda g: parse_sheldon_grade(g)):
        stats = metrics['per_grade'][grade]
        acc = 100.0 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        mae = np.mean(stats['errors']) if stats['errors'] else 0
        print(f"{grade:<10} {acc:>6.1f}%  {mae:>6.2f}  {stats['total']:>6}")
    
    # Per-company breakdown
    if metrics['per_company']:
        print("\n" + "="*70)
        print("PER-COMPANY ACCURACY")
        print("="*70)
        print(f"{'Company':<10} {'Exact':<8} {'MAE':<8} {'Total':<8}")
        print("-"*70)
        
        for company in sorted(metrics['per_company'].keys()):
            stats = metrics['per_company'][company]
            acc = 100.0 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            mae = np.mean(stats['errors']) if stats['errors'] else 0
            print(f"{company:<10} {acc:>6.1f}%  {mae:>6.2f}  {stats['total']:>6}")


def save_results(results, metrics, output_file):
    """Save results to CSV or JSON."""
    output_path = Path(output_file)
    
    if output_path.suffix == '.csv':
        with open(output_path, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        print(f"\n✓ Results saved to: {output_file}")
    
    elif output_path.suffix == '.json':
        output_data = {
            'metrics': metrics,
            'predictions': results
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")
    
    else:
        print(f"\n⚠️  Unknown output format: {output_path.suffix}")
        print("   Supported formats: .csv, .json")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Process entire David Lawrence dataset with ordinal regression',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', type=str, default='models/coin_ordinal_best.pth',
                       help='Path to trained ordinal regression model')
    parser.add_argument('--dataset', type=str, default='davidlawrence_dataset/Circulation',
                       help='Path to dataset root directory')
    parser.add_argument('--output', type=str,
                       help='Save results to file (.csv or .json)')
    parser.add_argument('--image-size', type=int, default=448,
                       help='Image size for model input')
    
    args = parser.parse_args()
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("="*70)
    print("PROCESS DATASET WITH ORDINAL REGRESSION")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    
    # Load model
    model, use_company, company_to_idx, idx_to_company = load_model(args.model, device)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process dataset
    results = process_dataset(
        args.dataset, model, device, use_company, company_to_idx, transform
    )
    
    if not results:
        print("\n⚠️  No valid samples found!")
        return
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    # Print results
    print_metrics(metrics)
    
    # Save if requested
    if args.output:
        save_results(results, metrics, args.output)
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()


