#!/usr/bin/env python3
"""
Ordinal Regression Model Evaluation Script

Loads a trained ordinal regression model and evaluates it on a test set.
Shows expected grade vs predicted grade for each coin.

Usage:
    python evaluate_ordinal_model.py --model models/coin_ordinal_best.pth --data davidlawrence_dataset/Circulation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
import re


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default grade normalization (will be overridden by model checkpoint)
GRADE_MIN = 1.0
GRADE_MAX = 70.0

# Default valid grades (will be overridden by model checkpoint for step-based models)
VALID_GRADES = [2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45,
                50, 53, 55, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68]
NUM_STEPS = len(VALID_GRADES)
GRADE_TO_STEP = {grade: step for step, grade in enumerate(VALID_GRADES)}
STEP_TO_GRADE = {step: grade for step, grade in enumerate(VALID_GRADES)}

# Encoding type (will be set when loading model)
ENCODING_TYPE = 'sheldon'  # 'sheldon' or 'step_based'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_sheldon_grade(grade_str):
    """Convert grade string to numeric Sheldon scale."""
    match = re.search(r'(\d+)', grade_str.lower())
    if match:
        return float(match.group(1))
    
    grade_map = {
        'poor': 1, 'fr': 2, 'ag': 3, 'g': 4, 'vg': 8,
        'f': 12, 'vf': 20, 'xf': 40, 'au': 50, 'ms': 60
    }
    
    grade_lower = grade_str.lower()
    for key, val in grade_map.items():
        if key in grade_lower:
            return float(val)
    
    return 50.0


def round_to_valid_grade(grade):
    """Round to nearest valid Sheldon grade."""
    return min(VALID_GRADES, key=lambda x: abs(x - grade))


def sheldon_to_step(sheldon_grade):
    """Convert Sheldon grade to step position."""
    rounded = round_to_valid_grade(sheldon_grade)
    return GRADE_TO_STEP.get(rounded, NUM_STEPS // 2)


def step_to_sheldon(step):
    """Convert step position back to Sheldon grade."""
    step = int(round(step))
    step = max(0, min(step, NUM_STEPS - 1))
    return STEP_TO_GRADE[step]


def normalize_grade(sheldon_grade):
    """Normalize grade to [0, 1] range (encoding-aware)."""
    if ENCODING_TYPE == 'step_based':
        step = sheldon_to_step(sheldon_grade)
        return float(step / (NUM_STEPS - 1))
    else:
        return float((sheldon_grade - GRADE_MIN) / (GRADE_MAX - GRADE_MIN))


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
        return f"MS{int(sheldon_grade)}"
    elif sheldon_grade >= 50:
        return f"AU{int(sheldon_grade)}"
    elif sheldon_grade >= 40:
        return f"XF{int(sheldon_grade)}"
    elif sheldon_grade >= 20:
        return f"VF{int(sheldon_grade)}"
    elif sheldon_grade >= 8:
        return f"VG{int(sheldon_grade):02d}"
    elif sheldon_grade >= 4:
        return f"G{int(sheldon_grade):02d}"
    elif sheldon_grade == 3:
        return "AG03"
    elif sheldon_grade == 2:
        return "FR02"
    else:
        return f"P{int(sheldon_grade):02d}"


# ============================================================================
# MODEL ARCHITECTURE (Must match training)
# ============================================================================

class OrdinalRegressionResNet(nn.Module):
    """ResNet for ordinal regression."""
    
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
# DATASET CLASS
# ============================================================================

class EvaluationDataset(Dataset):
    """Dataset for ordinal regression evaluation."""
    
    def __init__(self, data_dir, transform=None, use_company=True, 
                 company_to_idx=None, test_split=0.2):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_company = use_company
        self.samples = []
        
        # Use model's company mapping if provided
        if company_to_idx is not None:
            self.company_to_idx = company_to_idx
        else:
            self.company_to_idx = {}
        
        # Collect samples
        temp_samples = []
        
        grade_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for grade_folder in grade_folders:
            grade_name = grade_folder.name
            grade_value = parse_sheldon_grade(grade_name)
            
            obverse_dir = grade_folder / 'obverse'
            reverse_dir = grade_folder / 'reverse'
            
            if not obverse_dir.exists() or not reverse_dir.exists():
                continue
            
            obverse_images = sorted([f for f in obverse_dir.glob('*.jpg') if f.is_file()])
            
            for obverse_img in obverse_images:
                reverse_img = reverse_dir / obverse_img.name
                
                if reverse_img.exists():
                    # Parse metadata from filename
                    parts = obverse_img.stem.split('-')
                    company = parts[1] if len(parts) >= 2 else 'UNKNOWN'
                    year = parts[2] if len(parts) >= 3 else 'unknown'
                    denom = parts[3] if len(parts) >= 4 else 'unknown'
                    cert = parts[4] if len(parts) >= 5 else 'unknown'
                    
                    # Skip unwanted companies
                    if company in ['OTHE', 'THAT']:
                        continue
                    
                    # Skip if company not in model's mapping
                    if self.use_company and company not in self.company_to_idx:
                        continue
                    
                    company_idx = self.company_to_idx.get(company, 0) if self.use_company else 0
                    
                    temp_samples.append({
                        'obverse': obverse_img,
                        'reverse': reverse_img,
                        'grade_name': grade_name,
                        'grade_value': grade_value,
                        'company': company,
                        'company_idx': company_idx,
                        'year': year,
                        'denomination': denom,
                        'cert_number': cert,
                        'filename': obverse_img.stem
                    })
        
        # 80/20 split - take last 20% as test
        np.random.seed(42)
        indices = np.random.permutation(len(temp_samples))
        n_test = int(test_split * len(temp_samples))
        test_indices = indices[-n_test:]
        
        self.samples = [temp_samples[i] for i in test_indices]
        
        # Statistics
        grade_counts = Counter([s['grade_name'] for s in self.samples])
        
        print(f"\nTest set: {len(self.samples)} samples")
        print(f"Unique grades: {len(grade_counts)}")
        print(f"Grade distribution:")
        for grade in sorted(grade_counts.keys(), key=lambda x: parse_sheldon_grade(x)):
            print(f"  {grade:8s}: {grade_counts[grade]:4d} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        obverse = Image.open(sample['obverse']).convert('RGB')
        reverse = Image.open(sample['reverse']).convert('RGB')
        
        if self.transform:
            obverse = self.transform(obverse)
            reverse = self.transform(reverse)
        
        # Normalize grade
        normalized_grade = torch.tensor(normalize_grade(sample['grade_value']), dtype=torch.float32)
        
        # Sample info (serializable)
        sample_info = {
            'grade_name': sample['grade_name'],
            'grade_value': sample['grade_value'],
            'company': sample['company'],
            'company_idx': sample['company_idx'],
            'year': sample['year'],
            'denomination': sample['denomination'],
            'cert_number': sample['cert_number'],
            'filename': sample['filename']
        }
        
        return obverse, reverse, normalized_grade, sample['company_idx'], sample_info


# ============================================================================
# EVALUATION FUNCTIONS
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
    global GRADE_MIN, GRADE_MAX, VALID_GRADES, NUM_STEPS, GRADE_TO_STEP, STEP_TO_GRADE, ENCODING_TYPE
    
    ENCODING_TYPE = checkpoint.get('encoding', 'sheldon')
    
    if ENCODING_TYPE == 'step_based':
        # Step-based encoding: use valid_grades from checkpoint
        VALID_GRADES = checkpoint.get('valid_grades', VALID_GRADES)
        NUM_STEPS = checkpoint.get('num_steps', len(VALID_GRADES))
        GRADE_TO_STEP = {grade: step for step, grade in enumerate(VALID_GRADES)}
        STEP_TO_GRADE = {step: grade for step, grade in enumerate(VALID_GRADES)}
        print(f"  Encoding: STEP-BASED ({NUM_STEPS} steps)")
    else:
        # Old Sheldon-based encoding
        GRADE_MIN = checkpoint.get('grade_min', 1.0)
        GRADE_MAX = checkpoint.get('grade_max', 70.0)
        print(f"  Encoding: SHELDON-BASED ({GRADE_MIN}-{GRADE_MAX})")
    
    num_companies = len(company_to_idx) if use_company and company_to_idx else None
    
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
    print(f"  Regression type: {checkpoint.get('regression_type', 'unknown')}")
    print(f"  Best val MAE: {checkpoint.get('val_mae', 'unknown'):.2f} {mae_unit}")
    print(f"  Company conditioning: {use_company}")
    if use_company and company_to_idx:
        print(f"  Companies: {list(company_to_idx.keys())}")
    print(f"  Grade range: {GRADE_MIN} - {GRADE_MAX}")
    
    return model, use_company, company_to_idx, idx_to_company


def evaluate_model(model, dataloader, device, use_company, idx_to_company=None):
    """Evaluate ordinal regression model."""
    all_predictions = []
    all_targets = []
    all_samples = []
    
    model.eval()
    
    print("\n" + "="*90)
    print("EVALUATING ORDINAL REGRESSION MODEL")
    print("="*90)
    
    # Valid grades for step counting
    valid_grades = [2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45,
                   50, 53, 55, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68]
    grade_to_position = {g: i for i, g in enumerate(valid_grades)}
    
    with torch.no_grad():
        for obverse, reverse, targets, company_idx, samples_batch in tqdm(dataloader, desc="Evaluating"):
            batch_size = len(targets)
            
            obverse = obverse.to(device)
            reverse = reverse.to(device)
            targets = targets.to(device, dtype=torch.float32)
            
            if use_company:
                company_idx = company_idx.to(device)
            else:
                company_idx = None
            
            # Forward pass
            predictions = model(obverse, reverse, company_idx)
            
            # Denormalize predictions
            pred_grades = denormalize_grade(predictions.float().cpu())
            true_grades = denormalize_grade(targets.float().cpu())
            
            # Round to valid grades
            pred_rounded = [round_to_valid_grade(p.item()) for p in pred_grades]
            true_rounded = [round_to_valid_grade(t.item()) for t in true_grades]
            
            # Reconstruct sample dicts
            for i in range(batch_size):
                sample = {
                    'grade_name': samples_batch['grade_name'][i],
                    'grade_value': samples_batch['grade_value'][i].item(),
                    'company': samples_batch['company'][i],
                    'year': samples_batch['year'][i],
                    'denomination': samples_batch['denomination'][i],
                    'cert_number': samples_batch['cert_number'][i],
                    'filename': samples_batch['filename'][i],
                    'predicted_continuous': pred_grades[i].item(),
                    'predicted_rounded': pred_rounded[i],
                    'predicted_grade_name': format_grade_name(pred_rounded[i]).lower()
                }
                all_samples.append(sample)
            
            all_predictions.extend(pred_rounded)
            all_targets.extend(true_rounded)
    
    # Calculate metrics
    total = len(all_predictions)
    
    # Exact match
    exact_match = sum([p == t for p, t in zip(all_predictions, all_targets)])
    
    # MAE in Sheldon grade numbers (raw points)
    mae_sheldon = np.mean([abs(p - t) for p, t in zip(all_predictions, all_targets)])
    
    # MAE in grading scale STEPS (more accurate representation)
    # e.g., VF25→VF30 = 1 step, MS64→MS65 = 1 step
    step_errors = []
    within_1 = 0
    within_2 = 0
    
    for pred, true in zip(all_predictions, all_targets):
        pred_pos = grade_to_position.get(pred)
        true_pos = grade_to_position.get(true)
        
        if pred_pos is not None and true_pos is not None:
            step_diff = abs(pred_pos - true_pos)
            step_errors.append(step_diff)
            if step_diff <= 1:
                within_1 += 1
            if step_diff <= 2:
                within_2 += 1
    
    mae_steps = np.mean(step_errors) if step_errors else 0
    
    # Per-grade metrics
    per_grade = defaultdict(lambda: {'predictions': [], 'errors_sheldon': [], 'errors_steps': [], 'count': 0, 'exact': 0})
    for sample, pred, true in zip(all_samples, all_predictions, all_targets):
        grade = sample['grade_name']
        per_grade[grade]['predictions'].append(pred)
        per_grade[grade]['errors_sheldon'].append(abs(pred - true))
        
        # Calculate step error for this sample
        pred_pos = grade_to_position.get(pred)
        true_pos = grade_to_position.get(true)
        if pred_pos is not None and true_pos is not None:
            per_grade[grade]['errors_steps'].append(abs(pred_pos - true_pos))
        
        per_grade[grade]['count'] += 1
        if pred == true:
            per_grade[grade]['exact'] += 1
    
    # Per-company metrics
    per_company = defaultdict(lambda: {'predictions': [], 'errors_sheldon': [], 'errors_steps': [], 'count': 0, 'exact': 0})
    for sample, pred, true in zip(all_samples, all_predictions, all_targets):
        company = sample['company']
        per_company[company]['predictions'].append(pred)
        per_company[company]['errors_sheldon'].append(abs(pred - true))
        
        # Calculate step error for this sample
        pred_pos = grade_to_position.get(pred)
        true_pos = grade_to_position.get(true)
        if pred_pos is not None and true_pos is not None:
            per_company[company]['errors_steps'].append(abs(pred_pos - true_pos))
        
        per_company[company]['count'] += 1
        if pred == true:
            per_company[company]['exact'] += 1
    
    # Print results
    print("\n" + "="*90)
    print("RESULTS")
    print("="*90)
    print(f"Total samples: {total}")
    print(f"")
    print(f"MAE (Steps):   {mae_steps:.2f} steps   ← More accurate (consistent across grade ranges)")
    print(f"MAE (Sheldon): {mae_sheldon:.2f} grades  (raw Sheldon point difference)")
    print(f"")
    print(f"Exact match:   {100.0 * exact_match / total:.2f}% ({exact_match}/{total})")
    print(f"Within ±1 step: {100.0 * within_1 / total:.2f}% ({within_1}/{total})")
    print(f"Within ±2 steps: {100.0 * within_2 / total:.2f}% ({within_2}/{total})")
    
    print("\n" + "-"*90)
    print("PER-GRADE ACCURACY")
    print("-"*90)
    print(f"{'Grade':<10} {'Exact':<10} {'MAE(Steps)':<12} {'MAE(Sheldon)':<14} {'Total':<8}")
    print("-"*90)
    
    for grade in sorted(per_grade.keys(), key=lambda x: parse_sheldon_grade(x)):
        stats = per_grade[grade]
        exact_pct = 100.0 * stats['exact'] / stats['count'] if stats['count'] > 0 else 0
        grade_mae_steps = np.mean(stats['errors_steps']) if stats['errors_steps'] else 0
        grade_mae_sheldon = np.mean(stats['errors_sheldon']) if stats['errors_sheldon'] else 0
        print(f"{grade:<10} {exact_pct:>6.1f}%    {grade_mae_steps:>8.2f}      {grade_mae_sheldon:>8.2f}       {stats['count']:>6}")
    
    print("\n" + "-"*90)
    print("PER-COMPANY ACCURACY")
    print("-"*90)
    print(f"{'Company':<10} {'Exact':<10} {'MAE(Steps)':<12} {'MAE(Sheldon)':<14} {'Total':<8}")
    print("-"*90)
    
    for company in sorted(per_company.keys()):
        stats = per_company[company]
        exact_pct = 100.0 * stats['exact'] / stats['count'] if stats['count'] > 0 else 0
        company_mae_steps = np.mean(stats['errors_steps']) if stats['errors_steps'] else 0
        company_mae_sheldon = np.mean(stats['errors_sheldon']) if stats['errors_sheldon'] else 0
        print(f"{company:<10} {exact_pct:>6.1f}%    {company_mae_steps:>8.2f}      {company_mae_sheldon:>8.2f}       {stats['count']:>6}")
    
    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'samples': all_samples,
        'mae_steps': mae_steps,
        'mae_sheldon': mae_sheldon,
        'exact_match': exact_match,
        'within_1': within_1,
        'within_2': within_2,
        'total': total,
        'per_grade': dict(per_grade),
        'per_company': dict(per_company)
    }


def print_detailed_results(results, show_errors_only=False, max_show=50):
    """Print detailed per-sample results."""
    samples = results['samples']
    predictions = results['predictions']
    targets = results['targets']
    
    print("\n" + "="*100)
    if show_errors_only:
        print("INCORRECT PREDICTIONS")
    else:
        print(f"DETAILED PREDICTIONS (showing first {max_show})")
    print("="*100)
    print(f"{'Filename':<40} {'Expected':<10} {'Predicted':<10} {'Diff':<8} {'Status':<7} {'Company':<8}")
    print("-"*100)
    
    shown = 0
    for sample, pred, true in zip(samples, predictions, targets):
        is_correct = pred == true
        
        if show_errors_only and is_correct:
            continue
        
        if shown >= max_show:
            break
        
        expected_name = format_grade_name(true)
        predicted_name = format_grade_name(pred)
        diff = pred - true
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        status = "✓" if is_correct else "✗"
        
        filename = sample['filename'][:40]
        company = sample['company']
        
        print(f"{filename:<40} {expected_name:<10} {predicted_name:<10} {diff_str:<8} {status:<7} {company:<8}")
        shown += 1
    
    if show_errors_only:
        print(f"\nTotal errors: {shown}")
    else:
        remaining = len(samples) - shown
        if remaining > 0:
            print(f"\n... and {remaining} more samples")


def save_results(results, output_file):
    """Save evaluation results to JSON."""
    output_data = {
        'metrics': {
            'total_samples': results['total'],
            'mae_steps': results['mae_steps'],
            'mae_sheldon': results['mae_sheldon'],
            'exact_match_pct': 100.0 * results['exact_match'] / results['total'],
            'within_1_step_pct': 100.0 * results['within_1'] / results['total'],
            'within_2_steps_pct': 100.0 * results['within_2'] / results['total'],
            'exact_match_count': results['exact_match'],
            'within_1_count': results['within_1'],
            'within_2_count': results['within_2']
        },
        'per_grade': {
            grade: {
                'count': stats['count'],
                'exact': stats['exact'],
                'exact_pct': 100.0 * stats['exact'] / stats['count'] if stats['count'] > 0 else 0,
                'mae_steps': float(np.mean(stats['errors_steps'])) if stats['errors_steps'] else 0,
                'mae_sheldon': float(np.mean(stats['errors_sheldon'])) if stats['errors_sheldon'] else 0
            }
            for grade, stats in results['per_grade'].items()
        },
        'per_company': {
            company: {
                'count': stats['count'],
                'exact': stats['exact'],
                'exact_pct': 100.0 * stats['exact'] / stats['count'] if stats['count'] > 0 else 0,
                'mae_steps': float(np.mean(stats['errors_steps'])) if stats['errors_steps'] else 0,
                'mae_sheldon': float(np.mean(stats['errors_sheldon'])) if stats['errors_sheldon'] else 0
            }
            for company, stats in results['per_company'].items()
        },
        'predictions': [
            {
                'filename': sample['filename'],
                'expected_grade': sample['grade_name'],
                'expected_sheldon': sample['grade_value'],
                'predicted_grade': sample['predicted_grade_name'],
                'predicted_sheldon': sample['predicted_rounded'],
                'predicted_continuous': sample['predicted_continuous'],
                'company': sample['company'],
                'year': sample['year'],
                'cert_number': sample['cert_number'],
                'error_sheldon': abs(sample['predicted_rounded'] - sample['grade_value']),
                'correct': sample['predicted_rounded'] == round_to_valid_grade(sample['grade_value'])
            }
            for sample in results['samples']
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ordinal regression model on test set',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', type=str, default='models/coin_ordinal_best-12-06-2025.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, default='davidlawrence_dataset/Circulation',
                       help='Path to dataset directory')
    parser.add_argument('--image-size', type=int, default=448,
                       help='Image size for model input')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--errors-only', action='store_true',
                       help='Show only incorrect predictions')
    parser.add_argument('--save', type=str,
                       help='Save results to JSON file')
    parser.add_argument('--max-show', type=int, default=50,
                       help='Maximum number of detailed predictions to show')
    
    args = parser.parse_args()
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("="*90)
    print("ORDINAL REGRESSION MODEL EVALUATION")
    print("="*90)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    
    # Load model
    model, use_company, company_to_idx, idx_to_company = load_model(args.model, device)
    
    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = EvaluationDataset(
        args.data,
        transform=transform,
        use_company=use_company,
        company_to_idx=company_to_idx,
        test_split=0.2
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Evaluate
    results = evaluate_model(model, dataloader, device, use_company, idx_to_company)
    
    # Print detailed results
    print_detailed_results(results, show_errors_only=args.errors_only, max_show=args.max_show)
    
    # Save if requested
    if args.save:
        save_results(results, args.save)
    
    print("\n" + "="*90)
    print("EVALUATION COMPLETE")
    print("="*90)


if __name__ == '__main__':
    main()

