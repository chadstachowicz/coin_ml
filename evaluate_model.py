#!/usr/bin/env python3
"""
Model Evaluation Script

Loads a trained dual-image model and evaluates it on a test set.
Shows expected grade vs predicted grade for each coin.

Usage:
    python evaluate_model.py --model models/coin_resnet_dual_best.pth --data davidlawrence_coins_indians
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
import matplotlib.pyplot as plt
import seaborn as sns
import re


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_sheldon_grade(grade_str):
    """
    Convert grade string to numeric Sheldon scale.
    
    Examples:
        'ms64' -> 64
        'au58' -> 58
        'xf40' -> 40
        'vf20' -> 20
        'g04' -> 4
    """
    # Extract the numeric part
    match = re.search(r'(\d+)', grade_str.lower())
    if match:
        return int(match.group(1))
    
    # Fallback mapping for non-standard grades
    grade_map = {
        'poor': 1, 'fr': 2, 'ag': 3, 'g': 4, 'vg': 8,
        'f': 12, 'vf': 20, 'xf': 40, 'au': 50, 'ms': 60
    }
    
    grade_lower = grade_str.lower()
    for key, val in grade_map.items():
        if key in grade_lower:
            return val
    
    # Default
    return 50


# ============================================================================
# MODEL ARCHITECTURE (Must match training)
# ============================================================================

class DualResNetClassifier(nn.Module):
    """ResNet-50 for dual-image classification."""
    
    def __init__(self, num_classes, freeze_backbone=False):
        super(DualResNetClassifier, self).__init__()
        
        # Load pretrained ResNet-50
        weights = ResNet50_Weights.IMAGENET1K_V2
        obverse_resnet = resnet50(weights=weights)
        reverse_resnet = resnet50(weights=weights)
        
        # Remove final FC layer
        self.obverse_encoder = nn.Sequential(*list(obverse_resnet.children())[:-1])
        self.reverse_encoder = nn.Sequential(*list(reverse_resnet.children())[:-1])
        
        self.feature_dim = 2048
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, obverse, reverse):
        obverse_feat = self.obverse_encoder(obverse).view(obverse.size(0), -1)
        reverse_feat = self.reverse_encoder(reverse).view(reverse.size(0), -1)
        combined = torch.cat([obverse_feat, reverse_feat], dim=1)
        fused = self.fusion(combined)
        output = self.classifier(fused)
        return output


# ============================================================================
# DATASET CLASS
# ============================================================================

class EvaluationDataset(Dataset):
    """Dataset for evaluation from JSON metadata and images."""
    
    def __init__(self, data_dir, images_dir, transform=None, test_split=0.2, 
                 model_class_to_idx=None):
        """
        Args:
            data_dir: Directory with JSON metadata files
            images_dir: Directory with coin images
            transform: Image transforms
            test_split: Fraction of data to use for testing (0.2 = 20%)
            model_class_to_idx: Class mapping from trained model (IMPORTANT!)
        """
        self.data_dir = Path(data_dir)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.samples = []
        
        # Use model's class mapping if provided
        if model_class_to_idx is not None:
            self.class_to_idx = model_class_to_idx
            self.idx_to_class = {v: k for k, v in model_class_to_idx.items()}
            print(f"Using model's class mapping: {len(self.class_to_idx)} classes")
        else:
            self.class_to_idx = {}
            self.idx_to_class = {}
        
        # Collect all samples from JSON files
        all_samples = []
        skipped_grades = set()
        
        json_files = sorted(list(self.data_dir.glob('*.json')))
        print(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    coin_data = json.load(f)
                
                grade = coin_data.get('grade')
                if not grade:
                    continue
                
                # Normalize grade
                grade = self.normalize_grade(grade)
                
                # Skip if grade not in model's classes
                if model_class_to_idx is not None and grade not in self.class_to_idx:
                    skipped_grades.add(grade)
                    continue
                
                # Find obverse and reverse images
                inventory_id = coin_data.get('inventory_id')
                obverse_img = self.images_dir / f"{inventory_id}_image_2.jpg"
                reverse_img = self.images_dir / f"{inventory_id}_image_3.jpg"
                
                if obverse_img.exists() and reverse_img.exists():
                    all_samples.append({
                        'obverse': obverse_img,
                        'reverse': reverse_img,
                        'grade': grade,
                        'inventory_id': inventory_id,
                        'cert_number': coin_data.get('cert_number', 'unknown'),
                        'year': coin_data.get('year', 'unknown'),
                        'grading_service': coin_data.get('grading_service', 'unknown')
                    })
            except Exception as e:
                print(f"Error loading {json_file.name}: {e}")
                continue
        
        if skipped_grades:
            print(f"\n⚠️  Skipped {len(skipped_grades)} grades not in model: {sorted(skipped_grades)}")
        
        # Split data (use last 20% as test set)
        np.random.seed(42)
        indices = np.random.permutation(len(all_samples))
        
        n_test = int(test_split * len(all_samples))
        test_indices = indices[-n_test:]  # Last 20%
        
        self.samples = [all_samples[i] for i in test_indices]
        
        # Add label indices
        for sample in self.samples:
            sample['label'] = self.class_to_idx[sample['grade']]
        
        print(f"\nTest set: {len(self.samples)} samples")
        print(f"Classes: {len(self.class_to_idx)}")
        print(f"Grade distribution:")
        grade_counts = Counter([s['grade'] for s in self.samples])
        for grade in sorted(self.class_to_idx.keys()):
            count = grade_counts.get(grade, 0)
            if count > 0:
                print(f"  {grade:8s}: {count:4d} samples")
    
    def normalize_grade(self, grade_str):
        """Normalize grade string."""
        if not grade_str:
            return 'unknown'
        
        grade_str = str(grade_str).strip().upper()
        
        # If already has prefix, lowercase it
        if not grade_str[0].isdigit():
            return grade_str.lower()
        
        # Convert number to grade designation
        try:
            grade_num = int(grade_str)
            
            if grade_num <= 2:
                return f'p{grade_num:02d}'
            elif grade_num == 2:
                return 'fr02'
            elif grade_num == 3:
                return 'ag03'
            elif 4 <= grade_num <= 6:
                return f'g{grade_num:02d}'
            elif 8 <= grade_num <= 10:
                return f'vg{grade_num:02d}'
            elif 12 <= grade_num <= 15:
                return f'f{grade_num:02d}'
            elif 20 <= grade_num <= 35:
                return f'vf{grade_num:02d}'
            elif 40 <= grade_num <= 45:
                return f'xf{grade_num:02d}'
            elif 50 <= grade_num <= 58:
                return f'au{grade_num:02d}'
            elif 60 <= grade_num <= 70:
                return f'ms{grade_num:02d}'
            else:
                return grade_str.lower()
        except ValueError:
            return grade_str.lower()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images
        obverse = Image.open(sample['obverse']).convert('RGB')
        reverse = Image.open(sample['reverse']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            obverse = self.transform(obverse)
            reverse = self.transform(reverse)
        
        # Create serializable sample info (no Path objects)
        sample_info = {
            'grade': sample['grade'],
            'label': sample['label'],
            'inventory_id': sample['inventory_id'],
            'cert_number': sample['cert_number'],
            'year': sample['year'],
            'grading_service': sample.get('grading_service', 'unknown')
        }
        
        return obverse, reverse, sample['label'], sample_info


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def load_model(model_path, device):
    """Load trained model from checkpoint."""
    print(f"\nLoading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get class mappings
    class_to_idx = checkpoint.get('class_to_idx', {})
    idx_to_class = checkpoint.get('idx_to_class', {})
    
    # Convert idx keys to int if they're strings
    if idx_to_class and isinstance(list(idx_to_class.keys())[0], str):
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    
    num_classes = len(class_to_idx)
    print(f"Model trained on {num_classes} classes")
    
    # Create model
    model = DualResNetClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded (Epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"  Training val_acc: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    
    return model, class_to_idx, idx_to_class


def evaluate_model(model, dataloader, device, idx_to_class):
    """Evaluate model and return detailed results."""
    all_predictions = []
    all_labels = []
    all_samples = []
    all_probs = []
    
    model.eval()
    
    print("\n" + "="*80)
    print("EVALUATING MODEL")
    print("="*80)
    
    with torch.no_grad():
        for obverse, reverse, labels, samples_batch in tqdm(dataloader, desc="Evaluating"):
            batch_size = len(labels)
            
            # Move to device
            obverse = obverse.to(device)
            reverse = reverse.to(device)
            labels = labels.to(device)
            
            # Reconstruct sample dicts
            batch_samples = []
            for i in range(batch_size):
                batch_samples.append({
                    'grade': samples_batch['grade'][i],
                    'label': samples_batch['label'][i],
                    'inventory_id': samples_batch['inventory_id'][i],
                    'cert_number': samples_batch['cert_number'][i],
                    'year': samples_batch['year'][i],
                    'grading_service': samples_batch['grading_service'][i]
                })
            
            # Forward pass
            outputs = model(obverse, reverse)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_samples.extend(batch_samples)
    
    # Calculate metrics
    correct = sum([p == l for p, l in zip(all_predictions, all_labels)])
    total = len(all_labels)
    accuracy = 100.0 * correct / total
    
    # Per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for pred, label in zip(all_predictions, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    # Calculate ±1 and ±2 grade accuracy
    # This measures "steps" in the grading scale, not Sheldon points
    # e.g., VF25 → VF30 is 1 step, MS64 → MS65 is 1 step
    within_1_grade = 0
    within_2_grades = 0
    
    # Create sorted list of class indices by Sheldon grade
    # This gives us the "grading scale" order
    class_order = sorted(class_total.keys(), 
                        key=lambda idx: parse_sheldon_grade(idx_to_class.get(idx, 'unknown')))
    
    # Create a mapping: class_idx → position in grading scale
    class_to_position = {class_idx: pos for pos, class_idx in enumerate(class_order)}
    
    for pred, label in zip(all_predictions, all_labels):
        if pred in class_to_position and label in class_to_position:
            # Distance in "grading steps" (how many grades apart in the actual scale)
            pred_pos = class_to_position[pred]
            label_pos = class_to_position[label]
            grade_step_diff = abs(pred_pos - label_pos)
            
            if grade_step_diff <= 1:
                within_1_grade += 1
            if grade_step_diff <= 2:
                within_2_grades += 1
    
    accuracy_within_1 = 100.0 * within_1_grade / total if total > 0 else 0
    accuracy_within_2 = 100.0 * within_2_grades / total if total > 0 else 0
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Within ±1 Grade:  {accuracy_within_1:.2f}% ({within_1_grade}/{total})  [1 step in grading scale]")
    print(f"Within ±2 Grades: {accuracy_within_2:.2f}% ({within_2_grades}/{total})  [2 steps in grading scale]")
    print("\nPer-Class Accuracy:")
    print("-"*80)
    
    for idx in sorted(class_total.keys()):
        grade = idx_to_class.get(idx, f'class_{idx}')
        acc = 100.0 * class_correct[idx] / class_total[idx] if class_total[idx] > 0 else 0
        print(f"  {grade:8s}: {acc:5.1f}% ({class_correct[idx]:3d}/{class_total[idx]:3d})")
    
    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'probs': all_probs,
        'samples': all_samples,
        'accuracy': accuracy,
        'accuracy_within_1': accuracy_within_1,
        'accuracy_within_2': accuracy_within_2,
        'within_1_grade': within_1_grade,
        'within_2_grades': within_2_grades,
        'class_correct': class_correct,
        'class_total': class_total
    }


def print_detailed_results(results, idx_to_class, show_errors_only=False, max_show=50):
    """Print detailed per-sample results."""
    predictions = results['predictions']
    labels = results['labels']
    probs = results['probs']
    samples = results['samples']
    
    print("\n" + "="*90)
    if show_errors_only:
        print("INCORRECT PREDICTIONS")
    else:
        print(f"DETAILED PREDICTIONS (showing first {max_show})")
    print("="*90)
    print(f"{'ID':<10} {'Expected':<8} {'Predicted':<8} {'Conf':<6} {'Status':<7} {'Service':<7} {'Cert':<15} {'Year':<6}")
    print("-"*90)
    
    shown = 0
    for i, (pred, label, prob, sample) in enumerate(zip(predictions, labels, probs, samples)):
        expected_grade = idx_to_class.get(label, 'unknown')
        predicted_grade = idx_to_class.get(pred, 'unknown')
        confidence = prob[pred] * 100
        is_correct = pred == label
        
        # Filter if showing errors only
        if show_errors_only and is_correct:
            continue
        
        if shown >= max_show:
            break
        
        inv_id = sample['inventory_id']
        cert = sample.get('cert_number', 'unknown')[:15]
        year = sample.get('year', 'unknown')
        service = sample.get('grading_service', 'UNK')[:7]
        
        status = "✓" if is_correct else "✗"
        
        print(f"{inv_id:<10} {expected_grade:<8} {predicted_grade:<8} {confidence:>5.1f}% {status:<7} {service:<7} {cert:<15} {year:<6}")
        shown += 1
    
    if show_errors_only:
        print(f"\nTotal errors: {shown}")
    else:
        remaining = len(predictions) - shown
        if remaining > 0:
            print(f"\n... and {remaining} more samples")


def save_results(results, idx_to_class, output_file='evaluation_results.json'):
    """Save evaluation results to JSON."""
    predictions = results['predictions']
    labels = results['labels']
    probs = results['probs']
    samples = results['samples']
    
    output_data = {
        'overall_accuracy': results['accuracy'],
        'accuracy_within_1_grade': results['accuracy_within_1'],
        'accuracy_within_2_grades': results['accuracy_within_2'],
        'total_samples': len(predictions),
        'correct_exact': int(results['accuracy'] * len(predictions) / 100),
        'correct_within_1': results['within_1_grade'],
        'correct_within_2': results['within_2_grades'],
        'per_class_accuracy': {
            idx_to_class.get(idx, f'class_{idx}'): {
                'accuracy': 100.0 * results['class_correct'][idx] / results['class_total'][idx],
                'correct': results['class_correct'][idx],
                'total': results['class_total'][idx]
            }
            for idx in results['class_total'].keys()
        },
        'predictions': [
            {
                'inventory_id': sample['inventory_id'],
                'cert_number': sample.get('cert_number', 'unknown'),
                'year': sample.get('year', 'unknown'),
                'grading_service': sample.get('grading_service', 'unknown'),
                'expected_grade': idx_to_class.get(label, 'unknown'),
                'predicted_grade': idx_to_class.get(pred, 'unknown'),
                'confidence': float(prob[pred]),
                'correct': bool(pred == label)
            }
            for pred, label, prob, sample in zip(predictions, labels, probs, samples)
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
        description='Evaluate trained model on test set',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', type=str, default='models/coin_resnet_dual_best.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, default='davidlawrence_coins_indians/data',
                       help='Path to JSON metadata directory')
    parser.add_argument('--images', type=str, default='davidlawrence_coins_indians/images',
                       help='Path to images directory')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--errors-only', action='store_true',
                       help='Show only incorrect predictions')
    parser.add_argument('--save', type=str,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("="*80)
    print("MODEL EVALUATION")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Images: {args.images}")
    
    # Load model
    model, class_to_idx, idx_to_class = load_model(args.model, device)
    
    # Prepare transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((1000, 1000)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset using model's class mapping
    dataset = EvaluationDataset(
        args.data,
        args.images,
        transform=transform,
        test_split=0.2,
        model_class_to_idx=class_to_idx  # Use model's classes!
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Evaluate
    results = evaluate_model(model, dataloader, device, idx_to_class)
    
    # Print detailed results
    print_detailed_results(results, idx_to_class, show_errors_only=args.errors_only)
    
    # Save if requested
    if args.save:
        save_results(results, idx_to_class, args.save)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

