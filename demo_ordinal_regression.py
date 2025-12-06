"""
Demo: Ordinal Regression Inference

Shows how regression predictions differ from classification:
- Continuous grade values (64.3 instead of just "MS64")
- Error measured in actual grade distance
- Can see when model is uncertain (e.g., "between MS64 and MS65")
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import sys
import json

from coin_classifier_ordinal_regression import (
    OrdinalRegressionResNet,
    denormalize_grade,
    round_to_valid_grade,
    parse_sheldon_grade
)


def load_image(image_path, transform):
    """Load and transform an image."""
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)


def predict_grade(model, obverse_path, reverse_path, company_name=None,
                 company_to_idx=None, device='cpu'):
    """
    Predict grade for a coin.
    
    Returns:
        continuous_grade: Raw prediction (e.g., 64.3)
        rounded_grade: Rounded to valid Sheldon grade (e.g., 64)
        grade_name: Standard name (e.g., "MS64")
    """
    # Transform
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load images
    obverse = load_image(obverse_path, transform).to(device)
    reverse = load_image(reverse_path, transform).to(device)
    
    # Prepare company
    company_idx = None
    if company_name and company_to_idx:
        company_idx = torch.tensor([company_to_idx[company_name]], device=device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        normalized_pred = model(obverse, reverse, company_idx)
        # Ensure float32 for MPS compatibility
        continuous_grade = denormalize_grade(normalized_pred.float().cpu()).item()
        rounded = round_to_valid_grade(continuous_grade)
    
    # Format grade name
    if rounded >= 60:
        grade_name = f"MS{rounded}"
    elif rounded >= 50:
        grade_name = f"AU{rounded}"
    elif rounded >= 40:
        grade_name = f"XF{rounded}"
    elif rounded >= 20:
        grade_name = f"VF{rounded}"
    elif rounded >= 8:
        grade_name = f"VG{rounded:02d}"
    else:
        grade_name = f"G{rounded:02d}"
    
    return continuous_grade, rounded, grade_name


def demo_single_prediction(model_path, obverse_path, reverse_path, true_grade=None):
    """Demo a single prediction showing regression outputs."""
    print("\n" + "="*70)
    print("ORDINAL REGRESSION DEMO - Single Prediction")
    print("="*70)
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    use_company = checkpoint.get('use_company', False)
    company_to_idx = checkpoint.get('company_to_idx')
    idx_to_company = checkpoint.get('idx_to_company')
    
    num_companies = len(company_to_idx) if use_company else None
    
    model = OrdinalRegressionResNet(
        num_companies=num_companies,
        company_embedding_dim=32,
        freeze_backbone=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\nModel info:")
    print(f"  Regression type: {checkpoint['regression_type']}")
    print(f"  Company conditioning: {use_company}")
    print(f"  Best validation MAE: {checkpoint['val_mae']:.2f} grades")
    
    # Parse company from filename if available
    filename = Path(obverse_path).stem
    parts = filename.split('-')
    coin_company = parts[1] if len(parts) >= 2 else None
    
    print(f"\nInput:")
    print(f"  Obverse: {obverse_path}")
    print(f"  Reverse: {reverse_path}")
    if coin_company:
        print(f"  Company: {coin_company}")
    if true_grade:
        print(f"  True grade: {true_grade}")
    
    # Predict
    if use_company and coin_company and coin_company in company_to_idx:
        continuous, rounded, grade_name = predict_grade(
            model, obverse_path, reverse_path, coin_company, company_to_idx
        )
    else:
        continuous, rounded, grade_name = predict_grade(
            model, obverse_path, reverse_path
        )
    
    print("\n" + "-"*70)
    print("PREDICTION:")
    print("-"*70)
    print(f"  Continuous: {continuous:.2f}")
    print(f"  Rounded:    {rounded}")
    print(f"  Grade:      {grade_name}")
    
    # Uncertainty
    uncertainty = abs(continuous - rounded)
    if uncertainty < 0.3:
        confidence = "High confidence"
    elif uncertainty < 0.7:
        confidence = "Medium confidence"
    else:
        confidence = "Low confidence (borderline)"
    
    print(f"\n  Uncertainty: {uncertainty:.2f} ({confidence})")
    
    if uncertainty > 0.4:
        # Might be between grades
        if continuous > rounded:
            next_grade = rounded + 1
            print(f"  → Model thinks this is between {grade_name} and MS{next_grade}")
        else:
            prev_grade = rounded - 1
            print(f"  → Model thinks this is between MS{prev_grade} and {grade_name}")
    
    # Compare to true grade if available
    if true_grade:
        true_value = parse_sheldon_grade(true_grade)
        error = abs(continuous - true_value)
        print(f"\n  Error: {error:.2f} grades")
        
        if error <= 1:
            print("  ✅ Excellent prediction (within 1 grade)")
        elif error <= 2:
            print("  ✓ Good prediction (within 2 grades)")
        elif error <= 5:
            print("  ~ Reasonable prediction (within 5 grades)")
        else:
            print("  ❌ Large error")
    
    print("="*70)


def demo_company_comparison(model_path, obverse_path, reverse_path):
    """Show how predictions differ by company."""
    print("\n" + "="*70)
    print("COMPANY-CONDITIONED ORDINAL REGRESSION")
    print("="*70)
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if not checkpoint.get('use_company', False):
        print("\n⚠️  This model was not trained with company conditioning!")
        return
    
    company_to_idx = checkpoint['company_to_idx']
    
    model = OrdinalRegressionResNet(
        num_companies=len(company_to_idx),
        company_embedding_dim=32,
        freeze_backbone=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\nComparing predictions across companies:")
    print("-"*70)
    
    results = {}
    for company in sorted(company_to_idx.keys()):
        continuous, rounded, grade_name = predict_grade(
            model, obverse_path, reverse_path, company, company_to_idx
        )
        results[company] = (continuous, rounded, grade_name)
        print(f"  {company:8s}: {grade_name:6s} (continuous: {continuous:.2f})")
    
    # Check for disagreements
    grades = [r[1] for r in results.values()]
    if len(set(grades)) > 1:
        print("\n💡 Companies disagree on grading!")
        print("   This captures real-world grading differences.")
    else:
        print("\n✓ Companies agree on the grade.")
    
    # Show spread
    continuous_grades = [r[0] for r in results.values()]
    spread = max(continuous_grades) - min(continuous_grades)
    print(f"\n   Grade spread: {spread:.2f} grades")
    if spread > 1.0:
        print("   → Significant disagreement")
    elif spread > 0.5:
        print("   → Moderate disagreement")
    else:
        print("   → Strong agreement")
    
    print("="*70)


def demo_batch_evaluation(model_path, test_dir):
    """Evaluate on a batch of test images."""
    print("\n" + "="*70)
    print("BATCH EVALUATION")
    print("="*70)
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    use_company = checkpoint.get('use_company', False)
    company_to_idx = checkpoint.get('company_to_idx')
    
    model = OrdinalRegressionResNet(
        num_companies=len(company_to_idx) if use_company else None,
        company_embedding_dim=32,
        freeze_backbone=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Find test images
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"\n⚠️  Test directory not found: {test_dir}")
        return
    
    # Collect predictions
    errors = []
    predictions_list = []
    
    for grade_folder in sorted(test_path.iterdir()):
        if not grade_folder.is_dir():
            continue
        
        true_grade_str = grade_folder.name
        true_grade_value = parse_sheldon_grade(true_grade_str)
        
        obverse_dir = grade_folder / 'obverse'
        reverse_dir = grade_folder / 'reverse'
        
        if not obverse_dir.exists():
            continue
        
        for obverse_img in list(obverse_dir.glob('*.jpg'))[:5]:  # Limit to 5 per grade
            reverse_img = reverse_dir / obverse_img.name
            if not reverse_img.exists():
                continue
            
            # Parse company
            parts = obverse_img.stem.split('-')
            company = parts[1] if len(parts) >= 2 else None
            
            # Predict
            try:
                if use_company and company and company in company_to_idx:
                    continuous, rounded, grade_name = predict_grade(
                        model, obverse_img, reverse_img, company, company_to_idx
                    )
                else:
                    continuous, rounded, grade_name = predict_grade(
                        model, obverse_img, reverse_img
                    )
                
                error = abs(continuous - true_grade_value)
                errors.append(error)
                
                predictions_list.append({
                    'true': true_grade_str,
                    'predicted': grade_name,
                    'continuous': continuous,
                    'error': error,
                    'company': company
                })
            except Exception as e:
                print(f"Error processing {obverse_img}: {e}")
                continue
    
    if not errors:
        print("\n⚠️  No test samples found!")
        return
    
    # Statistics
    mae = sum(errors) / len(errors)
    within_1 = sum(1 for e in errors if e <= 1) / len(errors) * 100
    within_2 = sum(1 for e in errors if e <= 2) / len(errors) * 100
    within_5 = sum(1 for e in errors if e <= 5) / len(errors) * 100
    
    print(f"\nEvaluated {len(errors)} coins:")
    print(f"  MAE: {mae:.2f} grades")
    print(f"  Within ±1 grade: {within_1:.1f}%")
    print(f"  Within ±2 grades: {within_2:.1f}%")
    print(f"  Within ±5 grades: {within_5:.1f}%")
    
    # Show some examples
    print("\nExample predictions:")
    print("-"*70)
    for pred in predictions_list[:10]:
        status = "✓" if pred['error'] <= 1 else "~" if pred['error'] <= 2 else "✗"
        print(f"  {status} True: {pred['true']:6s} | Pred: {pred['predicted']:6s} "
              f"({pred['continuous']:.1f}) | Error: {pred['error']:.2f}")
    
    print("="*70)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("ORDINAL REGRESSION DEMO")
        print("="*70)
        print("\nUsage:")
        print("  1. Single prediction:")
        print("     python demo_ordinal_regression.py single <obverse.jpg> <reverse.jpg> [true_grade]")
        print("\n  2. Company comparison:")
        print("     python demo_ordinal_regression.py compare <obverse.jpg> <reverse.jpg>")
        print("\n  3. Batch evaluation:")
        print("     python demo_ordinal_regression.py batch <test_dir>")
        sys.exit(1)
    
    mode = sys.argv[1]
    model_path = 'models/coin_ordinal_best.pth'
    
    if mode == 'single':
        if len(sys.argv) < 4:
            print("Usage: python demo_ordinal_regression.py single <obverse> <reverse> [true_grade]")
            sys.exit(1)
        obverse = sys.argv[2]
        reverse = sys.argv[3]
        true_grade = sys.argv[4] if len(sys.argv) > 4 else None
        demo_single_prediction(model_path, obverse, reverse, true_grade)
    
    elif mode == 'compare':
        if len(sys.argv) < 4:
            print("Usage: python demo_ordinal_regression.py compare <obverse> <reverse>")
            sys.exit(1)
        obverse = sys.argv[2]
        reverse = sys.argv[3]
        demo_company_comparison(model_path, obverse, reverse)
    
    elif mode == 'batch':
        if len(sys.argv) < 3:
            print("Usage: python demo_ordinal_regression.py batch <test_dir>")
            sys.exit(1)
        test_dir = sys.argv[2]
        demo_batch_evaluation(model_path, test_dir)
    
    else:
        print(f"Unknown mode: {mode}")
        print("Use: 'single', 'compare', or 'batch'")

