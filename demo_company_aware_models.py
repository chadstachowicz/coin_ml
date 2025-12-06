"""
Demo: Company-Aware Model Inference

This script demonstrates how to use both company-aware models:

1. Multi-Task Model (Approach A):
   - Predicts both grade and company simultaneously
   - Learns company biases as auxiliary task

2. Company-Conditioned Model (Approach B):
   - You specify which company to mimic
   - Can ask "What would PCGS call this?" vs "What would NGC call this?"
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import sys

# Import our model classes
from coin_classifier_multitask import MultiTaskResNetClassifier
from coin_classifier_company_conditioned import CompanyConditionedResNet


def load_image(image_path, transform):
    """Load and transform an image."""
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)


def demo_multitask_model(model_path, obverse_path, reverse_path):
    """
    Demo Approach A: Multi-Task Model
    
    Predicts both grade and company from the same features.
    """
    print("\n" + "="*70)
    print("APPROACH A: MULTI-TASK MODEL")
    print("="*70)
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    num_grades = len(checkpoint['grade_to_idx'])
    num_companies = len(checkpoint['company_to_idx'])
    
    model = MultiTaskResNetClassifier(num_grades, num_companies, freeze_backbone=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load images
    transform = transforms.Compose([
        transforms.Resize((1000, 1000)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    obverse = load_image(obverse_path, transform)
    reverse = load_image(reverse_path, transform)
    
    # Inference
    with torch.no_grad():
        grade_output, company_output = model(obverse, reverse)
        
        grade_probs = F.softmax(grade_output, dim=1)[0]
        company_probs = F.softmax(company_output, dim=1)[0]
        
        # Top grades
        grade_top5_probs, grade_top5_idx = torch.topk(grade_probs, 5)
        
        # Top companies
        company_top3_probs, company_top3_idx = torch.topk(company_probs, min(3, num_companies))
    
    print("\nPredictions:")
    print("-" * 70)
    
    print("\n📊 Grade Predictions (Top 5):")
    for i, (prob, idx) in enumerate(zip(grade_top5_probs, grade_top5_idx), 1):
        grade = checkpoint['idx_to_grade'][int(idx)]
        print(f"  {i}. {grade:8s} - {prob*100:5.2f}%")
    
    print("\n🏢 Company Predictions:")
    for prob, idx in zip(company_top3_probs, company_top3_idx):
        company = checkpoint['idx_to_company'][int(idx)]
        print(f"  {company:8s} - {prob*100:5.2f}%")
    
    print("\n💡 Insight:")
    print("   This model learned that certain visual features correlate with")
    print("   specific grading companies, potentially capturing their biases.")
    

def demo_company_conditioned_model(model_path, obverse_path, reverse_path):
    """
    Demo Approach B: Company-Conditioned Model
    
    You specify which company to mimic, and see how predictions differ.
    """
    print("\n" + "="*70)
    print("APPROACH B: COMPANY-CONDITIONED MODEL")
    print("="*70)
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    num_grades = len(checkpoint['grade_to_idx'])
    num_companies = len(checkpoint['company_to_idx'])
    company_embedding_dim = checkpoint.get('company_embedding_dim', 32)
    
    model = CompanyConditionedResNet(
        num_grades, num_companies, company_embedding_dim, freeze_backbone=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load images
    transform = transforms.Compose([
        transforms.Resize((1000, 1000)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    obverse = load_image(obverse_path, transform)
    reverse = load_image(reverse_path, transform)
    
    company_to_idx = checkpoint['company_to_idx']
    idx_to_grade = checkpoint['idx_to_grade']
    
    print("\nCompany-Specific Predictions:")
    print("-" * 70)
    
    # Predict for each company
    results = {}
    
    with torch.no_grad():
        for company_name in sorted(company_to_idx.keys()):
            output = model.predict_with_company(obverse, reverse, company_name, company_to_idx)
            probs = F.softmax(output, dim=1)[0]
            
            # Top prediction
            top_prob, top_idx = torch.max(probs, 0)
            top_grade = idx_to_grade[int(top_idx)]
            
            # Top 3
            top3_probs, top3_idx = torch.topk(probs, 3)
            
            results[company_name] = {
                'top_grade': top_grade,
                'top_prob': float(top_prob),
                'top3': [(idx_to_grade[int(idx)], float(prob)) 
                         for idx, prob in zip(top3_idx, top3_probs)]
            }
    
    # Display results
    for company_name in sorted(results.keys()):
        result = results[company_name]
        print(f"\n🏢 {company_name}:")
        print(f"   Top Prediction: {result['top_grade']} ({result['top_prob']*100:.1f}%)")
        print(f"   Top 3:")
        for grade, prob in result['top3']:
            print(f"     • {grade:8s} - {prob*100:5.1f}%")
    
    print("\n💡 Insight:")
    print("   Notice how predictions can differ by company! This model learned")
    print("   that PCGS vs NGC vs others have different grading tendencies.")
    print("\n   At inference, you can explicitly ask:")
    print("   'What would PCGS call this?' by feeding in PCGS encoding.")


def compare_predictions(model_path_a, model_path_b, obverse_path, reverse_path):
    """Compare both approaches side-by-side."""
    print("\n" + "="*70)
    print("COMPARISON: MULTI-TASK vs COMPANY-CONDITIONED")
    print("="*70)
    
    # Approach A
    checkpoint_a = torch.load(model_path_a, map_location='cpu')
    model_a = MultiTaskResNetClassifier(
        len(checkpoint_a['grade_to_idx']),
        len(checkpoint_a['company_to_idx']),
        freeze_backbone=False
    )
    model_a.load_state_dict(checkpoint_a['model_state_dict'])
    model_a.eval()
    
    # Approach B
    checkpoint_b = torch.load(model_path_b, map_location='cpu')
    model_b = CompanyConditionedResNet(
        len(checkpoint_b['grade_to_idx']),
        len(checkpoint_b['company_to_idx']),
        checkpoint_b.get('company_embedding_dim', 32),
        freeze_backbone=False
    )
    model_b.load_state_dict(checkpoint_b['model_state_dict'])
    model_b.eval()
    
    # Load images
    transform = transforms.Compose([
        transforms.Resize((1000, 1000)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    obverse = load_image(obverse_path, transform)
    reverse = load_image(reverse_path, transform)
    
    with torch.no_grad():
        # Model A
        grade_out_a, company_out_a = model_a(obverse, reverse)
        grade_probs_a = F.softmax(grade_out_a, dim=1)[0]
        company_probs_a = F.softmax(company_out_a, dim=1)[0]
        
        top_grade_idx_a = torch.argmax(grade_probs_a)
        top_company_idx_a = torch.argmax(company_probs_a)
        
        top_grade_a = checkpoint_a['idx_to_grade'][int(top_grade_idx_a)]
        top_company_a = checkpoint_a['idx_to_company'][int(top_company_idx_a)]
        
        # Model B (use predicted company from A)
        if top_company_a in checkpoint_b['company_to_idx']:
            grade_out_b = model_b.predict_with_company(
                obverse, reverse, top_company_a, checkpoint_b['company_to_idx']
            )
            grade_probs_b = F.softmax(grade_out_b, dim=1)[0]
            top_grade_idx_b = torch.argmax(grade_probs_b)
            top_grade_b = checkpoint_b['idx_to_grade'][int(top_grade_idx_b)]
        else:
            top_grade_b = "N/A"
    
    print(f"\nModel A (Multi-Task):")
    print(f"  Predicted Grade:   {top_grade_a} ({float(grade_probs_a[top_grade_idx_a])*100:.1f}%)")
    print(f"  Predicted Company: {top_company_a} ({float(company_probs_a[top_company_idx_a])*100:.1f}%)")
    
    print(f"\nModel B (Company-Conditioned with {top_company_a}):")
    print(f"  Predicted Grade:   {top_grade_b}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("COMPANY-AWARE MODEL DEMO")
    print("="*70)
    print("\nUsage examples:")
    print("  1. Demo multi-task model:")
    print("     python demo_company_aware_models.py multitask <obverse.jpg> <reverse.jpg>")
    print("\n  2. Demo company-conditioned model:")
    print("     python demo_company_aware_models.py conditioned <obverse.jpg> <reverse.jpg>")
    print("\n  3. Compare both:")
    print("     python demo_company_aware_models.py compare <obverse.jpg> <reverse.jpg>")
    
    if len(sys.argv) < 4:
        print("\n⚠️  Please provide: <mode> <obverse_path> <reverse_path>")
        print("     mode: 'multitask', 'conditioned', or 'compare'")
        sys.exit(1)
    
    mode = sys.argv[1]
    obverse_path = sys.argv[2]
    reverse_path = sys.argv[3]
    
    if mode == 'multitask':
        model_path = 'models/coin_multitask_best.pth'
        demo_multitask_model(model_path, obverse_path, reverse_path)
    
    elif mode == 'conditioned':
        model_path = 'models/coin_company_conditioned_best.pth'
        demo_company_conditioned_model(model_path, obverse_path, reverse_path)
    
    elif mode == 'compare':
        model_path_a = 'models/coin_multitask_best.pth'
        model_path_b = 'models/coin_company_conditioned_best.pth'
        compare_predictions(model_path_a, model_path_b, obverse_path, reverse_path)
    
    else:
        print(f"\n⚠️  Unknown mode: {mode}")
        print("     Use: 'multitask', 'conditioned', or 'compare'")



