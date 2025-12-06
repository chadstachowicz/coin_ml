"""
Ordinal Regression for Coin Grading

Instead of treating grades as independent classes (classification),
treat them as ordered values (regression/ordinal regression).

Two approaches:
1. Simple Regression: MSE/MAE on normalized Sheldon grades
2. Ordinal Regression: Explicitly model the ordering with ranked predictions

Advantages:
- Penalizes being off by 10 grades more than off by 1 grade
- More natural for the Sheldon scale (which is inherently ordered)
- Better evaluation metrics (MAE in actual grade numbers)
- Can predict intermediate values (e.g., "between MS64 and MS65")

Features:
- Dual-image input (obverse + reverse)
- ResNet-50 backbone
- Company-conditioned (optional)
- Multiple loss functions: MSE, MAE, Ordinal
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
from datetime import datetime
from tqdm import tqdm
import re


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_DIR = 'davidlawrence_dataset/Circulation'
OUTPUT_DIR = 'models'
LOG_DIR = 'runs/ordinal_' + datetime.now().strftime('%Y%m%d_%H%M%S')

# Model hyperparameters
IMAGE_SIZE = 512
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
FREEZE_BACKBONE = True
UNFREEZE_EPOCH = 15

# Regression settings
REGRESSION_TYPE = 'ordinal'  # 'mse', 'mae', or 'ordinal'
USE_COMPANY_CONDITIONING = True  # Include company as input
COMPANY_EMBEDDING_DIM = 32

# ============================================================================
# COMPANY BIAS (in steps - applied during training)
# ============================================================================
# Positive = company grades stricter → bump their labeled grade UP
# Negative = company grades looser → bump their labeled grade DOWN
# 
# Example: If CACG's MS64 looks like PCGS's MS65, set CACG to +1.0
#          This teaches the model that CACG MS64 = step position of MS65
#
# Set to 0 for companies with no known bias or as baseline (usually PCGS)
COMPANY_BIAS = {
    'PCGS': 0.0,      # Baseline - most common reference
    'NGC':  0.0,      # Generally comparable to PCGS
    'CACG': 0.5,      # CAC Grading - tends to grade stricter
    'ANAC': 0.0,      # ANACS - adjust if needed
    'ICG':  0.0,      # ICG
    'SEGS': 0.0,      # SEGS
    # Add more as needed
}
USE_COMPANY_BIAS = True  # Set to False to disable bias adjustment

# Grade normalization (Sheldon scale typically 1-70)
GRADE_MIN = 1.0
GRADE_MAX = 70.0

# Device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Using CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("Using MPS (Apple Silicon GPU)")
else:
    DEVICE = torch.device('cpu')
    print("Using CPU")

NUM_WORKERS = 0
PIN_MEMORY = True if torch.cuda.is_available() else False

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("="*70)
print("ORDINAL REGRESSION COIN GRADING")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Regression type: {REGRESSION_TYPE}")
print(f"Company conditioning: {USE_COMPANY_CONDITIONING}")
if USE_COMPANY_BIAS:
    print(f"Company bias: ENABLED")
    for company, bias in COMPANY_BIAS.items():
        if bias != 0:
            direction = "↑" if bias > 0 else "↓"
            print(f"  {company}: {bias:+.1f} steps {direction}")
else:
    print(f"Company bias: DISABLED")
print("="*70)


# ============================================================================
# HELPER FUNCTIONS - STEP-BASED ENCODING
# ============================================================================

# Valid grades in the grading scale (each step = 1 in loss function)
# This ensures VF25→VF30 and MS64→MS65 are treated equally!
VALID_GRADES = [2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45,
                50, 53, 55, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68]

# Create mappings
GRADE_TO_STEP = {grade: step for step, grade in enumerate(VALID_GRADES)}
STEP_TO_GRADE = {step: grade for step, grade in enumerate(VALID_GRADES)}
NUM_STEPS = len(VALID_GRADES)

print(f"Step-based encoding: {NUM_STEPS} valid grades")
print(f"  Grade range: {VALID_GRADES[0]} to {VALID_GRADES[-1]}")
print(f"  Each step = 1 in loss function (uniform penalty)")


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
        return float(match.group(1))
    
    # Fallback mapping for non-standard grades
    grade_map = {
        'poor': 1, 'fr': 2, 'ag': 3, 'g': 4, 'vg': 8,
        'f': 12, 'vf': 20, 'xf': 40, 'au': 50, 'ms': 60
    }
    
    grade_lower = grade_str.lower()
    for key, val in grade_map.items():
        if key in grade_lower:
            return float(val)
    
    # Default
    return 50.0


def sheldon_to_step(sheldon_grade):
    """Convert Sheldon grade to step position (0 to NUM_STEPS-1)."""
    # Round to nearest valid grade first
    rounded = round_to_valid_grade(sheldon_grade)
    return GRADE_TO_STEP.get(rounded, NUM_STEPS // 2)


def step_to_sheldon(step):
    """Convert step position back to Sheldon grade."""
    step = int(round(step))
    step = max(0, min(step, NUM_STEPS - 1))
    return STEP_TO_GRADE[step]


def normalize_grade(sheldon_grade):
    """
    Normalize grade to [0, 1] range using STEP-BASED encoding.
    
    This ensures each grade step contributes equally to the loss:
    - VF25 → VF30 = 1 step = same penalty as MS64 → MS65
    """
    step = sheldon_to_step(sheldon_grade)
    return float(step / (NUM_STEPS - 1))


def denormalize_grade(normalized_grade):
    """
    Convert normalized grade back to Sheldon scale.
    
    Input: normalized value in [0, 1]
    Output: Sheldon grade number
    """
    if isinstance(normalized_grade, torch.Tensor):
        # Convert to step, then to Sheldon
        steps = normalized_grade * (NUM_STEPS - 1)
        # Vectorized conversion
        result = torch.zeros_like(steps)
        for i, s in enumerate(steps):
            result[i] = step_to_sheldon(s.item())
        return result
    else:
        step = normalized_grade * (NUM_STEPS - 1)
        return step_to_sheldon(step)


def round_to_valid_grade(grade):
    """Round to nearest valid Sheldon grade."""
    return min(VALID_GRADES, key=lambda x: abs(x - grade))


# ============================================================================
# DATASET
# ============================================================================

class OrdinalCoinDataset(Dataset):
    """Dataset for ordinal regression on coin grades."""
    
    def __init__(self, data_dir, split='train', transform=None, use_company=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_company = use_company
        self.samples = []
        
        # Company mapping
        self.company_to_idx = {}
        self.idx_to_company = {}
        
        # Grade statistics
        self.grade_min = float('inf')
        self.grade_max = float('-inf')
        
        # Collect samples
        temp_samples = []
        companies = set()
        
        grade_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for grade_folder in grade_folders:
            grade_name = grade_folder.name
            grade_value = parse_sheldon_grade(grade_name)
            
            self.grade_min = min(self.grade_min, grade_value)
            self.grade_max = max(self.grade_max, grade_value)
            
            obverse_dir = grade_folder / 'obverse'
            reverse_dir = grade_folder / 'reverse'
            
            if not obverse_dir.exists() or not reverse_dir.exists():
                continue
            
            obverse_images = sorted([f for f in obverse_dir.glob('*.jpg') if f.is_file()])
            
            for obverse_img in obverse_images:
                reverse_img = reverse_dir / obverse_img.name
                
                if reverse_img.exists():
                    # Parse company from filename
                    parts = obverse_img.stem.split('-')
                    company = parts[1] if len(parts) >= 2 else 'UNKNOWN'
                    
                    # Skip unwanted companies
                    if company in ['OTHE', 'THAT', 'NONE']:
                        continue
                    
                    companies.add(company)
                    
                    temp_samples.append({
                        'obverse': obverse_img,
                        'reverse': reverse_img,
                        'grade_name': grade_name,
                        'grade_value': grade_value,
                        'company': company
                    })
        
        # Create company mapping
        for company_idx, company in enumerate(sorted(companies)):
            self.company_to_idx[company] = company_idx
            self.idx_to_company[company_idx] = company
        
        # Add company indices
        for sample in temp_samples:
            sample['company_idx'] = self.company_to_idx[sample['company']]
        
        self.samples = temp_samples
        
        # 80/20 split
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        n_train = int(0.8 * len(self.samples))
        
        if split == 'train':
            indices = indices[:n_train]
        else:
            indices = indices[n_train:]
        
        self.samples = [self.samples[i] for i in indices]
        
        # Statistics
        from collections import Counter
        grade_counts = Counter([s['grade_name'] for s in self.samples])
        company_counts = Counter([s['company'] for s in self.samples])
        
        grades = [s['grade_value'] for s in self.samples]
        
        print(f"\n{split.upper()}: {len(self.samples)} samples")
        print(f"  Grade range: {min(grades):.0f} - {max(grades):.0f}")
        print(f"  Mean grade: {np.mean(grades):.1f} ± {np.std(grades):.1f}")
        print(f"  Unique grades: {len(grade_counts)}")
        print(f"  Companies: {len(self.company_to_idx)} - {list(self.company_to_idx.keys())}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        obverse = Image.open(sample['obverse']).convert('RGB')
        reverse = Image.open(sample['reverse']).convert('RGB')
        
        if self.transform:
            obverse = self.transform(obverse)
            reverse = self.transform(reverse)
        
        # Normalize grade to [0, 1] - use float32 for MPS compatibility
        normalized_grade = normalize_grade(sample['grade_value'])
        
        # Apply company bias (in steps, converted to normalized space)
        if USE_COMPANY_BIAS and sample['company'] in COMPANY_BIAS:
            bias_steps = COMPANY_BIAS[sample['company']]
            # Convert step bias to normalized space: bias_steps / (NUM_STEPS - 1)
            bias_normalized = bias_steps / (NUM_STEPS - 1)
            normalized_grade = normalized_grade + bias_normalized
            # Clamp to valid range
            normalized_grade = max(0.0, min(1.0, normalized_grade))
        
        normalized_grade = torch.tensor(normalized_grade, dtype=torch.float32)
        grade_value = torch.tensor(sample['grade_value'], dtype=torch.float32)
        
        if self.use_company:
            return obverse, reverse, normalized_grade, sample['company_idx'], grade_value
        else:
            return obverse, reverse, normalized_grade, grade_value


# ============================================================================
# DATA TRANSFORMS
# ============================================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(degrees=5, fill=(255, 255, 255)),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05),
        fill=(255, 255, 255)
    ),
    transforms.ColorJitter(
        brightness=0.05,
        contrast=0.05,
        saturation=0.05,
        hue=0.01
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================================================
# ORDINAL REGRESSION MODEL
# ============================================================================

class OrdinalRegressionResNet(nn.Module):
    """
    ResNet for ordinal regression.
    
    Outputs a single continuous value (normalized grade).
    Optionally conditioned on grading company.
    """
    
    def __init__(self, num_companies=None, company_embedding_dim=32, freeze_backbone=True):
        super(OrdinalRegressionResNet, self).__init__()
        
        self.use_company = num_companies is not None
        
        # Company embedding
        if self.use_company:
            self.company_embedding = nn.Embedding(num_companies, company_embedding_dim)
        
        # Load pretrained ResNet-50
        weights = ResNet50_Weights.IMAGENET1K_V2
        obverse_resnet = resnet50(weights=weights)
        reverse_resnet = resnet50(weights=weights)
        
        # Encoders
        self.obverse_encoder = nn.Sequential(*list(obverse_resnet.children())[:-1])
        self.reverse_encoder = nn.Sequential(*list(reverse_resnet.children())[:-1])
        
        if freeze_backbone:
            for param in self.obverse_encoder.parameters():
                param.requires_grad = False
            for param in self.reverse_encoder.parameters():
                param.requires_grad = False
            print("✓ Backbone frozen")
        
        self.feature_dim = 2048
        
        # Fusion layer
        fusion_input_dim = self.feature_dim * 2
        if self.use_company:
            fusion_input_dim += company_embedding_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Regression head (outputs single value in [0, 1])
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
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def unfreeze_backbone(self):
        for param in self.obverse_encoder.parameters():
            param.requires_grad = True
        for param in self.reverse_encoder.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen")
    
    def forward(self, obverse, reverse, company_idx=None):
        # Encode images
        obverse_feat = self.obverse_encoder(obverse).view(obverse.size(0), -1)
        reverse_feat = self.reverse_encoder(reverse).view(reverse.size(0), -1)
        
        # Concatenate features
        combined = torch.cat([obverse_feat, reverse_feat], dim=1)
        
        # Add company embedding if available
        if self.use_company and company_idx is not None:
            company_emb = self.company_embedding(company_idx)
            combined = torch.cat([combined, company_emb], dim=1)
        
        # Fusion + regression
        fused = self.fusion(combined)
        output = self.regression_head(fused).squeeze(-1)  # [batch_size]
        
        return output


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class OrdinalLoss(nn.Module):
    """
    Ordinal regression loss.
    
    Penalizes predictions based on distance from true grade:
    - Off by 1 grade: small penalty
    - Off by 10 grades: large penalty
    
    Uses squared error but on the ordinal scale.
    """
    
    def __init__(self, reduction='mean'):
        super(OrdinalLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        # Both in [0, 1] normalized space
        # Squared error naturally penalizes larger distances more
        se = (predictions - targets) ** 2
        
        if self.reduction == 'mean':
            return se.mean()
        elif self.reduction == 'sum':
            return se.sum()
        else:
            return se


def get_loss_function(loss_type):
    """Get loss function based on type."""
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'ordinal':
        return OrdinalLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ============================================================================
# EVALUATION METRICS (Step-based)
# ============================================================================

def compute_mae_in_steps(predictions, targets):
    """
    Compute Mean Absolute Error in STEPS (not Sheldon points).
    
    Since we use step-based encoding, predictions and targets are already
    in normalized step space [0, 1]. We just convert back to steps.
    
    Args:
        predictions: normalized predictions [0, 1]
        targets: normalized targets [0, 1]
    
    Returns:
        MAE in grade steps (e.g., 1.5 means average error of 1.5 grade steps)
    """
    # Convert from [0, 1] back to step positions
    pred_steps = predictions * (NUM_STEPS - 1)
    true_steps = targets * (NUM_STEPS - 1)
    
    # MAE in steps
    mae = torch.abs(pred_steps - true_steps).mean()
    return mae.item()


def compute_accuracy_within_n_steps(predictions, targets, n=1):
    """
    Compute accuracy: % of predictions within N steps of truth.
    
    Since we use step-based encoding, this is straightforward.
    
    Args:
        predictions: normalized predictions [0, 1]
        targets: normalized targets [0, 1]
        n: tolerance in grade steps
    
    Returns:
        Accuracy (0-100)
    """
    # Convert from [0, 1] back to step positions
    pred_steps = predictions * (NUM_STEPS - 1)
    true_steps = targets * (NUM_STEPS - 1)
    
    # Round to nearest step
    pred_rounded = torch.round(pred_steps)
    true_rounded = torch.round(true_steps)
    
    # Count within n steps
    step_diff = torch.abs(pred_rounded - true_rounded)
    correct_count = (step_diff <= n).sum().item()
    
    return 100.0 * correct_count / len(predictions)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch, use_company):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
    
    for batch in pbar:
        if use_company:
            obverse, reverse, targets, company_idx, _ = batch
            obverse = obverse.to(device)
            reverse = reverse.to(device)
            targets = targets.to(device, dtype=torch.float32)  # Explicit float32 for MPS
            company_idx = company_idx.to(device)
        else:
            obverse, reverse, targets, _ = batch
            obverse = obverse.to(device)
            reverse = reverse.to(device)
            targets = targets.to(device, dtype=torch.float32)  # Explicit float32 for MPS
            company_idx = None
        
        optimizer.zero_grad()
        
        # Forward
        predictions = model(obverse, reverse, company_idx)
        loss = criterion(predictions, targets)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistics
        batch_size = obverse.size(0)
        running_loss += loss.item() * batch_size
        
        all_preds.append(predictions.detach())
        all_targets.append(targets.detach())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    mae = compute_mae_in_steps(all_preds, all_targets)
    acc_1 = compute_accuracy_within_n_steps(all_preds, all_targets, n=1)
    acc_2 = compute_accuracy_within_n_steps(all_preds, all_targets, n=2)
    
    return running_loss / len(loader.dataset), mae, acc_1, acc_2


def validate(model, loader, criterion, device, use_company):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Val')
        for batch in pbar:
            if use_company:
                obverse, reverse, targets, company_idx, _ = batch
                obverse = obverse.to(device)
                reverse = reverse.to(device)
                targets = targets.to(device, dtype=torch.float32)  # Explicit float32 for MPS
                company_idx = company_idx.to(device)
            else:
                obverse, reverse, targets, _ = batch
                obverse = obverse.to(device)
                reverse = reverse.to(device)
                targets = targets.to(device, dtype=torch.float32)  # Explicit float32 for MPS
                company_idx = None
            
            predictions = model(obverse, reverse, company_idx)
            loss = criterion(predictions, targets)
            
            batch_size = obverse.size(0)
            running_loss += loss.item() * batch_size
            
            all_preds.append(predictions)
            all_targets.append(targets)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    mae = compute_mae_in_steps(all_preds, all_targets)
    acc_1 = compute_accuracy_within_n_steps(all_preds, all_targets, n=1)
    acc_2 = compute_accuracy_within_n_steps(all_preds, all_targets, n=2)
    
    return running_loss / len(loader.dataset), mae, acc_1, acc_2


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\nCreating datasets...")
    train_dataset = OrdinalCoinDataset(
        DATA_DIR, split='train', transform=train_transform,
        use_company=USE_COMPANY_CONDITIONING
    )
    test_dataset = OrdinalCoinDataset(
        DATA_DIR, split='test', transform=val_transform,
        use_company=USE_COMPANY_CONDITIONING
    )
    val_dataset = test_dataset
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = test_loader
    
    print(f"\nDataloaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Test/Val: {len(test_loader)} batches")
    
    # Create model
    num_companies = len(train_dataset.company_to_idx) if USE_COMPANY_CONDITIONING else None
    
    model = OrdinalRegressionResNet(
        num_companies=num_companies,
        company_embedding_dim=COMPANY_EMBEDDING_DIM,
        freeze_backbone=FREEZE_BACKBONE
    )
    model = model.to(DEVICE)
    
    print(f"\nModel created:")
    print(f"  Regression type: {REGRESSION_TYPE}")
    print(f"  Company conditioning: {USE_COMPANY_CONDITIONING}")
    if USE_COMPANY_CONDITIONING:
        print(f"  Companies: {num_companies}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss function
    criterion = get_loss_function(REGRESSION_TYPE)
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    writer = SummaryWriter(LOG_DIR)
    
    # Training loop
    history = {
        'train_loss': [], 'train_mae': [], 'train_acc1': [], 'train_acc2': [],
        'val_loss': [], 'val_mae': [], 'val_acc1': [], 'val_acc2': []
    }
    
    best_mae = float('inf')
    best_model_path = os.path.join(OUTPUT_DIR, 'coin_ordinal_best.pth')
    
    print("\n" + "="*70)
    print("STARTING ORDINAL REGRESSION TRAINING")
    print("="*70)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Unfreeze backbone
        if FREEZE_BACKBONE and epoch == UNFREEZE_EPOCH:
            print(f"\n🔓 Unfreezing backbone")
            model.unfreeze_backbone()
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE/10, weight_decay=0.01)
        
        # Train
        train_loss, train_mae, train_acc1, train_acc2 = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch,
            USE_COMPANY_CONDITIONING
        )
        
        # Validate
        val_loss, val_mae, val_acc1, val_acc2 = validate(
            model, val_loader, criterion, DEVICE, USE_COMPANY_CONDITIONING
        )
        
        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['train_acc1'].append(train_acc1)
        history['train_acc2'].append(train_acc2)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_acc1'].append(val_acc1)
        history['val_acc2'].append(val_acc2)
        
        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MAE/train', train_mae, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)
        writer.add_scalar('Accuracy_±1/train', train_acc1, epoch)
        writer.add_scalar('Accuracy_±1/val', val_acc1, epoch)
        writer.add_scalar('Accuracy_±2/train', train_acc2, epoch)
        writer.add_scalar('Accuracy_±2/val', val_acc2, epoch)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train: MAE={train_mae:.2f} steps, ±1={train_acc1:.1f}%, ±2={train_acc2:.1f}%")
        print(f"  Val:   MAE={val_mae:.2f} steps, ±1={val_acc1:.1f}%, ±2={val_acc2:.1f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mae': val_mae,
                'val_acc1': val_acc1,
                'val_acc2': val_acc2,
                'company_to_idx': train_dataset.company_to_idx if USE_COMPANY_CONDITIONING else None,
                'idx_to_company': train_dataset.idx_to_company if USE_COMPANY_CONDITIONING else None,
                'grade_min': GRADE_MIN,
                'grade_max': GRADE_MAX,
                'valid_grades': VALID_GRADES,
                'num_steps': NUM_STEPS,
                'regression_type': REGRESSION_TYPE,
                'use_company': USE_COMPANY_CONDITIONING,
                'encoding': 'step_based',  # Mark this model as step-based
                'company_bias': COMPANY_BIAS if USE_COMPANY_BIAS else None,
                'use_company_bias': USE_COMPANY_BIAS
            }, best_model_path)
            print(f"  ✓ New best! MAE: {val_mae:.2f} steps")
    
    writer.close()
    
    # Save results
    with open(os.path.join(OUTPUT_DIR, 'history_ordinal.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best MAE: {best_mae:.2f} steps")
    print(f"Model saved: {best_model_path}")
    print("\n💡 Step-based encoding means:")
    print(f"   - VF25→VF30 = 1 step (same penalty as MS64→MS65)")
    print(f"   - MAE: Mean Absolute Error in grade STEPS")
    print(f"   - ±1: Within 1 step (e.g., MS64↔MS65, VF25↔VF30)")
    print(f"   - ±2: Within 2 steps (e.g., MS64↔MS66, VF25↔VF35)")

