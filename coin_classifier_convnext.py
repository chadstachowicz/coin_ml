"""
Ordinal Regression for Coin Grading - ConvNeXt-Small

Uses ConvNeXt-Small backbone instead of ResNet-50.
ConvNeXt is a modernized ConvNet that matches Vision Transformer performance
while maintaining the simplicity of standard ConvNets.

Key differences from ResNet:
- Uses depthwise separable convolutions
- Layer normalization instead of batch normalization
- GELU activation instead of ReLU
- Larger kernel sizes (7x7)
- Better accuracy/compute tradeoff

Features:
- Dual-image input (obverse + reverse)
- ConvNeXt-Small backbone (768-dim features)
- Company-conditioned (optional)
- Step-based ordinal regression
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
from tqdm import tqdm
import re


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_DIR = 'davidlawrence_dataset/Circulation'
OUTPUT_DIR = 'models'
LOG_DIR = 'runs/convnext_' + datetime.now().strftime('%Y%m%d_%H%M%S')

# Model hyperparameters
IMAGE_SIZE = 512
BATCH_SIZE = 16  # ConvNeXt uses more memory, may need smaller batch
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
FREEZE_BACKBONE = True
UNFREEZE_EPOCH = 5

# Regression settings
REGRESSION_TYPE = 'ordinal'  # 'mse', 'mae', or 'ordinal'
USE_COMPANY_CONDITIONING = False  # Include company as input
COMPANY_EMBEDDING_DIM = 32

# Preprocessing settings
USE_PREPROCESSING = False  # Hough circle detection + white background

# ============================================================================
# COMPANY BIAS (in steps - applied during training)
# ============================================================================
COMPANY_BIAS = {
    'PCGS': 0.0,      # Baseline
    'NGC':  0.0,      # Generally comparable to PCGS
    'CACG': 0.5,      # CAC Grading - tends to grade stricter
    'ANAC': 0.0,      # ANACS
    'ICG':  0.0,      # ICG
    'SEGS': 0.0,      # SEGS
}
USE_COMPANY_BIAS = True

# Grade normalization
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
print("CONVNEXT-SMALL ORDINAL REGRESSION COIN GRADING")
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

VALID_GRADES = [2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45,
                50, 53, 55, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68]

GRADE_TO_STEP = {grade: step for step, grade in enumerate(VALID_GRADES)}
STEP_TO_GRADE = {step: grade for step, grade in enumerate(VALID_GRADES)}
NUM_STEPS = len(VALID_GRADES)

print(f"Step-based encoding: {NUM_STEPS} valid grades")
print(f"  Grade range: {VALID_GRADES[0]} to {VALID_GRADES[-1]}")
print(f"  Each step = 1 in loss function (uniform penalty)")


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


def sheldon_to_step(sheldon_grade):
    """Convert Sheldon grade to step position (0 to NUM_STEPS-1)."""
    rounded = round_to_valid_grade(sheldon_grade)
    return GRADE_TO_STEP.get(rounded, NUM_STEPS // 2)


def step_to_sheldon(step):
    """Convert step position back to Sheldon grade."""
    step = int(round(step))
    step = max(0, min(step, NUM_STEPS - 1))
    return STEP_TO_GRADE[step]


def normalize_grade(sheldon_grade):
    """Normalize grade to [0, 1] range using STEP-BASED encoding."""
    step = sheldon_to_step(sheldon_grade)
    return float(step / (NUM_STEPS - 1))


def denormalize_grade(normalized_grade):
    """Convert normalized grade back to Sheldon scale."""
    if isinstance(normalized_grade, torch.Tensor):
        steps = normalized_grade * (NUM_STEPS - 1)
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
# PREPROCESSING - HOUGH CIRCLE DETECTION
# ============================================================================

def preprocess_coin_image(pil_image, output_size=None):
    """Preprocess coin image with Hough circle detection and white background."""
    img_rgb = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    height, width = img_bgr.shape[:2]
    min_dim = min(height, width)
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dim // 2,
        param1=50, param2=30,
        minRadius=int(min_dim * 0.2), maxRadius=int(min_dim * 0.5)
    )
    
    if circles is None:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dim // 3,
            param1=100, param2=20,
            minRadius=int(min_dim * 0.15), maxRadius=int(min_dim * 0.55)
        )
    
    if circles is None:
        cx, cy = width // 2, height // 2
        radius = min_dim // 2 - 10
    else:
        circles = np.uint16(np.around(circles))
        cx, cy, radius = circles[0][0]
    
    mask_radius = int(radius * 1.02)
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), mask_radius, 255, -1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    
    white_img = np.ones_like(img_bgr) * 255
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    blended = (img_bgr.astype(float) * mask_3ch + white_img.astype(float) * (1 - mask_3ch)).astype(np.uint8)
    
    padding = int(radius * 0.05)
    crop_radius = radius + padding
    x1 = max(0, int(cx - crop_radius))
    y1 = max(0, int(cy - crop_radius))
    x2 = min(width, int(cx + crop_radius))
    y2 = min(height, int(cy + crop_radius))
    
    cropped = blended[y1:y2, x1:x2]
    
    if output_size is not None:
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
            
            cropped = white_bg
    
    result_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


print(f"Hough circle preprocessing: {'ENABLED' if USE_PREPROCESSING else 'DISABLED'}")


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
        
        self.company_to_idx = {}
        self.idx_to_company = {}
        
        self.grade_min = float('inf')
        self.grade_max = float('-inf')
        
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
                    parts = obverse_img.stem.split('-')
                    company = parts[1] if len(parts) >= 2 else 'UNKNOWN'
                    
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
        
        for company_idx, company in enumerate(sorted(companies)):
            self.company_to_idx[company] = company_idx
            self.idx_to_company[company_idx] = company
        
        for sample in temp_samples:
            sample['company_idx'] = self.company_to_idx[sample['company']]
        
        self.samples = temp_samples
        
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        n_train = int(0.8 * len(self.samples))
        
        if split == 'train':
            indices = indices[:n_train]
        else:
            indices = indices[n_train:]
        
        self.samples = [self.samples[i] for i in indices]
        
        from collections import Counter
        grade_counts = Counter([s['grade_name'] for s in self.samples])
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
        
        if USE_PREPROCESSING:
            obverse = preprocess_coin_image(obverse, output_size=IMAGE_SIZE)
            reverse = preprocess_coin_image(reverse, output_size=IMAGE_SIZE)
        
        if self.transform:
            obverse = self.transform(obverse)
            reverse = self.transform(reverse)
        
        normalized_grade = normalize_grade(sample['grade_value'])
        
        if USE_COMPANY_BIAS and sample['company'] in COMPANY_BIAS:
            bias_steps = COMPANY_BIAS[sample['company']]
            bias_normalized = bias_steps / (NUM_STEPS - 1)
            normalized_grade = normalized_grade + bias_normalized
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

_train_transforms = []
_val_transforms = []

if not USE_PREPROCESSING:
    _train_transforms.append(transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)))
    _val_transforms.append(transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)))

_train_transforms.extend([
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

_val_transforms.extend([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transform = transforms.Compose(_train_transforms)
val_transform = transforms.Compose(_val_transforms)


# ============================================================================
# CONVNEXT ORDINAL REGRESSION MODEL
# ============================================================================

class OrdinalRegressionConvNeXt(nn.Module):
    """
    ConvNeXt-Small for ordinal regression.
    
    ConvNeXt-Small specs:
    - Feature dimension: 768 (vs 2048 for ResNet-50)
    - ~50M parameters
    - Better accuracy than ResNet at similar compute
    
    Outputs a single continuous value (normalized grade).
    Optionally conditioned on grading company.
    """
    
    def __init__(self, num_companies=None, company_embedding_dim=32, freeze_backbone=True):
        super(OrdinalRegressionConvNeXt, self).__init__()
        
        self.use_company = num_companies is not None
        
        # Company embedding
        if self.use_company:
            self.company_embedding = nn.Embedding(num_companies, company_embedding_dim)
        
        # Load pretrained ConvNeXt-Small
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
        obverse_convnext = convnext_small(weights=weights)
        reverse_convnext = convnext_small(weights=weights)
        
        # ConvNeXt structure: features -> avgpool -> classifier
        # We want features + avgpool, excluding the classifier
        self.obverse_features = obverse_convnext.features
        self.obverse_avgpool = obverse_convnext.avgpool
        
        self.reverse_features = reverse_convnext.features
        self.reverse_avgpool = reverse_convnext.avgpool
        
        if freeze_backbone:
            for param in self.obverse_features.parameters():
                param.requires_grad = False
            for param in self.reverse_features.parameters():
                param.requires_grad = False
            print("✓ ConvNeXt backbone frozen")
        
        # ConvNeXt-Small outputs 768-dimensional features
        self.feature_dim = 768
        
        # Fusion layer
        fusion_input_dim = self.feature_dim * 2
        if self.use_company:
            fusion_input_dim += company_embedding_dim
        
        # Use LayerNorm instead of BatchNorm to match ConvNeXt style
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),  # ConvNeXt uses GELU
        )
        
        # Regression head (outputs single value in [0, 1])
        self.regression_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def unfreeze_backbone(self):
        """Unfreeze the ConvNeXt backbone for fine-tuning."""
        for param in self.obverse_features.parameters():
            param.requires_grad = True
        for param in self.reverse_features.parameters():
            param.requires_grad = True
        print("✓ ConvNeXt backbone unfrozen")
    
    def forward(self, obverse, reverse, company_idx=None):
        # Encode images through ConvNeXt
        # features output: [B, 768, H/32, W/32]
        # avgpool output: [B, 768, 1, 1]
        obverse_feat = self.obverse_avgpool(self.obverse_features(obverse))
        obverse_feat = obverse_feat.view(obverse.size(0), -1)  # [B, 768]
        
        reverse_feat = self.reverse_avgpool(self.reverse_features(reverse))
        reverse_feat = reverse_feat.view(reverse.size(0), -1)  # [B, 768]
        
        # Concatenate features
        combined = torch.cat([obverse_feat, reverse_feat], dim=1)  # [B, 1536]
        
        # Add company embedding if available
        if self.use_company and company_idx is not None:
            company_emb = self.company_embedding(company_idx)
            combined = torch.cat([combined, company_emb], dim=1)
        
        # Fusion + regression
        fused = self.fusion(combined)
        output = self.regression_head(fused).squeeze(-1)
        
        return output


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class OrdinalLoss(nn.Module):
    """Ordinal regression loss using squared error."""
    
    def __init__(self, reduction='mean'):
        super(OrdinalLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, predictions, targets):
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
    """Compute Mean Absolute Error in STEPS."""
    pred_steps = predictions * (NUM_STEPS - 1)
    true_steps = targets * (NUM_STEPS - 1)
    mae = torch.abs(pred_steps - true_steps).mean()
    return mae.item()


def compute_accuracy_within_n_steps(predictions, targets, n=1):
    """Compute accuracy: % of predictions within N steps of truth."""
    pred_steps = predictions * (NUM_STEPS - 1)
    true_steps = targets * (NUM_STEPS - 1)
    
    pred_rounded = torch.round(pred_steps)
    true_rounded = torch.round(true_steps)
    
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
            targets = targets.to(device, dtype=torch.float32)
            company_idx = company_idx.to(device)
        else:
            obverse, reverse, targets, _ = batch
            obverse = obverse.to(device)
            reverse = reverse.to(device)
            targets = targets.to(device, dtype=torch.float32)
            company_idx = None
        
        optimizer.zero_grad()
        
        predictions = model(obverse, reverse, company_idx)
        loss = criterion(predictions, targets)
        
        loss.backward()
        optimizer.step()
        
        batch_size = obverse.size(0)
        running_loss += loss.item() * batch_size
        
        all_preds.append(predictions.detach())
        all_targets.append(targets.detach())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
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
                targets = targets.to(device, dtype=torch.float32)
                company_idx = company_idx.to(device)
            else:
                obverse, reverse, targets, _ = batch
                obverse = obverse.to(device)
                reverse = reverse.to(device)
                targets = targets.to(device, dtype=torch.float32)
                company_idx = None
            
            predictions = model(obverse, reverse, company_idx)
            loss = criterion(predictions, targets)
            
            batch_size = obverse.size(0)
            running_loss += loss.item() * batch_size
            
            all_preds.append(predictions)
            all_targets.append(targets)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
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
    
    model = OrdinalRegressionConvNeXt(
        num_companies=num_companies,
        company_embedding_dim=COMPANY_EMBEDDING_DIM,
        freeze_backbone=FREEZE_BACKBONE
    )
    model = model.to(DEVICE)
    
    print(f"\nModel created (ConvNeXt-Small):")
    print(f"  Regression type: {REGRESSION_TYPE}")
    print(f"  Company conditioning: {USE_COMPANY_CONDITIONING}")
    if USE_COMPANY_CONDITIONING:
        print(f"  Companies: {num_companies}")
    print(f"  Feature dim: 768 (ConvNeXt-Small)")
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
        optimizer, mode='min', factor=0.1, patience=4, verbose=True
    )
    
    writer = SummaryWriter(LOG_DIR)
    
    # Training loop
    history = {
        'train_loss': [], 'train_mae': [], 'train_acc1': [], 'train_acc2': [],
        'val_loss': [], 'val_mae': [], 'val_acc1': [], 'val_acc2': []
    }
    
    best_mae = float('inf')
    best_model_path = os.path.join(OUTPUT_DIR, 'coin_convnext_best.pth')
    
    print("\n" + "="*70)
    print("STARTING CONVNEXT ORDINAL REGRESSION TRAINING")
    print("="*70)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Unfreeze backbone
        if FREEZE_BACKBONE and epoch == UNFREEZE_EPOCH:
            print(f"\n🔓 Unfreezing ConvNeXt backbone")
            model.unfreeze_backbone()
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE/10, weight_decay=0.01)
            # Recreate scheduler for the new optimizer
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
        
        # Train
        train_loss, train_mae, train_acc1, train_acc2 = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch,
            USE_COMPANY_CONDITIONING
        )
        
        # Validate
        val_loss, val_mae, val_acc1, val_acc2 = validate(
            model, val_loader, criterion, DEVICE, USE_COMPANY_CONDITIONING
        )
        
        scheduler.step(val_loss)  # Use loss (MSE-like) for smoother scheduling
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
                'encoding': 'step_based',
                'company_bias': COMPANY_BIAS if USE_COMPANY_BIAS else None,
                'use_company_bias': USE_COMPANY_BIAS,
                'backbone': 'convnext_small'  # Mark the backbone type
            }, best_model_path)
            print(f"  ✓ New best! MAE: {val_mae:.2f} steps")
    
    writer.close()
    
    # Save results
    with open(os.path.join(OUTPUT_DIR, 'history_convnext.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best MAE: {best_mae:.2f} steps")
    print(f"Model saved: {best_model_path}")
    print("\n💡 ConvNeXt-Small advantages:")
    print("   - Modern architecture (2022) with transformer-inspired designs")
    print("   - LayerNorm + GELU activation (better than BatchNorm + ReLU)")
    print("   - Better accuracy/compute tradeoff than ResNet")
    print("   - 768-dim features (more compact than ResNet's 2048)")








