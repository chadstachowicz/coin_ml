"""
Strike Type Classification with ConvNeXt-Small

Classifies US coins as either Proof or Circulation strikes using dual-image
input (obverse + reverse) with optional year/denomination conditioning.

Strike Type Classes:
- Circulation: Business strike coins intended for commerce
- Proof: Special high-quality coins struck for collectors

Visual Differences:
- Proof coins have mirror-like fields, frosted devices, sharp details
- Circulation strikes have standard finishes, may show wear
- Proof coins often have deeper, more reflective surfaces

Architecture:
- ConvNeXt-Small backbone (pretrained on ImageNet)
- Dual image input (obverse + reverse)
- Optional year/denomination embeddings
- Binary classification head with softmax output

Usage:
  python coin_classifier_strike.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
from datetime import datetime
from tqdm import tqdm
from collections import Counter
import re


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_DIR = 'davidlawrence_dataset'
OUTPUT_DIR = 'models'
LOG_DIR = 'runs/strike_' + datetime.now().strftime('%Y%m%d_%H%M%S')

# Model hyperparameters
IMAGE_SIZE = 256  # Smaller since proof/circulation is about surface finish, not fine details
BATCH_SIZE = 32   # Can use larger batch with smaller images
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
FREEZE_BACKBONE = True
UNFREEZE_EPOCH = 3

# Class balancing - important since Proof is much smaller than Circulation
USE_WEIGHTED_SAMPLING = True
UNDERSAMPLE_CIRCULATION = 0.5  # Keep only 50% of circulation strikes to speed up training

# Year conditioning (disabled for strike type - visual only)
USE_YEAR_CONDITIONING = False
YEAR_EMBEDDING_DIM = 32
YEAR_MIN = 1793
YEAR_MAX = 2025

# Denomination conditioning (disabled for strike type - visual only)
USE_DENOM_CONDITIONING = False
DENOM_EMBEDDING_DIM = 16

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
print("CONVNEXT-SMALL STRIKE TYPE CLASSIFICATION")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Data directory: {DATA_DIR}")
print(f"Weighted sampling: {USE_WEIGHTED_SAMPLING}")
print(f"Year conditioning: {USE_YEAR_CONDITIONING}")
if USE_YEAR_CONDITIONING:
    print(f"  Year range: {YEAR_MIN}-{YEAR_MAX}")
    print(f"  Year embedding dim: {YEAR_EMBEDDING_DIM}")
print(f"Denomination conditioning: {USE_DENOM_CONDITIONING}")
if USE_DENOM_CONDITIONING:
    print(f"  Denom embedding dim: {DENOM_EMBEDDING_DIM}")
print("="*70)


# ============================================================================
# STRIKE TYPE DEFINITIONS
# ============================================================================

# Strike type classes
STRIKE_TYPES = ['Circulation', 'Proof']

# Mapping from strike type to description
STRIKE_TYPE_INFO = {
    'Circulation': 'Business strike for commerce',
    'Proof': 'Collector proof strike'
}

# Common US coin denominations (same as mintmark classifier)
DENOMINATION_MAP = {
    # Half cents
    '1-2C': 'half_cent', '1/2C': 'half_cent', 'H1C': 'half_cent',
    # Cents
    '1C': 'cent', '1c': 'cent',
    # Two cents
    '2C': 'two_cent', '2c': 'two_cent',
    # Three cents (silver and nickel)
    '3CS': 'three_cent_silver', '3cS': 'three_cent_silver',
    '3CN': 'three_cent_nickel', '3cN': 'three_cent_nickel',
    # Half dimes / Nickels
    'H10C': 'half_dime', 'H10c': 'half_dime',
    '5C': 'nickel', '5c': 'nickel',
    # Dimes
    '10C': 'dime', '10c': 'dime',
    # Twenty cents
    '20C': 'twenty_cent', '20c': 'twenty_cent',
    # Quarters
    '25C': 'quarter', '25c': 'quarter',
    # Half dollars
    '50C': 'half_dollar', '50c': 'half_dollar',
    # Dollars (silver/clad)
    '$1': 'dollar', '1$': 'dollar',
    # Trade dollars
    'T$1': 'trade_dollar', 'T$': 'trade_dollar',
    # Gold coins
    'G$1': 'gold_dollar', 'G1$': 'gold_dollar',
    '$2.50': 'quarter_eagle', '$2-50': 'quarter_eagle', '$2-1-2': 'quarter_eagle',
    '$3': 'three_dollar_gold',
    '$5': 'half_eagle',
    '$10': 'eagle',
    '$20': 'double_eagle',
    '$50': 'fifty_dollar_gold',
}

DENOMINATIONS = list(set(DENOMINATION_MAP.values()))

def normalize_denomination(denom_str):
    """Normalize denomination string to standard category."""
    if not denom_str:
        return 'unknown'
    
    denom = str(denom_str).strip().upper().replace(' ', '')
    
    if denom in DENOMINATION_MAP:
        return DENOMINATION_MAP[denom]
    
    for key, val in DENOMINATION_MAP.items():
        if key.upper() == denom:
            return val
    
    if '$' in denom_str:
        if '2' in denom_str and ('1/2' in denom_str or '2.5' in denom_str or '2-1' in denom_str):
            return 'quarter_eagle'
        elif '20' in denom_str:
            return 'double_eagle'
        elif '10' in denom_str:
            return 'eagle'
        elif '50' in denom_str:
            return 'fifty_dollar_gold'
        elif '5' in denom_str:
            return 'half_eagle'
        elif '3' in denom_str:
            return 'three_dollar_gold'
        elif '1' in denom_str:
            if 'G' in denom_str.upper():
                return 'gold_dollar'
            return 'dollar'
    
    return 'unknown'


# ============================================================================
# DATASET
# ============================================================================

class StrikeTypeDataset(Dataset):
    """Dataset for strike type classification (Proof vs Circulation)."""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # Fixed class mappings for binary classification
        self.class_to_idx = {'Circulation': 0, 'Proof': 1}
        self.idx_to_class = {0: 'Circulation', 1: 'Proof'}
        
        # Denomination mappings (built from dataset)
        self.denom_to_idx = {}
        self.idx_to_denom = {}
        denoms_found = set()
        
        temp_samples = []
        
        # Find strike type folders (Circulation and Proof)
        strike_folders = sorted([d for d in self.data_dir.iterdir() 
                                 if d.is_dir() and d.name in STRIKE_TYPES])
        
        for strike_folder in strike_folders:
            strike_type = strike_folder.name
            class_idx = self.class_to_idx[strike_type]
            
            # Find all grade subfolders
            grade_folders = sorted([g for g in strike_folder.iterdir() if g.is_dir()])
            
            for grade_folder in grade_folders:
                grade = grade_folder.name
                obverse_dir = grade_folder / 'obverse'
                reverse_dir = grade_folder / 'reverse'
                
                if not obverse_dir.exists() or not reverse_dir.exists():
                    continue
                
                obverse_images = sorted([f for f in obverse_dir.glob('*.jpg') if f.is_file()])
                
                for obverse_img in obverse_images:
                    reverse_img = reverse_dir / obverse_img.name
                    
                    if reverse_img.exists():
                        # Parse filename: YEAR_MINTMARK_DENOM_GRADE_SERVICE_CERT.jpg
                        parts = obverse_img.stem.split('_')
                        year = parts[0] if len(parts) >= 1 else 'unknown'
                        raw_denom = parts[2] if len(parts) >= 3 else 'unknown'
                        denom = normalize_denomination(raw_denom)
                        denoms_found.add(denom)
                        
                        temp_samples.append({
                            'obverse': obverse_img,
                            'reverse': reverse_img,
                            'strike_type': strike_type,
                            'class_idx': class_idx,
                            'grade': grade,
                            'year': year,
                            'denom': denom,
                            'raw_denom': raw_denom
                        })
        
        # Build denomination index mapping
        for denom_idx, denom in enumerate(sorted(denoms_found)):
            self.denom_to_idx[denom] = denom_idx
            self.idx_to_denom[denom_idx] = denom
        
        self.num_denoms = len(self.denom_to_idx)
        
        # Add denom_idx to each sample
        for sample in temp_samples:
            sample['denom_idx'] = self.denom_to_idx[sample['denom']]
        
        self.num_classes = len(self.class_to_idx)
        
        # Split dataset (stratified to maintain class balance)
        np.random.seed(42)
        
        # Separate by class for stratified split
        circulation_samples = [s for s in temp_samples if s['strike_type'] == 'Circulation']
        proof_samples = [s for s in temp_samples if s['strike_type'] == 'Proof']
        
        # Undersample circulation strikes to speed up training
        if UNDERSAMPLE_CIRCULATION < 1.0:
            np.random.seed(42)  # Reproducible undersampling
            n_keep = int(len(circulation_samples) * UNDERSAMPLE_CIRCULATION)
            indices = np.random.permutation(len(circulation_samples))[:n_keep]
            circulation_samples = [circulation_samples[i] for i in sorted(indices)]
            print(f"  Undersampled Circulation: kept {n_keep} of {len(temp_samples) - len(proof_samples)} ({UNDERSAMPLE_CIRCULATION*100:.0f}%)")
        
        def split_data(samples):
            indices = np.random.permutation(len(samples))
            n_train = int(0.8 * len(samples))
            n_val = int(0.1 * len(samples))
            
            if split == 'train':
                return [samples[i] for i in indices[:n_train]]
            elif split == 'val':
                return [samples[i] for i in indices[n_train:n_train + n_val]]
            else:  # test
                return [samples[i] for i in indices[n_train + n_val:]]
        
        self.samples = split_data(circulation_samples) + split_data(proof_samples)
        
        # Shuffle the combined samples
        np.random.shuffle(self.samples)
        
        # Compute class weights for balanced sampling
        class_counts = Counter([s['class_idx'] for s in self.samples])
        total = len(self.samples)
        self.class_weights = {c: total / count for c, count in class_counts.items()}
        self.sample_weights = [self.class_weights[s['class_idx']] for s in self.samples]
        
        # Print statistics
        print(f"\n{split.upper()}: {len(self.samples)} samples")
        print(f"  Strike Type Classes: {self.num_classes}")
        for strike_type, idx in sorted(self.class_to_idx.items(), key=lambda x: x[1]):
            count = class_counts.get(idx, 0)
            pct = 100 * count / total if total > 0 else 0
            info = STRIKE_TYPE_INFO.get(strike_type, strike_type)
            print(f"    {strike_type:12s} ({info}): {count:6d} ({pct:5.1f}%)")
        
        # Print grade distribution
        grade_counts = Counter([s['grade'] for s in self.samples])
        print(f"  Grades: {len(grade_counts)}")
        for grade, count in sorted(grade_counts.items(), key=lambda x: -x[1])[:10]:
            pct = 100 * count / total if total > 0 else 0
            print(f"    {grade:10s}: {count:5d} ({pct:5.1f}%)")
        
        # Print denomination distribution
        denom_counts = Counter([s['denom'] for s in self.samples])
        print(f"  Denominations: {self.num_denoms}")
        for denom, count in sorted(denom_counts.items(), key=lambda x: -x[1])[:10]:
            pct = 100 * count / total if total > 0 else 0
            print(f"    {denom:20s}: {count:5d} ({pct:5.1f}%)")
        if len(denom_counts) > 10:
            print(f"    ... and {len(denom_counts) - 10} more")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        obverse = Image.open(sample['obverse']).convert('RGB')
        reverse = Image.open(sample['reverse']).convert('RGB')
        
        if self.transform:
            obverse = self.transform(obverse)
            reverse = self.transform(reverse)
        
        label = torch.tensor(sample['class_idx'], dtype=torch.long)
        
        # Parse and normalize year
        try:
            year = int(sample['year'])
        except (ValueError, TypeError):
            year = (YEAR_MIN + YEAR_MAX) // 2
        
        year = max(YEAR_MIN, min(YEAR_MAX, year))
        year_normalized = (year - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)
        year_tensor = torch.tensor(year_normalized, dtype=torch.float32)
        
        year_idx = year - YEAR_MIN
        year_idx_tensor = torch.tensor(year_idx, dtype=torch.long)
        
        denom_idx = sample['denom_idx']
        denom_idx_tensor = torch.tensor(denom_idx, dtype=torch.long)
        
        return obverse, reverse, label, year_tensor, year_idx_tensor, denom_idx_tensor


# ============================================================================
# DATA TRANSFORMS
# ============================================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(degrees=10, fill=(255, 255, 255)),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
        fill=(255, 255, 255)
    ),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.02
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================================================
# CONVNEXT STRIKE TYPE CLASSIFIER
# ============================================================================

class StrikeTypeConvNeXt(nn.Module):
    """
    ConvNeXt-Small for strike type classification (Proof vs Circulation).
    
    Uses dual image input (obverse + reverse) with separate backbones.
    Year and denomination can be provided as additional context.
    
    Binary classification with softmax output.
    """
    
    def __init__(self, num_classes=2, freeze_backbone=True, 
                 use_year=True, year_embedding_dim=32, num_years=None,
                 use_denom=True, denom_embedding_dim=16, num_denoms=None):
        super(StrikeTypeConvNeXt, self).__init__()
        
        self.num_classes = num_classes
        self.use_year = use_year
        self.use_denom = use_denom
        
        # Year embedding
        if use_year:
            if num_years is None:
                num_years = YEAR_MAX - YEAR_MIN + 1
            self.year_embedding = nn.Embedding(num_years, year_embedding_dim)
            self.year_embedding_dim = year_embedding_dim
            print(f"✓ Year embedding: {num_years} years → {year_embedding_dim}D")
        
        # Denomination embedding
        if use_denom:
            if num_denoms is None:
                num_denoms = 20
            self.denom_embedding = nn.Embedding(num_denoms, denom_embedding_dim)
            self.denom_embedding_dim = denom_embedding_dim
            print(f"✓ Denomination embedding: {num_denoms} types → {denom_embedding_dim}D")
        
        # Load pretrained ConvNeXt-Small
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
        obverse_convnext = convnext_small(weights=weights)
        reverse_convnext = convnext_small(weights=weights)
        
        # Extract feature extractor parts
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
        fusion_input_dim = self.feature_dim * 2  # 1536 from images
        if use_year:
            fusion_input_dim += year_embedding_dim
        if use_denom:
            fusion_input_dim += denom_embedding_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def unfreeze_backbone(self):
        """Unfreeze the ConvNeXt backbone for fine-tuning."""
        for param in self.obverse_features.parameters():
            param.requires_grad = True
        for param in self.reverse_features.parameters():
            param.requires_grad = True
        print("✓ ConvNeXt backbone unfrozen")
    
    def forward(self, obverse, reverse, year_idx=None, denom_idx=None):
        # Extract image features
        obverse_feat = self.obverse_avgpool(self.obverse_features(obverse))
        obverse_feat = obverse_feat.view(obverse.size(0), -1)
        
        reverse_feat = self.reverse_avgpool(self.reverse_features(reverse))
        reverse_feat = reverse_feat.view(reverse.size(0), -1)
        
        combined = torch.cat([obverse_feat, reverse_feat], dim=1)
        
        if self.use_year and year_idx is not None:
            year_emb = self.year_embedding(year_idx)
            combined = torch.cat([combined, year_emb], dim=1)
        
        if self.use_denom and denom_idx is not None:
            denom_emb = self.denom_embedding(denom_idx)
            combined = torch.cat([combined, denom_emb], dim=1)
        
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        
        return logits


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs, 
                use_year=True, use_denom=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    
    for batch in pbar:
        obverse, reverse, labels, year_norm, year_idx, denom_idx = batch
        
        obverse = obverse.to(device)
        reverse = reverse.to(device)
        labels = labels.to(device)
        year_idx = year_idx.to(device) if use_year else None
        denom_idx = denom_idx.to(device) if use_denom else None
        
        optimizer.zero_grad()
        
        logits = model(obverse, reverse, year_idx, denom_idx)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * obverse.size(0)
        
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.1f}%'
        })
    
    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, loader, criterion, device, use_year=True, use_denom=True):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        
        for batch in pbar:
            obverse, reverse, labels, year_norm, year_idx, denom_idx = batch
            
            obverse = obverse.to(device)
            reverse = reverse.to(device)
            labels = labels.to(device)
            year_idx = year_idx.to(device) if use_year else None
            denom_idx = denom_idx.to(device) if use_denom else None
            
            logits = model(obverse, reverse, year_idx, denom_idx)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * obverse.size(0)
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.1f}%'
            })
    
    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels


def compute_per_class_accuracy(preds, labels, idx_to_class):
    """Compute accuracy for each class."""
    preds = np.array(preds)
    labels = np.array(labels)
    
    results = {}
    for idx, class_name in idx_to_class.items():
        mask = labels == idx
        if mask.sum() > 0:
            class_correct = (preds[mask] == labels[mask]).sum()
            class_total = mask.sum()
            results[class_name] = {
                'accuracy': 100 * class_correct / class_total,
                'correct': int(class_correct),
                'total': int(class_total)
            }
    
    return results


def compute_precision_recall_f1(preds, labels, idx_to_class):
    """Compute precision, recall, and F1 for binary classification."""
    preds = np.array(preds)
    labels = np.array(labels)
    
    results = {}
    for idx, class_name in idx_to_class.items():
        # True positives: predicted class_idx and actually class_idx
        tp = ((preds == idx) & (labels == idx)).sum()
        # False positives: predicted class_idx but not actually class_idx
        fp = ((preds == idx) & (labels != idx)).sum()
        # False negatives: not predicted class_idx but actually class_idx
        fn = ((preds != idx) & (labels == idx)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[class_name] = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\nCreating datasets...")
    
    # Check if dataset exists
    if not Path(DATA_DIR).exists():
        print(f"\n❌ Error: Dataset not found at '{DATA_DIR}'")
        exit(1)
    
    train_dataset = StrikeTypeDataset(DATA_DIR, split='train', transform=train_transform)
    val_dataset = StrikeTypeDataset(DATA_DIR, split='val', transform=val_transform)
    test_dataset = StrikeTypeDataset(DATA_DIR, split='test', transform=val_transform)
    
    # Create data loaders
    if USE_WEIGHTED_SAMPLING:
        sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
    
    # Create model
    num_classes = train_dataset.num_classes
    num_years = YEAR_MAX - YEAR_MIN + 1
    num_denoms = train_dataset.num_denoms
    
    model = StrikeTypeConvNeXt(
        num_classes=num_classes,
        freeze_backbone=FREEZE_BACKBONE,
        use_year=USE_YEAR_CONDITIONING,
        year_embedding_dim=YEAR_EMBEDDING_DIM,
        num_years=num_years,
        use_denom=USE_DENOM_CONDITIONING,
        denom_embedding_dim=DENOM_EMBEDDING_DIM,
        num_denoms=num_denoms
    )
    model = model.to(DEVICE)
    
    print(f"\nModel created (ConvNeXt-Small):")
    print(f"  Task: Strike Type Classification (Proof vs Circulation)")
    print(f"  Year conditioning: {USE_YEAR_CONDITIONING}")
    if USE_YEAR_CONDITIONING:
        print(f"    Year range: {YEAR_MIN}-{YEAR_MAX} ({num_years} years)")
        print(f"    Year embedding dim: {YEAR_EMBEDDING_DIM}")
    print(f"  Denomination conditioning: {USE_DENOM_CONDITIONING}")
    if USE_DENOM_CONDITIONING:
        print(f"    Denominations: {num_denoms} types")
        print(f"    Denom embedding dim: {DENOM_EMBEDDING_DIM}")
    print(f"  Classes: {num_classes}")
    for idx, class_name in train_dataset.idx_to_class.items():
        info = STRIKE_TYPE_INFO.get(class_name, class_name)
        print(f"    {idx}: {class_name} ({info})")
    print(f"  Feature dim: 768 (ConvNeXt-Small)")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss function
    if USE_WEIGHTED_SAMPLING:
        criterion = nn.CrossEntropyLoss()
    else:
        class_weights = torch.tensor(
            [train_dataset.class_weights.get(i, 1.0) for i in range(num_classes)],
            dtype=torch.float32
        ).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    writer = SummaryWriter(LOG_DIR)
    
    # Training loop
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_f1 = 0.0  # Track best F1 score (better than accuracy for imbalanced data)
    best_model_path = os.path.join(OUTPUT_DIR, 'coin_strike_best.pth')
    
    print("\n" + "="*70)
    print("STARTING STRIKE TYPE CLASSIFICATION TRAINING")
    print("="*70)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Unfreeze backbone after initial epochs
        if FREEZE_BACKBONE and epoch == UNFREEZE_EPOCH:
            print(f"\n🔓 Unfreezing ConvNeXt backbone")
            model.unfreeze_backbone()
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE/10, weight_decay=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3, verbose=True
            )
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch, NUM_EPOCHS,
            use_year=USE_YEAR_CONDITIONING, use_denom=USE_DENOM_CONDITIONING
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, DEVICE,
            use_year=USE_YEAR_CONDITIONING, use_denom=USE_DENOM_CONDITIONING
        )
        
        # Compute F1 scores first (needed for scheduler and best model selection)
        pr_results = compute_precision_recall_f1(val_preds, val_labels, val_dataset.idx_to_class)
        macro_f1 = np.mean([stats['f1'] for stats in pr_results.values()])
        
        scheduler.step(macro_f1)  # Step on F1, not accuracy
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.1f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.1f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # Per-class accuracy
        per_class = compute_per_class_accuracy(val_preds, val_labels, val_dataset.idx_to_class)
        print(f"  Per-class accuracy:")
        for class_name, stats in sorted(per_class.items()):
            print(f"    {class_name:12s}: {stats['accuracy']:5.1f}% ({stats['correct']}/{stats['total']})")
        
        # Precision, Recall, F1 (better metrics for imbalanced data)
        print(f"  Precision / Recall / F1:")
        for class_name, stats in sorted(pr_results.items()):
            print(f"    {class_name:12s}: P={stats['precision']:5.1f}% R={stats['recall']:5.1f}% F1={stats['f1']:5.1f}%")
        print(f"  Macro F1: {macro_f1:.1f}%")
        
        # Log to TensorBoard
        for class_name, stats in pr_results.items():
            writer.add_scalar(f'Precision/{class_name}', stats['precision'], epoch)
            writer.add_scalar(f'Recall/{class_name}', stats['recall'], epoch)
            writer.add_scalar(f'F1/{class_name}', stats['f1'], epoch)
        writer.add_scalar('F1/macro', macro_f1, epoch)
        
        # Save best model (use F1 instead of accuracy for imbalanced data)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_f1': macro_f1,
                'val_loss': val_loss,
                'class_to_idx': train_dataset.class_to_idx,
                'idx_to_class': train_dataset.idx_to_class,
                'num_classes': num_classes,
                'image_size': IMAGE_SIZE,
                'backbone': 'convnext_small',
                'use_year_conditioning': USE_YEAR_CONDITIONING,
                'year_embedding_dim': YEAR_EMBEDDING_DIM if USE_YEAR_CONDITIONING else None,
                'year_min': YEAR_MIN,
                'year_max': YEAR_MAX,
                'use_denom_conditioning': USE_DENOM_CONDITIONING,
                'denom_embedding_dim': DENOM_EMBEDDING_DIM if USE_DENOM_CONDITIONING else None,
                'denom_to_idx': train_dataset.denom_to_idx,
                'idx_to_denom': train_dataset.idx_to_denom,
                'num_denoms': num_denoms
            }, best_model_path)
            print(f"  ✓ New best! F1: {macro_f1:.1f}%")
    
    writer.close()
    
    # Final evaluation on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, DEVICE,
        use_year=USE_YEAR_CONDITIONING, use_denom=USE_DENOM_CONDITIONING
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.1f}%")
    
    # Per-class accuracy on test set
    per_class = compute_per_class_accuracy(test_preds, test_labels, test_dataset.idx_to_class)
    print(f"\nPer-class Test Accuracy:")
    for class_name, stats in sorted(per_class.items()):
        info = STRIKE_TYPE_INFO.get(class_name, class_name)
        print(f"  {class_name:12s} ({info}): {stats['accuracy']:5.1f}% ({stats['correct']}/{stats['total']})")
    
    # Precision, Recall, F1
    pr_results = compute_precision_recall_f1(test_preds, test_labels, test_dataset.idx_to_class)
    print(f"\nPrecision / Recall / F1:")
    for class_name, stats in sorted(pr_results.items()):
        print(f"  {class_name:12s}: P={stats['precision']:5.1f}% R={stats['recall']:5.1f}% F1={stats['f1']:5.1f}%")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    from collections import defaultdict
    confusion = defaultdict(lambda: defaultdict(int))
    for pred, label in zip(test_preds, test_labels):
        pred_class = test_dataset.idx_to_class[pred]
        true_class = test_dataset.idx_to_class[label]
        confusion[true_class][pred_class] += 1
    
    classes = sorted(test_dataset.class_to_idx.keys())
    header = "True\\Pred    " + " ".join([f"{c:>12s}" for c in classes])
    print(header)
    print("-" * len(header))
    
    for true_class in classes:
        row = f"{true_class:12s} "
        for pred_class in classes:
            count = confusion[true_class][pred_class]
            row += f"{count:12d} "
        print(row)
    
    # Save results
    results = {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'per_class_accuracy': per_class,
        'precision_recall_f1': {k: {kk: float(vv) for kk, vv in v.items()} 
                                for k, v in pr_results.items()},
        'class_to_idx': train_dataset.class_to_idx,
        'best_epoch': checkpoint['epoch']
    }
    
    with open(os.path.join(OUTPUT_DIR, 'strike_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, 'history_strike.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best Validation F1: {best_f1:.1f}%")
    print(f"Test Accuracy: {test_acc:.1f}%")
    
    # Compute test F1
    test_pr = compute_precision_recall_f1(test_preds, test_labels, test_dataset.idx_to_class)
    test_macro_f1 = np.mean([stats['f1'] for stats in test_pr.values()])
    print(f"Test Macro F1: {test_macro_f1:.1f}%")
    print(f"Model saved: {best_model_path}")
    print(f"Results saved: {os.path.join(OUTPUT_DIR, 'strike_results.json')}")

