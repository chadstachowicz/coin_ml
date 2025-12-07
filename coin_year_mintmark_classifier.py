"""
Coin Year and Mint Mark Classifier

Multi-task learning model to predict:
1. Year of the coin (classification)
2. Mint mark (P/None, D, S, O, CC, W)

Uses dual-image input (obverse + reverse) with ResNet backbone.
"""

import os
import re
import json
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# ============================================================================
# CONFIGURATION
# ============================================================================

# Find all davidlawrence_coins* directories with data/ subdirectory
DATA_DIRS = sorted([
    d for d in glob.glob('davidlawrence_coins*') 
    if os.path.isdir(d) and os.path.exists(os.path.join(d, 'data'))
])
print(f"Found {len(DATA_DIRS)} data directories: {DATA_DIRS}")

OUTPUT_DIR = 'models'
LOG_DIR = 'runs/year_mintmark'

# Model hyperparameters
IMAGE_SIZE = 448
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
FREEZE_BACKBONE = True
UNFREEZE_EPOCH = 15

# Mint marks
MINT_MARKS = ['P', 'D', 'S', 'O', 'CC', 'W']  # P = Philadelphia (no mark)
MINT_MARK_TO_IDX = {mm: i for i, mm in enumerate(MINT_MARKS)}

# Year range (will be determined from data)
MIN_YEAR = 1793
MAX_YEAR = 2024

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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_description(description):
    """
    Parse year and mint mark from description.
    
    Examples:
        "1920-D 25c PCGS MS61" -> (1920, 'D')
        "1919 25c PCGS AU55" -> (1919, 'P')
        "1917-S Type 1 25c PCGS MS64" -> (1917, 'S')
        "1878-CC $1 PCGS MS63" -> (1878, 'CC')
    """
    if not description:
        return None, None
    
    # Pattern: year optionally followed by -mintmark
    # Mint marks: D, S, O, CC, W, P (rare)
    pattern = r'^(\d{4})(?:-([DSOCCW]{1,2}))?'
    
    match = re.match(pattern, description.strip())
    if match:
        year = int(match.group(1))
        mint_mark = match.group(2) if match.group(2) else 'P'  # Default to Philadelphia
        return year, mint_mark
    
    return None, None


def get_year_class(year, min_year, num_years):
    """Convert year to class index."""
    return year - min_year


def get_year_from_class(class_idx, min_year):
    """Convert class index back to year."""
    return class_idx + min_year


# ============================================================================
# DATASET
# ============================================================================

class YearMintMarkDataset(Dataset):
    """Dataset for year and mint mark classification."""
    
    def __init__(self, data_dirs, split='train', transform=None, min_year=None, max_year=None):
        self.transform = transform
        self.samples = []
        
        # Collect all samples
        all_samples = []
        years_found = set()
        mint_marks_found = Counter()
        
        for data_dir in data_dirs:
            data_path = Path(data_dir) / 'data'
            images_path = Path(data_dir) / 'images'
            
            if not data_path.exists():
                print(f"  Skipping {data_dir} (not found)")
                continue
            
            json_files = list(data_path.glob('*.json'))
            print(f"  {data_dir}: {len(json_files)} JSON files")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    description = data.get('description', '')
                    year, mint_mark = parse_description(description)
                    
                    if year is None:
                        continue
                    
                    # Get images
                    images = data.get('images', [])
                    if len(images) < 2:
                        continue
                    
                    # Find obverse (image_0 or image_1) and reverse
                    obverse_img = None
                    reverse_img = None
                    
                    for img_path in images:
                        full_path = Path(img_path)
                        if not full_path.exists():
                            # Try relative to workspace
                            full_path = Path(data_dir).parent / img_path
                        
                        if full_path.exists():
                            if 'image_0' in str(img_path) or 'image_1' in str(img_path):
                                if obverse_img is None:
                                    obverse_img = full_path
                                elif reverse_img is None:
                                    reverse_img = full_path
                            elif 'image_2' in str(img_path) or 'image_3' in str(img_path):
                                if reverse_img is None:
                                    reverse_img = full_path
                    
                    if obverse_img is None or reverse_img is None:
                        continue
                    
                    years_found.add(year)
                    mint_marks_found[mint_mark] += 1
                    
                    all_samples.append({
                        'obverse': obverse_img,
                        'reverse': reverse_img,
                        'year': year,
                        'mint_mark': mint_mark,
                        'description': description
                    })
                    
                except Exception as e:
                    continue
        
        print(f"\nTotal samples found: {len(all_samples)}")
        print(f"Years: {min(years_found)} - {max(years_found)} ({len(years_found)} unique)")
        print(f"Mint marks: {dict(mint_marks_found)}")
        
        # Determine year range
        if min_year is None:
            self.min_year = min(years_found)
        else:
            self.min_year = min_year
        
        if max_year is None:
            self.max_year = max(years_found)
        else:
            self.max_year = max_year
        
        self.num_years = self.max_year - self.min_year + 1
        
        # Filter samples to year range
        all_samples = [s for s in all_samples if self.min_year <= s['year'] <= self.max_year]
        
        # Split data (80% train, 20% test)
        np.random.seed(42)
        indices = np.random.permutation(len(all_samples))
        split_idx = int(len(all_samples) * 0.8)
        
        if split == 'train':
            selected_indices = indices[:split_idx]
        else:
            selected_indices = indices[split_idx:]
        
        self.samples = [all_samples[i] for i in selected_indices]
        
        # Convert labels
        for sample in self.samples:
            sample['year_class'] = get_year_class(sample['year'], self.min_year, self.num_years)
            sample['mint_mark_idx'] = MINT_MARK_TO_IDX.get(sample['mint_mark'], 0)
        
        # Stats
        year_counts = Counter([s['year'] for s in self.samples])
        mint_counts = Counter([s['mint_mark'] for s in self.samples])
        
        print(f"\n{split.upper()} SET: {len(self.samples)} samples")
        print(f"  Year range: {self.min_year} - {self.max_year} ({self.num_years} classes)")
        print(f"  Mint marks: {dict(mint_counts)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        obverse = Image.open(sample['obverse']).convert('RGB')
        reverse = Image.open(sample['reverse']).convert('RGB')
        
        if self.transform:
            obverse = self.transform(obverse)
            reverse = self.transform(reverse)
        
        year_class = torch.tensor(sample['year_class'], dtype=torch.long)
        mint_mark_idx = torch.tensor(sample['mint_mark_idx'], dtype=torch.long)
        
        return obverse, reverse, year_class, mint_mark_idx


# ============================================================================
# MODEL
# ============================================================================

class YearMintMarkResNet(nn.Module):
    """
    Multi-task ResNet for year and mint mark classification.
    
    Architecture:
    - Dual ResNet-50 backbone (obverse + reverse)
    - Shared fusion layer
    - Separate heads for year and mint mark
    """
    
    def __init__(self, num_years, num_mint_marks=6, freeze_backbone=True):
        super(YearMintMarkResNet, self).__init__()
        
        self.num_years = num_years
        self.num_mint_marks = num_mint_marks
        
        # Load pretrained ResNet-50
        weights = ResNet50_Weights.IMAGENET1K_V2
        obverse_resnet = resnet50(weights=weights)
        reverse_resnet = resnet50(weights=weights)
        
        # Encoders (remove final FC layer)
        self.obverse_encoder = nn.Sequential(*list(obverse_resnet.children())[:-1])
        self.reverse_encoder = nn.Sequential(*list(reverse_resnet.children())[:-1])
        
        if freeze_backbone:
            for param in self.obverse_encoder.parameters():
                param.requires_grad = False
            for param in self.reverse_encoder.parameters():
                param.requires_grad = False
            print("✓ Backbone frozen")
        
        # Feature dimension
        self.feature_dim = 2048
        
        # Shared fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Year classification head
        self.year_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_years)
        )
        
        # Mint mark classification head
        self.mint_mark_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_mint_marks)
        )
    
    def unfreeze_backbone(self):
        for param in self.obverse_encoder.parameters():
            param.requires_grad = True
        for param in self.reverse_encoder.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen")
    
    def forward(self, obverse, reverse):
        # Encode images
        obverse_feat = self.obverse_encoder(obverse).view(obverse.size(0), -1)
        reverse_feat = self.reverse_encoder(reverse).view(reverse.size(0), -1)
        
        # Concatenate features
        combined = torch.cat([obverse_feat, reverse_feat], dim=1)
        
        # Shared fusion
        fused = self.fusion(combined)
        
        # Task-specific heads
        year_logits = self.year_head(fused)
        mint_mark_logits = self.mint_mark_head(fused)
        
        return year_logits, mint_mark_logits


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
# TRAINING
# ============================================================================

def train_epoch(model, loader, year_criterion, mint_criterion, optimizer, device, epoch):
    model.train()
    
    total_loss = 0
    year_correct = 0
    mint_correct = 0
    total = 0
    
    # Track year accuracy within N years
    year_within_1 = 0
    year_within_5 = 0
    year_within_10 = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [Train]')
    for batch in pbar:
        obverse, reverse, year_labels, mint_labels = batch
        obverse = obverse.to(device)
        reverse = reverse.to(device)
        year_labels = year_labels.to(device)
        mint_labels = mint_labels.to(device)
        
        optimizer.zero_grad()
        
        year_logits, mint_logits = model(obverse, reverse)
        
        # Losses (weight year more heavily)
        year_loss = year_criterion(year_logits, year_labels)
        mint_loss = mint_criterion(mint_logits, mint_labels)
        loss = 1.0 * year_loss + 0.5 * mint_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * obverse.size(0)
        
        # Accuracy
        year_pred = year_logits.argmax(dim=1)
        mint_pred = mint_logits.argmax(dim=1)
        
        year_correct += (year_pred == year_labels).sum().item()
        mint_correct += (mint_pred == mint_labels).sum().item()
        
        # Year within N
        year_diff = torch.abs(year_pred - year_labels)
        year_within_1 += (year_diff <= 1).sum().item()
        year_within_5 += (year_diff <= 5).sum().item()
        year_within_10 += (year_diff <= 10).sum().item()
        
        total += obverse.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'year_acc': f'{100*year_correct/total:.1f}%',
            'mint_acc': f'{100*mint_correct/total:.1f}%'
        })
    
    return {
        'loss': total_loss / total,
        'year_acc': 100 * year_correct / total,
        'mint_acc': 100 * mint_correct / total,
        'year_within_1': 100 * year_within_1 / total,
        'year_within_5': 100 * year_within_5 / total,
        'year_within_10': 100 * year_within_10 / total
    }


def validate(model, loader, year_criterion, mint_criterion, device):
    model.eval()
    
    total_loss = 0
    year_correct = 0
    mint_correct = 0
    total = 0
    
    year_within_1 = 0
    year_within_5 = 0
    year_within_10 = 0
    
    with torch.no_grad():
        for batch in loader:
            obverse, reverse, year_labels, mint_labels = batch
            obverse = obverse.to(device)
            reverse = reverse.to(device)
            year_labels = year_labels.to(device)
            mint_labels = mint_labels.to(device)
            
            year_logits, mint_logits = model(obverse, reverse)
            
            year_loss = year_criterion(year_logits, year_labels)
            mint_loss = mint_criterion(mint_logits, mint_labels)
            loss = 1.0 * year_loss + 0.5 * mint_loss
            
            total_loss += loss.item() * obverse.size(0)
            
            year_pred = year_logits.argmax(dim=1)
            mint_pred = mint_logits.argmax(dim=1)
            
            year_correct += (year_pred == year_labels).sum().item()
            mint_correct += (mint_pred == mint_labels).sum().item()
            
            year_diff = torch.abs(year_pred - year_labels)
            year_within_1 += (year_diff <= 1).sum().item()
            year_within_5 += (year_diff <= 5).sum().item()
            year_within_10 += (year_diff <= 10).sum().item()
            
            total += obverse.size(0)
    
    return {
        'loss': total_loss / total,
        'year_acc': 100 * year_correct / total,
        'mint_acc': 100 * mint_correct / total,
        'year_within_1': 100 * year_within_1 / total,
        'year_within_5': 100 * year_within_5 / total,
        'year_within_10': 100 * year_within_10 / total
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("COIN YEAR & MINT MARK CLASSIFIER")
    print("="*70)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = YearMintMarkDataset(DATA_DIRS, split='train', transform=train_transform)
    val_dataset = YearMintMarkDataset(
        DATA_DIRS, split='val', transform=val_transform,
        min_year=train_dataset.min_year,
        max_year=train_dataset.max_year
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    
    # Model
    print("\nInitializing model...")
    model = YearMintMarkResNet(
        num_years=train_dataset.num_years,
        num_mint_marks=len(MINT_MARKS),
        freeze_backbone=FREEZE_BACKBONE
    )
    model = model.to(DEVICE)
    
    print(f"  Year classes: {train_dataset.num_years} ({train_dataset.min_year}-{train_dataset.max_year})")
    print(f"  Mint mark classes: {len(MINT_MARKS)} ({MINT_MARKS})")
    
    # Loss functions
    year_criterion = nn.CrossEntropyLoss()
    mint_criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # TensorBoard
    writer = SummaryWriter(LOG_DIR)
    
    # Training loop
    best_year_acc = 0
    best_model_path = os.path.join(OUTPUT_DIR, 'coin_year_mintmark_best.pth')
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    for epoch in range(NUM_EPOCHS):
        # Unfreeze backbone after N epochs
        if epoch == UNFREEZE_EPOCH and FREEZE_BACKBONE:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE * 0.1,
                weight_decay=0.01
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS - epoch
            )
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, year_criterion, mint_criterion,
            optimizer, DEVICE, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, year_criterion, mint_criterion, DEVICE)
        
        scheduler.step()
        
        # Log
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Year_Accuracy/train', train_metrics['year_acc'], epoch)
        writer.add_scalar('Year_Accuracy/val', val_metrics['year_acc'], epoch)
        writer.add_scalar('Mint_Accuracy/train', train_metrics['mint_acc'], epoch)
        writer.add_scalar('Mint_Accuracy/val', val_metrics['mint_acc'], epoch)
        writer.add_scalar('Year_Within_5/val', val_metrics['year_within_5'], epoch)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train: Year={train_metrics['year_acc']:.1f}%, Mint={train_metrics['mint_acc']:.1f}%")
        print(f"  Val:   Year={val_metrics['year_acc']:.1f}%, Mint={val_metrics['mint_acc']:.1f}%")
        print(f"  Year within ±1: {val_metrics['year_within_1']:.1f}%, ±5: {val_metrics['year_within_5']:.1f}%, ±10: {val_metrics['year_within_10']:.1f}%")
        
        # Save best model
        if val_metrics['year_acc'] > best_year_acc:
            best_year_acc = val_metrics['year_acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'year_acc': val_metrics['year_acc'],
                'mint_acc': val_metrics['mint_acc'],
                'year_within_5': val_metrics['year_within_5'],
                'min_year': train_dataset.min_year,
                'max_year': train_dataset.max_year,
                'num_years': train_dataset.num_years,
                'mint_marks': MINT_MARKS
            }, best_model_path)
            print(f"  ✓ New best! Year accuracy: {val_metrics['year_acc']:.1f}%")
    
    writer.close()
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best year accuracy: {best_year_acc:.1f}%")
    print(f"Model saved: {best_model_path}")
    print("\n💡 Key Metrics:")
    print(f"   - Year Accuracy: Exact year match")
    print(f"   - Year Within ±5: Predicted year within 5 years of actual")
    print(f"   - Mint Accuracy: Correct mint mark (P, D, S, O, CC, W)")

