"""
Details vs Straight Binary Classifier - ConvNeXt

A binary classifier to detect if a coin has been graded as "Details" 
(problem coins: cleaned, damaged, etc.) vs "Straight" (normal grade).

This is a simple binary classification problem - the actual grade doesn't matter,
only whether the coin has issues that would result in a Details designation.

Features:
- Dual-image input (obverse + reverse)
- ConvNeXt-Small backbone
- Binary classification (not ordinal)
- Balanced dataset training

Dataset structure expected:
  details_dataset/
    Details/
      obverse/
      reverse/
    Straight/
      obverse/
      reverse/
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_DIR = 'dlrc_details_dataset'
OUTPUT_DIR = 'models'
LOG_DIR = 'runs/details_classifier_' + datetime.now().strftime('%Y%m%d_%H%M%S')

# Model hyperparameters
IMAGE_SIZE = 700
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
FREEZE_BACKBONE = True
UNFREEZE_EPOCH = 5

# Class labels
CLASS_NAMES = ['Straight', 'Details']  # 0 = Straight, 1 = Details

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
print("DETAILS VS STRAIGHT BINARY CLASSIFIER (ConvNeXt)")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Classes: {CLASS_NAMES}")
print(f"Data directory: {DATA_DIR}")
print("="*70)


# ============================================================================
# DATASET
# ============================================================================

class DetailsDataset(Dataset):
    """Dataset for Details vs Straight binary classification."""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # Class mapping
        self.class_to_idx = {'Straight': 0, 'Details': 1}
        self.idx_to_class = {0: 'Straight', 1: 'Details'}
        
        # Collect all samples
        temp_samples = []
        
        for class_name in ['Straight', 'Details']:
            class_dir = self.data_dir / class_name
            obverse_dir = class_dir / 'obverse'
            reverse_dir = class_dir / 'reverse'
            
            if not obverse_dir.exists() or not reverse_dir.exists():
                print(f"⚠️ Missing directory for {class_name}")
                continue
            
            obverse_images = sorted([f for f in obverse_dir.glob('*.jpg') if f.is_file()])
            
            for obverse_img in obverse_images:
                reverse_img = reverse_dir / obverse_img.name
                
                if reverse_img.exists():
                    temp_samples.append({
                        'obverse': obverse_img,
                        'reverse': reverse_img,
                        'class_name': class_name,
                        'class_idx': self.class_to_idx[class_name]
                    })
        
        self.samples = temp_samples
        
        # 70/15/15 split
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        n_total = len(self.samples)
        n_train = int(0.70 * n_total)
        n_val = int(0.15 * n_total)
        
        if split == 'train':
            indices = indices[:n_train]
        elif split == 'val':
            indices = indices[n_train:n_train + n_val]
        else:  # test
            indices = indices[n_train + n_val:]
        
        self.samples = [self.samples[i] for i in indices]
        
        # Statistics
        from collections import Counter
        class_counts = Counter([s['class_name'] for s in self.samples])
        
        print(f"\n{split.upper()}: {len(self.samples)} samples")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")
    
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
        
        return obverse, reverse, label


# ============================================================================
# DATA TRANSFORMS
# ============================================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(degrees=5, fill=(255, 255, 255)),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05),
        fill=(255, 255, 255)
    ),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.02
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
# MODEL
# ============================================================================

class DetailsClassifierConvNeXt(nn.Module):
    """
    ConvNeXt-based binary classifier for Details vs Straight.
    
    Takes obverse and reverse images, outputs binary classification.
    """
    
    def __init__(self, freeze_backbone=True):
        super(DetailsClassifierConvNeXt, self).__init__()
        
        # Load pretrained ConvNeXt-Small
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
        obverse_convnext = convnext_small(weights=weights)
        reverse_convnext = convnext_small(weights=weights)
        
        # Extract feature extractors
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
        
        # Fusion layer (768 * 2 = 1536 input)
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # Classification head (binary output)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # 2 classes: Straight, Details
        )
    
    def unfreeze_backbone(self):
        """Unfreeze the ConvNeXt backbone for fine-tuning."""
        for param in self.obverse_features.parameters():
            param.requires_grad = True
        for param in self.reverse_features.parameters():
            param.requires_grad = True
        print("✓ ConvNeXt backbone unfrozen")
    
    def forward(self, obverse, reverse):
        # Encode images
        obverse_feat = self.obverse_avgpool(self.obverse_features(obverse))
        obverse_feat = obverse_feat.view(obverse.size(0), -1)
        
        reverse_feat = self.reverse_avgpool(self.reverse_features(reverse))
        reverse_feat = reverse_feat.view(reverse.size(0), -1)
        
        # Concatenate features
        combined = torch.cat([obverse_feat, reverse_feat], dim=1)
        
        # Fusion + classification
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        
        return logits


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
    
    for obverse, reverse, labels in pbar:
        obverse = obverse.to(device)
        reverse = reverse.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(obverse, reverse)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * obverse.size(0)
        
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.1f}%'})
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validating')
        for obverse, reverse, labels in pbar:
            obverse = obverse.to(device)
            reverse = reverse.to(device)
            labels = labels.to(device)
            
            logits = model(obverse, reverse)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * obverse.size(0)
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.1f}%'})
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    # Per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    class_accs = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = all_labels == class_idx
        if mask.sum() > 0:
            class_accs[class_name] = 100 * (all_preds[mask] == all_labels[mask]).mean()
    
    return epoch_loss, epoch_acc, class_accs


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\nCreating datasets (70/15/15 split)...")
    
    train_dataset = DetailsDataset(DATA_DIR, split='train', transform=train_transform)
    val_dataset = DetailsDataset(DATA_DIR, split='val', transform=val_transform)
    test_dataset = DetailsDataset(DATA_DIR, split='test', transform=val_transform)
    
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
    model = DetailsClassifierConvNeXt(freeze_backbone=FREEZE_BACKBONE)
    model = model.to(DEVICE)
    
    print(f"\nModel created (ConvNeXt-Small):")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  Feature dim: 768")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss function (CrossEntropy for binary classification)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # TensorBoard
    writer = SummaryWriter(LOG_DIR)
    
    # Training loop
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_acc = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, 'details_classifier_best.pth')
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
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
            model, train_loader, criterion, optimizer, DEVICE, epoch
        )
        
        # Validate
        val_loss, val_acc, class_accs = validate(
            model, val_loader, criterion, DEVICE
        )
        
        scheduler.step(val_acc)
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
        for class_name, acc in class_accs.items():
            writer.add_scalar(f'Accuracy_{class_name}/val', acc, epoch)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.1f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.1f}%")
        for class_name, acc in class_accs.items():
            print(f"    {class_name}: {acc:.1f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_accs': class_accs,
                'class_names': CLASS_NAMES,
                'class_to_idx': train_dataset.class_to_idx,
                'idx_to_class': train_dataset.idx_to_class,
                'backbone': 'convnext_small'
            }, best_model_path)
            print(f"  ✓ New best! Accuracy: {val_acc:.1f}%")
    
    writer.close()
    
    # Final evaluation on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_class_accs = validate(model, test_loader, criterion, DEVICE)
    
    print(f"\nTest Results:")
    print(f"  Overall Accuracy: {test_acc:.1f}%")
    for class_name, acc in test_class_accs.items():
        print(f"  {class_name}: {acc:.1f}%")
    
    # Save history
    with open(os.path.join(OUTPUT_DIR, 'history_details_classifier.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best Validation Accuracy: {best_acc:.1f}%")
    print(f"Test Accuracy: {test_acc:.1f}%")
    print(f"Model saved: {best_model_path}")






