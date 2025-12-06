"""
Multi-Task ResNet for Coin Grading with Company Prediction

Approach A: Multi-task learning with two heads
- Head 1: Predict coin grade (main task)
- Head 2: Predict grading company (auxiliary task)

This forces the model to learn features useful for both tasks,
potentially improving grade prediction by understanding company biases.

Features:
- Dual-image input (obverse + reverse)
- ResNet-50 backbone
- Two classification heads
- Combined loss function
"""

import torch
import torch.nn as nn
import torch.optim as optim
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
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_DIR = 'davidlawrence_dataset/Circulation'
OUTPUT_DIR = 'models'
LOG_DIR = 'runs/multitask_' + datetime.now().strftime('%Y%m%d_%H%M%S')

# Model hyperparameters
IMAGE_SIZE = 448
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
FREEZE_BACKBONE = True
UNFREEZE_EPOCH = 25

# Multi-task loss weights
GRADE_LOSS_WEIGHT = 1.0      # Main task
COMPANY_LOSS_WEIGHT = 0.3    # Auxiliary task (helps regularization)

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

print("="*60)
print("MULTI-TASK DUAL RESNET CLASSIFIER")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Grade loss weight: {GRADE_LOSS_WEIGHT}")
print(f"Company loss weight: {COMPANY_LOSS_WEIGHT}")
print("="*60)


# ============================================================================
# DATASET CLASS
# ============================================================================

class MultiTaskCoinDataset(Dataset):
    """Dataset that loads dual images and provides grade + company labels."""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.grade_to_idx = {}
        self.idx_to_grade = {}
        self.company_to_idx = {}
        self.idx_to_company = {}
        
        # First, collect all samples to find companies
        temp_samples = []
        companies = set()
        
        # We need to read JSON metadata to get grading service
        # Assuming structure: davidlawrence_dataset/Proof/<grade>/obverse/
        # And we can parse grading service from filename or need JSON
        
        grade_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for grade_idx, grade_folder in enumerate(grade_folders):
            grade_name = grade_folder.name
            self.grade_to_idx[grade_name] = grade_idx
            self.idx_to_grade[grade_idx] = grade_name
            
            obverse_dir = grade_folder / 'obverse'
            reverse_dir = grade_folder / 'reverse'
            
            if not obverse_dir.exists() or not reverse_dir.exists():
                continue
            
            obverse_images = sorted([f for f in obverse_dir.glob('*.jpg') if f.is_file()])
            
            for obverse_img in obverse_images:
                reverse_img = reverse_dir / obverse_img.name
                
                if reverse_img.exists():
                    # Parse grading service from filename
                    # Format: <grade>-<SERVICE>-<year>-<denom>-<cert>.jpg
                    parts = obverse_img.stem.split('-')
                    if len(parts) >= 2:
                        company = parts[1]  # PCGS, NGC, etc.
                        
                        # Skip unwanted companies
                        if company in ['OTHE', 'THAT']:
                            continue
                        
                        companies.add(company)
                        
                        temp_samples.append({
                            'obverse': obverse_img,
                            'reverse': reverse_img,
                            'grade': grade_name,
                            'grade_idx': grade_idx,
                            'company': company
                        })
        
        # Create company mapping
        for company_idx, company in enumerate(sorted(companies)):
            self.company_to_idx[company] = company_idx
            self.idx_to_company[company_idx] = company
        
        # Add company indices to samples
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
        
        # Stats
        from collections import Counter
        grade_counts = Counter([s['grade'] for s in self.samples])
        company_counts = Counter([s['company'] for s in self.samples])
        
        print(f"\n{split.upper()}: {len(self.samples)} samples")
        print(f"  Grades: {len(self.grade_to_idx)}")
        print(f"  Companies: {len(self.company_to_idx)} - {list(self.company_to_idx.keys())}")
        print(f"  Company distribution: {dict(company_counts)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        obverse = Image.open(sample['obverse']).convert('RGB')
        reverse = Image.open(sample['reverse']).convert('RGB')
        
        if self.transform:
            obverse = self.transform(obverse)
            reverse = self.transform(reverse)
        
        return obverse, reverse, sample['grade_idx'], sample['company_idx']


# ============================================================================
# DATA TRANSFORMS
# ============================================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(degrees=5, fill=(255, 255, 255)),
    transforms.ToTensor(),
        transforms.RandomAffine(
        degrees=0,              # rotation already handled
        translate=(0.03, 0.03), # small shifts in x/y
        scale=(0.95, 1.05),     # small zoom in/out
        fill=(255, 255, 255)
    ),

    # Photometric: tiny lighting/contrast wiggles
    transforms.ColorJitter(
        brightness=0.05,
        contrast=0.05,
        saturation=0.05,
        hue=0.01
    ),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================================================
# MULTI-TASK MODEL
# ============================================================================

class MultiTaskResNetClassifier(nn.Module):
    """
    Multi-task ResNet with two heads:
    1. Grade prediction (main task)
    2. Company prediction (auxiliary task)
    """
    
    def __init__(self, num_grades, num_companies, freeze_backbone=True):
        super(MultiTaskResNetClassifier, self).__init__()
        
        # Load pretrained ResNet-50
        weights = ResNet50_Weights.IMAGENET1K_V2
        obverse_resnet = resnet50(weights=weights)
        reverse_resnet = resnet50(weights=weights)
        
        # Shared encoders
        self.obverse_encoder = nn.Sequential(*list(obverse_resnet.children())[:-1])
        self.reverse_encoder = nn.Sequential(*list(reverse_resnet.children())[:-1])
        
        if freeze_backbone:
            for param in self.obverse_encoder.parameters():
                param.requires_grad = False
            for param in self.reverse_encoder.parameters():
                param.requires_grad = False
            print("✓ Backbone frozen")
        
        self.feature_dim = 2048
        
        # Shared fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Head 1: Grade classification (main task)
        self.grade_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_grades)
        )
        
        # Head 2: Company classification (auxiliary task)
        self.company_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_companies)
        )
    
    def unfreeze_backbone(self):
        for param in self.obverse_encoder.parameters():
            param.requires_grad = True
        for param in self.reverse_encoder.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen")
    
    def forward(self, obverse, reverse):
        # Encode both images
        obverse_feat = self.obverse_encoder(obverse).view(obverse.size(0), -1)
        reverse_feat = self.reverse_encoder(reverse).view(reverse.size(0), -1)
        
        # Fuse features
        combined = torch.cat([obverse_feat, reverse_feat], dim=1)
        shared_features = self.fusion(combined)
        
        # Two heads
        grade_output = self.grade_head(shared_features)
        company_output = self.company_head(shared_features)
        
        return grade_output, company_output


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, grade_criterion, company_criterion, optimizer, device, epoch):
    model.train()
    running_grade_loss = 0.0
    running_company_loss = 0.0
    running_total_loss = 0.0
    grade_correct = 0
    company_correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
    
    for obverse, reverse, grade_labels, company_labels in pbar:
        obverse = obverse.to(device)
        reverse = reverse.to(device)
        grade_labels = grade_labels.to(device)
        company_labels = company_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (two outputs)
        grade_outputs, company_outputs = model(obverse, reverse)
        
        # Compute losses
        grade_loss = grade_criterion(grade_outputs, grade_labels)
        company_loss = company_criterion(company_outputs, company_labels)
        
        # Combined loss
        total_loss = GRADE_LOSS_WEIGHT * grade_loss + COMPANY_LOSS_WEIGHT * company_loss
        
        # Backward
        total_loss.backward()
        optimizer.step()
        
        # Statistics
        batch_size = obverse.size(0)
        running_grade_loss += grade_loss.item() * batch_size
        running_company_loss += company_loss.item() * batch_size
        running_total_loss += total_loss.item() * batch_size
        
        _, grade_pred = torch.max(grade_outputs, 1)
        _, company_pred = torch.max(company_outputs, 1)
        total += batch_size
        grade_correct += (grade_pred == grade_labels).sum().item()
        company_correct += (company_pred == company_labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'grade_acc': f'{100*grade_correct/total:.1f}%',
            'comp_acc': f'{100*company_correct/total:.1f}%'
        })
    
    return (running_total_loss / total, running_grade_loss / total, running_company_loss / total,
            100 * grade_correct / total, 100 * company_correct / total)


def validate(model, loader, grade_criterion, company_criterion, device):
    model.eval()
    running_grade_loss = 0.0
    running_company_loss = 0.0
    running_total_loss = 0.0
    grade_correct = 0
    company_correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Val')
        for obverse, reverse, grade_labels, company_labels in pbar:
            obverse = obverse.to(device)
            reverse = reverse.to(device)
            grade_labels = grade_labels.to(device)
            company_labels = company_labels.to(device)
            
            grade_outputs, company_outputs = model(obverse, reverse)
            
            grade_loss = grade_criterion(grade_outputs, grade_labels)
            company_loss = company_criterion(company_outputs, company_labels)
            total_loss = GRADE_LOSS_WEIGHT * grade_loss + COMPANY_LOSS_WEIGHT * company_loss
            
            batch_size = obverse.size(0)
            running_grade_loss += grade_loss.item() * batch_size
            running_company_loss += company_loss.item() * batch_size
            running_total_loss += total_loss.item() * batch_size
            
            _, grade_pred = torch.max(grade_outputs, 1)
            _, company_pred = torch.max(company_outputs, 1)
            total += batch_size
            grade_correct += (grade_pred == grade_labels).sum().item()
            company_correct += (company_pred == company_labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'grade_acc': f'{100*grade_correct/total:.1f}%'
            })
    
    return (running_total_loss / total, running_grade_loss / total, running_company_loss / total,
            100 * grade_correct / total, 100 * company_correct / total)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\nCreating datasets...")
    train_dataset = MultiTaskCoinDataset(DATA_DIR, split='train', transform=train_transform)
    test_dataset = MultiTaskCoinDataset(DATA_DIR, split='test', transform=val_transform)
    val_dataset = test_dataset
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = test_loader
    
    print(f"\nDataloaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Test/Val: {len(test_loader)} batches")
    
    # Create model
    num_grades = len(train_dataset.grade_to_idx)
    num_companies = len(train_dataset.company_to_idx)
    
    model = MultiTaskResNetClassifier(
        num_grades=num_grades,
        num_companies=num_companies,
        freeze_backbone=FREEZE_BACKBONE
    )
    model = model.to(DEVICE)
    
    print(f"\nModel created:")
    print(f"  Grades: {num_grades}")
    print(f"  Companies: {num_companies}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss functions
    grade_criterion = nn.CrossEntropyLoss()
    company_criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    writer = SummaryWriter(LOG_DIR)
    
    # Training loop
    history = {
        'train_total_loss': [], 'train_grade_loss': [], 'train_company_loss': [],
        'train_grade_acc': [], 'train_company_acc': [],
        'val_total_loss': [], 'val_grade_loss': [], 'val_company_loss': [],
        'val_grade_acc': [], 'val_company_acc': []
    }
    
    best_grade_acc = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, 'coin_multitask_best.pth')
    
    print("\n" + "="*60)
    print("STARTING MULTI-TASK TRAINING")
    print("="*60)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Unfreeze backbone
        if FREEZE_BACKBONE and epoch == UNFREEZE_EPOCH:
            print(f"\n🔓 Unfreezing backbone")
            model.unfreeze_backbone()
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE/10, weight_decay=0.01)
        
        # Train
        train_total_loss, train_grade_loss, train_company_loss, train_grade_acc, train_company_acc = \
            train_epoch(model, train_loader, grade_criterion, company_criterion, optimizer, DEVICE, epoch)
        
        # Validate
        val_total_loss, val_grade_loss, val_company_loss, val_grade_acc, val_company_acc = \
            validate(model, val_loader, grade_criterion, company_criterion, DEVICE)
        
        scheduler.step(val_grade_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_total_loss'].append(train_total_loss)
        history['train_grade_loss'].append(train_grade_loss)
        history['train_company_loss'].append(train_company_loss)
        history['train_grade_acc'].append(train_grade_acc)
        history['train_company_acc'].append(train_company_acc)
        history['val_total_loss'].append(val_total_loss)
        history['val_grade_loss'].append(val_grade_loss)
        history['val_company_loss'].append(val_company_loss)
        history['val_grade_acc'].append(val_grade_acc)
        history['val_company_acc'].append(val_company_acc)
        
        # TensorBoard
        writer.add_scalar('Loss/train_total', train_total_loss, epoch)
        writer.add_scalar('Loss/train_grade', train_grade_loss, epoch)
        writer.add_scalar('Loss/train_company', train_company_loss, epoch)
        writer.add_scalar('Accuracy/train_grade', train_grade_acc, epoch)
        writer.add_scalar('Accuracy/train_company', train_company_acc, epoch)
        writer.add_scalar('Accuracy/val_grade', val_grade_acc, epoch)
        writer.add_scalar('Accuracy/val_company', val_company_acc, epoch)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train: Grade={train_grade_acc:.1f}%, Company={train_company_acc:.1f}%")
        print(f"  Val:   Grade={val_grade_acc:.1f}%, Company={val_company_acc:.1f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model based on grade accuracy
        if val_grade_acc > best_grade_acc:
            best_grade_acc = val_grade_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_grade_acc': val_grade_acc,
                'val_company_acc': val_company_acc,
                'grade_to_idx': train_dataset.grade_to_idx,
                'idx_to_grade': train_dataset.idx_to_grade,
                'company_to_idx': train_dataset.company_to_idx,
                'idx_to_company': train_dataset.idx_to_company
            }, best_model_path)
            print(f"  ✓ New best! Grade Acc: {val_grade_acc:.2f}%")
    
    writer.close()
    
    # Save results
    with open(os.path.join(OUTPUT_DIR, 'history_multitask.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best grade accuracy: {best_grade_acc:.2f}%")
    print(f"Model saved: {best_model_path}")

