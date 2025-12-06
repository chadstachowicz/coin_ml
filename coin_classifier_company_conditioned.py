"""
Company-Conditioned ResNet for Coin Grading

Approach B: Feed grading company as an input feature

At inference time, you can specify which company's style to mimic:
- "What would PCGS call this?" → feed in PCGS encoding
- "What would NGC call this?" → feed in NGC encoding

The model learns: "Given coin features + company style → grade"

Features:
- Dual-image input (obverse + reverse)
- ResNet-50 backbone
- Company embedding concatenated to features
- Can predict grades conditioned on different companies at inference
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


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_DIR = 'davidlawrence_dataset/Circulation'
OUTPUT_DIR = 'models'
LOG_DIR = 'runs/company_conditioned_' + datetime.now().strftime('%Y%m%d_%H%M%S')

# Model hyperparameters
IMAGE_SIZE = 448
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
FREEZE_BACKBONE = True
UNFREEZE_EPOCH = 25

# Company embedding dimension
COMPANY_EMBEDDING_DIM = 32  # Small embedding for company identity

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
print("COMPANY-CONDITIONED DUAL RESNET CLASSIFIER")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Company embedding dim: {COMPANY_EMBEDDING_DIM}")
print("="*60)


# ============================================================================
# DATASET CLASS
# ============================================================================

class CompanyConditionedDataset(Dataset):
    """Dataset that loads dual images and company labels."""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.grade_to_idx = {}
        self.idx_to_grade = {}
        self.company_to_idx = {}
        self.idx_to_company = {}
        
        # First pass: collect companies
        temp_samples = []
        companies = set()
        
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
        
        # Stats
        from collections import Counter
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
# COMPANY-CONDITIONED MODEL
# ============================================================================

class CompanyConditionedResNet(nn.Module):
    """
    ResNet conditioned on grading company.
    
    At inference, you can change the company input to ask:
    "What would PCGS call this?" vs "What would NGC call this?"
    """
    
    def __init__(self, num_grades, num_companies, company_embedding_dim=32, freeze_backbone=True):
        super(CompanyConditionedResNet, self).__init__()
        
        # Company embedding layer
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
        
        # Fusion layer: concatenate [obverse_feat, reverse_feat, company_embedding]
        fusion_input_dim = self.feature_dim * 2 + company_embedding_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Grade prediction head
        self.grade_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_grades)
        )
    
    def unfreeze_backbone(self):
        for param in self.obverse_encoder.parameters():
            param.requires_grad = True
        for param in self.reverse_encoder.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen")
    
    def forward(self, obverse, reverse, company_idx):
        # Encode images
        obverse_feat = self.obverse_encoder(obverse).view(obverse.size(0), -1)
        reverse_feat = self.reverse_encoder(reverse).view(reverse.size(0), -1)
        
        # Get company embedding
        company_emb = self.company_embedding(company_idx)
        
        # Concatenate: [obverse, reverse, company]
        combined = torch.cat([obverse_feat, reverse_feat, company_emb], dim=1)
        
        # Fusion + prediction
        fused_features = self.fusion(combined)
        grade_output = self.grade_head(fused_features)
        
        return grade_output
    
    def predict_with_company(self, obverse, reverse, company_name, company_to_idx):
        """
        Inference helper: predict grade for a specific company.
        
        Example:
            pred_pcgs = model.predict_with_company(obv, rev, 'PCGS', company_to_idx)
            pred_ngc = model.predict_with_company(obv, rev, 'NGC', company_to_idx)
        """
        company_idx = torch.tensor([company_to_idx[company_name]], device=obverse.device)
        company_idx = company_idx.expand(obverse.size(0))
        return self.forward(obverse, reverse, company_idx)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
    
    for obverse, reverse, grade_labels, company_labels in pbar:
        obverse = obverse.to(device)
        reverse = reverse.to(device)
        grade_labels = grade_labels.to(device)
        company_labels = company_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (company-conditioned)
        outputs = model(obverse, reverse, company_labels)
        loss = criterion(outputs, grade_labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistics
        batch_size = obverse.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs, 1)
        total += batch_size
        correct += (predicted == grade_labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.1f}%'
        })
    
    return running_loss / total, 100 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Val')
        for obverse, reverse, grade_labels, company_labels in pbar:
            obverse = obverse.to(device)
            reverse = reverse.to(device)
            grade_labels = grade_labels.to(device)
            company_labels = company_labels.to(device)
            
            outputs = model(obverse, reverse, company_labels)
            loss = criterion(outputs, grade_labels)
            
            batch_size = obverse.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)
            total += batch_size
            correct += (predicted == grade_labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.1f}%'
            })
    
    return running_loss / total, 100 * correct / total


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\nCreating datasets...")
    train_dataset = CompanyConditionedDataset(DATA_DIR, split='train', transform=train_transform)
    test_dataset = CompanyConditionedDataset(DATA_DIR, split='test', transform=val_transform)
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
    
    model = CompanyConditionedResNet(
        num_grades=num_grades,
        num_companies=num_companies,
        company_embedding_dim=COMPANY_EMBEDDING_DIM,
        freeze_backbone=FREEZE_BACKBONE
    )
    model = model.to(DEVICE)
    
    print(f"\nModel created:")
    print(f"  Grades: {num_grades}")
    print(f"  Companies: {num_companies}")
    print(f"  Company embedding dim: {COMPANY_EMBEDDING_DIM}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
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
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_acc = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, 'coin_company_conditioned_best.pth')
    
    print("\n" + "="*60)
    print("STARTING COMPANY-CONDITIONED TRAINING")
    print("="*60)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Unfreeze backbone
        if FREEZE_BACKBONE and epoch == UNFREEZE_EPOCH:
            print(f"\n🔓 Unfreezing backbone")
            model.unfreeze_backbone()
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE/10, weight_decay=0.01)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
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
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train: Acc={train_acc:.1f}%, Loss={train_loss:.4f}")
        print(f"  Val:   Acc={val_acc:.1f}%, Loss={val_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'grade_to_idx': train_dataset.grade_to_idx,
                'idx_to_grade': train_dataset.idx_to_grade,
                'company_to_idx': train_dataset.company_to_idx,
                'idx_to_company': train_dataset.idx_to_company,
                'company_embedding_dim': COMPANY_EMBEDDING_DIM
            }, best_model_path)
            print(f"  ✓ New best! Acc: {val_acc:.2f}%")
    
    writer.close()
    
    # Save results
    with open(os.path.join(OUTPUT_DIR, 'history_company_conditioned.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Model saved: {best_model_path}")
    
    # Demonstrate company-conditioned inference
    print(f"\n{'='*60}")
    print("COMPANY-CONDITIONED INFERENCE EXAMPLE")
    print("="*60)
    print("\nAt inference time, you can now ask:")
    print("  model.predict_with_company(obverse, reverse, 'PCGS', company_to_idx)")
    print("  model.predict_with_company(obverse, reverse, 'NGC', company_to_idx)")
    print("\nThis lets you see how different companies might grade the same coin!")

