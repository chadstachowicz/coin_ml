"""
Dual-Image ResNet Fine-Tuning for Coin Grade Classification

Fine-tunes a pretrained ResNet model using BOTH obverse and reverse images
at full 1000x1000 resolution.

Features:
- Uses both sides of the coin
- Full 1000x1000 resolution (no downsampling)
- ResNet-50 pretrained on ImageNet
- Feature fusion from both sides
- Fine-tuning with frozen early layers option
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
DATA_DIR = 'davidlawrence_dataset/Proof'
OUTPUT_DIR = 'models'
LOG_DIR = 'runs/resnet_dual_' + datetime.now().strftime('%Y%m%d_%H%M%S')

# Model hyperparameters
IMAGE_SIZE = 1000  # Full resolution
BATCH_SIZE = 2     # Small batch due to large images
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
FREEZE_BACKBONE = True  # Freeze early ResNet layers initially
UNFREEZE_EPOCH = 15     # Unfreeze all layers after this epoch

# Device selection
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

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print(f"\nDevice: {DEVICE}")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Data directory: {DATA_DIR}")


# ============================================================================
# DATASET CLASS
# ============================================================================

class DualCoinDataset(Dataset):
    """Dataset that loads both obverse and reverse images for each coin."""
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: Root directory with structure: <grade>/obverse/ and <grade>/reverse/
            split: 'train', 'test', or 'val'
            transform: Optional transform to apply to both images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Scan for grade folders
        grade_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for idx, grade_folder in enumerate(grade_folders):
            grade_name = grade_folder.name
            self.class_to_idx[grade_name] = idx
            self.idx_to_class[idx] = grade_name
            
            obverse_dir = grade_folder / 'obverse'
            reverse_dir = grade_folder / 'reverse'
            
            if not obverse_dir.exists() or not reverse_dir.exists():
                print(f"Warning: Missing obverse or reverse folder for {grade_name}")
                continue
            
            # Get all obverse images
            obverse_images = sorted([f for f in obverse_dir.glob('*.jpg') if f.is_file()])
            
            for obverse_img in obverse_images:
                # Find matching reverse image (same filename)
                reverse_img = reverse_dir / obverse_img.name
                
                if reverse_img.exists():
                    self.samples.append({
                        'obverse': obverse_img,
                        'reverse': reverse_img,
                        'label': idx,
                        'grade': grade_name
                    })
                else:
                    print(f"Warning: No matching reverse for {obverse_img.name}")
        
        # Split data (70% train, 20% test, 10% val)
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        
        n_train = int(0.7 * len(self.samples))
        n_test = int(0.2 * len(self.samples))
        
        if split == 'train':
            indices = indices[:n_train]
        elif split == 'test':
            indices = indices[n_train:n_train + n_test]
        else:  # val
            indices = indices[n_train + n_test:]
        
        self.samples = [self.samples[i] for i in indices]
        
        print(f"{split.upper()} Dataset:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Classes: {len(self.class_to_idx)}")
        print(f"  Class names: {list(self.class_to_idx.keys())}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load both images
        obverse = Image.open(sample['obverse']).convert('RGB')
        reverse = Image.open(sample['reverse']).convert('RGB')
        
        # Apply same transforms to both
        if self.transform:
            obverse = self.transform(obverse)
            reverse = self.transform(reverse)
        
        return obverse, reverse, sample['label']


# ============================================================================
# DATA TRANSFORMS
# ============================================================================

# Training transforms - minimal augmentation to preserve detail
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/test transforms - no augmentation
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("\n✓ Transforms configured")


# ============================================================================
# CREATE DATASETS AND DATALOADERS
# ============================================================================

# Create datasets
train_dataset = DualCoinDataset(DATA_DIR, split='train', transform=train_transform)
test_dataset = DualCoinDataset(DATA_DIR, split='test', transform=val_transform)
val_dataset = DualCoinDataset(DATA_DIR, split='val', transform=val_transform)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

print(f"\nDataloaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")
print(f"  Val batches: {len(val_loader)}")


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class DualResNetClassifier(nn.Module):
    """ResNet-50 for dual-image (obverse + reverse) classification."""
    
    def __init__(self, num_classes, freeze_backbone=True):
        super(DualResNetClassifier, self).__init__()
        
        # Load pretrained ResNet-50 models for both sides
        weights = ResNet50_Weights.IMAGENET1K_V2
        obverse_resnet = resnet50(weights=weights)
        reverse_resnet = resnet50(weights=weights)
        
        # Remove the final classification layer (FC)
        # ResNet output is 2048-dimensional before FC layer
        self.obverse_encoder = nn.Sequential(*list(obverse_resnet.children())[:-1])
        self.reverse_encoder = nn.Sequential(*list(reverse_resnet.children())[:-1])
        
        # Freeze backbone layers if requested
        if freeze_backbone:
            for param in self.obverse_encoder.parameters():
                param.requires_grad = False
            for param in self.reverse_encoder.parameters():
                param.requires_grad = False
            print("✓ Backbone layers frozen")
        
        # Feature dimension from ResNet-50
        self.feature_dim = 2048
        
        # Fusion layer - combines features from both sides
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
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone layers for fine-tuning."""
        for param in self.obverse_encoder.parameters():
            param.requires_grad = True
        for param in self.reverse_encoder.parameters():
            param.requires_grad = True
        print("✓ Backbone layers unfrozen")
    
    def forward(self, obverse, reverse):
        # Process obverse image
        obverse_feat = self.obverse_encoder(obverse)  # [B, 2048, 1, 1]
        obverse_feat = obverse_feat.view(obverse_feat.size(0), -1)  # [B, 2048]
        
        # Process reverse image
        reverse_feat = self.reverse_encoder(reverse)  # [B, 2048, 1, 1]
        reverse_feat = reverse_feat.view(reverse_feat.size(0), -1)  # [B, 2048]
        
        # Concatenate features from both sides
        combined = torch.cat([obverse_feat, reverse_feat], dim=1)  # [B, 4096]
        
        # Fusion
        fused = self.fusion(combined)  # [B, 2048]
        
        # Classification
        output = self.classifier(fused)  # [B, num_classes]
        
        return output


# Create model
num_classes = len(train_dataset.class_to_idx)
model = DualResNetClassifier(num_classes=num_classes, freeze_backbone=FREEZE_BACKBONE)
model = model.to(DEVICE)

print(f"\nModel created:")
print(f"  Number of classes: {num_classes}")
print(f"  Classes: {list(train_dataset.class_to_idx.keys())}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Frozen parameters: {total_params - trainable_params:,}")


# ============================================================================
# TRAINING SETUP
# ============================================================================

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer - only optimize trainable parameters
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5,
    verbose=True
)

# TensorBoard writer
writer = SummaryWriter(LOG_DIR)

print("\nTraining setup complete")
print(f"  Optimizer: AdamW")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Scheduler: ReduceLROnPlateau")
print(f"  TensorBoard logs: {LOG_DIR}")


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
    
    for obverse, reverse, labels in pbar:
        obverse = obverse.to(device)
        reverse = reverse.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(obverse, reverse)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * obverse.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device, split='Val'):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f'{split}')
        for obverse, reverse, labels in pbar:
            obverse = obverse.to(device)
            reverse = reverse.to(device)
            labels = labels.to(device)
            
            outputs = model(obverse, reverse)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * obverse.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

# Training history
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

best_val_acc = 0.0
best_model_path = os.path.join(OUTPUT_DIR, 'coin_resnet_dual_best.pth')

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 60)
    
    # Unfreeze backbone after specified epoch
    if FREEZE_BACKBONE and epoch == UNFREEZE_EPOCH:
        print(f"\n🔓 Unfreezing backbone at epoch {epoch+1}")
        model.unfreeze_backbone()
        
        # Re-create optimizer to include newly unfrozen parameters
        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE / 10,  # Lower LR for fine-tuning
            weight_decay=0.01
        )
        print(f"  Updated learning rate: {LEARNING_RATE / 10}")
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, split='Val')
    
    # Update learning rate
    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Log to TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('Learning_rate', current_lr, epoch)
    
    # Print summary
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    print(f"  LR: {current_lr:.6f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'class_to_idx': train_dataset.class_to_idx,
            'idx_to_class': train_dataset.idx_to_class
        }, best_model_path)
        print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Best model saved to: {best_model_path}")

writer.close()


# ============================================================================
# TEST EVALUATION
# ============================================================================

# Load best model
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_acc = validate(model, test_loader, criterion, DEVICE, split='Test')

print(f"\n{'='*60}")
print("FINAL TEST RESULTS")
print("="*60)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print("="*60)


# ============================================================================
# SAVE RESULTS
# ============================================================================

# Save training history
with open(os.path.join(OUTPUT_DIR, 'history_resnet.json'), 'w') as f:
    json.dump(history, f, indent=2)

# Save configuration
config = {
    'architecture': 'DualResNetClassifier (ResNet-50)',
    'image_size': IMAGE_SIZE,
    'batch_size': BATCH_SIZE,
    'num_epochs': NUM_EPOCHS,
    'learning_rate': LEARNING_RATE,
    'freeze_backbone': FREEZE_BACKBONE,
    'unfreeze_epoch': UNFREEZE_EPOCH,
    'num_classes': num_classes,
    'classes': list(train_dataset.class_to_idx.keys()),
    'best_val_acc': best_val_acc,
    'test_acc': test_acc,
    'total_params': total_params,
    'trainable_params_initial': trainable_params
}

with open(os.path.join(OUTPUT_DIR, 'config_resnet.json'), 'w') as f:
    json.dump(config, f, indent=2)

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
ax1.plot(history['train_loss'], label='Train Loss', marker='o')
ax1.plot(history['val_loss'], label='Val Loss', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

# Accuracy plot
ax2.plot(history['train_acc'], label='Train Acc', marker='o')
ax2.plot(history['val_acc'], label='Val Acc', marker='s')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

# Mark unfreeze point if applicable
if FREEZE_BACKBONE and UNFREEZE_EPOCH < NUM_EPOCHS:
    ax1.axvline(x=UNFREEZE_EPOCH, color='red', linestyle='--', alpha=0.5, label='Unfreeze')
    ax2.axvline(x=UNFREEZE_EPOCH, color='red', linestyle='--', alpha=0.5, label='Unfreeze')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_history_resnet.png'), dpi=300, bbox_inches='tight')
print(f"\nTraining plots saved to: {OUTPUT_DIR}/training_history_resnet.png")

print("\nResults saved:")
print(f"  Model: {best_model_path}")
print(f"  History: {OUTPUT_DIR}/history_resnet.json")
print(f"  Config: {OUTPUT_DIR}/config_resnet.json")
print(f"  Plots: {OUTPUT_DIR}/training_history_resnet.png")
print(f"\n✓ All done! Best validation accuracy: {best_val_acc:.2f}%")
print(f"✓ Test accuracy: {test_acc:.2f}%")

