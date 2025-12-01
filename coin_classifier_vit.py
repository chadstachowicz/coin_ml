"""
Coin Grade Classifier - Vision Transformer (ViT)

Trains a Vision Transformer model to classify coin grades using BOTH obverse 
and reverse images at full 1000x1000 resolution.

Key Features:
- Dual-Image Input: Uses both sides of each coin
- Full Resolution: 1000x1000 pixels (preserves fine details)
- Vision Transformer: State-of-the-art architecture
- Feature Fusion: Combines information from both sides

Architecture:
- Two ViT-B/16 encoders (one for obverse, one for reverse)
- Feature fusion layer
- Classification head
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

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
DATA_DIR = 'images'  # Directory containing <grade>/obverse/ and <grade>/reverse/
OUTPUT_DIR = 'models'
LOG_DIR = 'runs/vit_dual_' + datetime.now().strftime('%Y%m%d_%H%M%S')

# Model hyperparameters
IMAGE_SIZE = 1000  # Keep full resolution
BATCH_SIZE = 2     # Small batch due to large images
                   # Increase if you have more memory:
                   # - M1/M2 8GB: use 2
                   # - M1/M2 16GB+: try 4
                   # - NVIDIA 8GB: try 2-4
                   # - NVIDIA 16GB+: try 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

# Training settings
# Device selection: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Using CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("Using MPS (Apple Silicon GPU)")
else:
    DEVICE = torch.device('cpu')
    print("Using CPU")

NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues on macOS
PIN_MEMORY = True if torch.cuda.is_available() else False

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("="*60)
print("COIN CLASSIFIER - VISION TRANSFORMER (DUAL IMAGE)")
print("="*60)
print(f"\nHardware Check:")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  MPS available: {torch.backends.mps.is_available()}")
print(f"  Selected device: {DEVICE}")
print(f"\nConfiguration:")
print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE} (FULL RESOLUTION)")
print(f"  Patch size: 50x50 (20x20 patches)")
print(f"  Training from scratch: Yes (no pretrained weights)")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {NUM_EPOCHS}")
print("="*60)


# ============================================================================
# DATASET CLASS - DUAL IMAGE LOADER
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
            
            # Get all reverse images
            reverse_images = sorted([f for f in reverse_dir.glob('*.jpg') if f.is_file()])
            
            # Create a mapping of cert numbers to reverse images
            reverse_map = {}
            for rev_img in reverse_images:
                # Extract cert number (e.g., 45407833 from vf25bn-1c-45407833-reverse-1.jpg)
                parts = rev_img.stem.split('-')
                if len(parts) >= 3:
                    cert_num = parts[2]  # The cert number is the 3rd part
                    reverse_map[cert_num] = rev_img
            
            for obverse_img in obverse_images:
                # Extract cert number from obverse filename
                parts = obverse_img.stem.split('-')
                if len(parts) >= 3:
                    cert_num = parts[2]
                    
                    # Find matching reverse by cert number
                    if cert_num in reverse_map:
                        self.samples.append({
                            'obverse': obverse_img,
                            'reverse': reverse_map[cert_num],
                            'label': idx,
                            'grade': grade_name
                        })
                    else:
                        print(f"Warning: No matching reverse for cert {cert_num} (obverse: {obverse_img.name})")
        
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
        
        print(f"\n{split.upper()} Dataset:")
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

print("\nConfiguring transforms...")

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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # ============================================================================
    # CREATE DATASETS AND DATALOADERS
    # ============================================================================
    
    print("\nCreating datasets...")
    
    train_dataset = DualCoinDataset(DATA_DIR, split='train', transform=train_transform)
    test_dataset = DualCoinDataset(DATA_DIR, split='test', transform=val_transform)
    val_dataset = DualCoinDataset(DATA_DIR, split='val', transform=val_transform)
    
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
# DUAL VISION TRANSFORMER MODEL
# ============================================================================

class DualViTClassifier(nn.Module):
    """Vision Transformer for dual-image (obverse + reverse) classification."""
    
    def __init__(self, num_classes, image_size=1000, pretrained=False):
        super(DualViTClassifier, self).__init__()
        
        # Import ViT components
        from torchvision.models.vision_transformer import VisionTransformer
        
        # Calculate patch parameters for 1000x1000 images
        patch_size = 50  # 1000/50 = 20x20 patches (reasonable for ViT)
        num_patches = (image_size // patch_size) ** 2  # 400 patches
        
        # Create ViT models from scratch (no pretrained weights)
        # ViT-B configuration: 12 layers, 768 hidden dim, 12 attention heads
        self.obverse_vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=768  # Output features, not classes
        )
        
        self.reverse_vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=768  # Output features, not classes
        )
        
        # Feature dimension (768 for ViT-B/16)
        self.feature_dim = 768
        
        # Fusion layer - combines features from both sides
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, obverse, reverse):
        # Process obverse image through ViT (returns features from CLS token)
        obverse_feat = self.obverse_vit(obverse)  # [B, 768]
        
        # Process reverse image through ViT
        reverse_feat = self.reverse_vit(reverse)  # [B, 768]
        
        # Concatenate features from both sides
        combined = torch.cat([obverse_feat, reverse_feat], dim=1)  # [B, 1536]
        
        # Fusion
        fused = self.fusion(combined)  # [B, 1024]
        
        # Classification
        output = self.classifier(fused)  # [B, num_classes]
        
        return output


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
# MAIN EXECUTION CONTINUES
# ============================================================================

if __name__ == '__main__':
    # Model creation and training
    
    print("\nCreating model...")
    num_classes = len(train_dataset.class_to_idx)
    model = DualViTClassifier(num_classes=num_classes, image_size=IMAGE_SIZE, pretrained=False)
    model = model.to(DEVICE)
    
    print(f"\nModel created:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {list(train_dataset.class_to_idx.keys())}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    writer = SummaryWriter(LOG_DIR)
    
    print("\nTraining setup complete")
    
    
    # Main training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, 'coin_vit_dual_best.pth')
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, split='Val')
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
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
            print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
    
    writer.close()
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    
    
    # Test evaluation
    print("\n" + "="*60)
    print("TEST EVALUATION")
    print("="*60)
    
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE, split='Test')
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    
    # Save results
    with open(os.path.join(OUTPUT_DIR, 'history_vit.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    config = {
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'num_classes': num_classes,
        'classes': list(train_dataset.class_to_idx.keys()),
        'best_val_acc': best_val_acc,
        'test_acc': test_acc
    }
    
    with open(os.path.join(OUTPUT_DIR, 'config_vit.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Training complete! Test accuracy: {test_acc:.2f}%")
    print(f"✓ Results saved to {OUTPUT_DIR}/")
