"""
Dual-Image Custom CNN for Coin Grade Classification

Trains a custom CNN from scratch using BOTH obverse and reverse images
at full 1000x1000 resolution.

Features:
- Uses both sides of the coin
- Full 1000x1000 resolution (no downsampling)
- Custom VGG-style CNN architecture
- Feature fusion from both sides
- Training from scratch
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
DATA_DIR = 'davidlawrence_dataset/Circulation'
OUTPUT_DIR = 'models'
LOG_DIR = 'runs/dual_cnn_' + datetime.now().strftime('%Y%m%d_%H%M%S')

# Model hyperparameters
IMAGE_SIZE = 512  # Full resolution
BATCH_SIZE = 32     # Small batch due to large images
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

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

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("="*60)
print("DUAL-IMAGE CUSTOM CNN CLASSIFIER")
print("="*60)
print(f"Hardware: {DEVICE}")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE} (FULL RESOLUTION)")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print("="*60)


# ============================================================================
# DATASET CLASS - DUAL IMAGE LOADER
# ============================================================================

class DualCoinDataset(Dataset):
    """Dataset that loads both obverse and reverse images for each coin."""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        grade_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for idx, grade_folder in enumerate(grade_folders):
            grade_name = grade_folder.name
            self.class_to_idx[grade_name] = idx
            self.idx_to_class[idx] = grade_name
            
            obverse_dir = grade_folder / 'obverse'
            reverse_dir = grade_folder / 'reverse'
            
            if not obverse_dir.exists() or not reverse_dir.exists():
                continue
            
            obverse_images = sorted([f for f in obverse_dir.glob('*.jpg') if f.is_file()])
            reverse_images = sorted([f for f in reverse_dir.glob('*.jpg') if f.is_file()])
            
            # Create reverse map by cert number
            reverse_map = {}
            for rev_img in reverse_images:
                parts = rev_img.stem.split('-')
                if len(parts) >= 3:
                    cert_num = parts[2]
                    reverse_map[cert_num] = rev_img
            
            for obverse_img in obverse_images:
                parts = obverse_img.stem.split('-')
                if len(parts) >= 3:
                    cert_num = parts[2]
                    if cert_num in reverse_map:
                        self.samples.append({
                            'obverse': obverse_img,
                            'reverse': reverse_map[cert_num],
                            'label': idx,
                            'grade': grade_name
                        })
        
        # Split data (70% train, 20% test, 10% val)
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        
        n_train = int(0.7 * len(self.samples))
        n_test = int(0.2 * len(self.samples))
        
        if split == 'train':
            indices = indices[:n_train]
        elif split == 'test':
            indices = indices[n_train:n_train + n_test]
        else:
            indices = indices[n_train + n_test:]
        
        self.samples = [self.samples[i] for i in indices]
        
        print(f"{split.upper()}: {len(self.samples)} samples, {len(self.class_to_idx)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        obverse = Image.open(sample['obverse']).convert('RGB')
        reverse = Image.open(sample['reverse']).convert('RGB')
        
        if self.transform:
            obverse = self.transform(obverse)
            reverse = self.transform(reverse)
        
        return obverse, reverse, sample['label']


# ============================================================================
# DATA TRANSFORMS
# ============================================================================

print("\nConfiguring transforms...")

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================================================
# DUAL CNN MODEL
# ============================================================================

class DualCNNClassifier(nn.Module):
    """Custom CNN for dual-image (obverse + reverse) classification."""
    
    def __init__(self, num_classes):
        super(DualCNNClassifier, self).__init__()
        
        # Single CNN architecture (will be used for both sides)
        def make_cnn():
            return nn.Sequential(
                # Block 1: 3 → 64
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # 1000 → 500
                
                # Block 2: 64 → 128
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # 500 → 250
                
                # Block 3: 128 → 256
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # 250 → 125
                
                # Block 4: 256 → 512
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # 125 → 62
                
                # Block 5: 512 → 512
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # 62 → 31
                
                nn.AdaptiveAvgPool2d((1, 1))  # → 512
            )
        
        # Separate CNNs for obverse and reverse
        self.obverse_cnn = make_cnn()
        self.reverse_cnn = make_cnn()
        
        # Feature fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(512 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, obverse, reverse):
        # Extract features from both sides
        obverse_feat = self.obverse_cnn(obverse)
        obverse_feat = obverse_feat.view(obverse_feat.size(0), -1)  # Flatten
        
        reverse_feat = self.reverse_cnn(reverse)
        reverse_feat = reverse_feat.view(reverse_feat.size(0), -1)  # Flatten
        
        # Concatenate features
        combined = torch.cat([obverse_feat, reverse_feat], dim=1)
        
        # Fusion and classification
        fused = self.fusion(combined)
        output = self.classifier(fused)
        
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
        
        optimizer.zero_grad()
        outputs = model(obverse, reverse)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * obverse.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    return running_loss / total, 100 * correct / total


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
    
    return running_loss / total, 100 * correct / total


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = DualCoinDataset(DATA_DIR, split='train', transform=train_transform)
    test_dataset = DualCoinDataset(DATA_DIR, split='test', transform=val_transform)
    val_dataset = DualCoinDataset(DATA_DIR, split='val', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    print(f"Train: {len(train_loader)} batches")
    print(f"Test: {len(test_loader)} batches")
    print(f"Val: {len(val_loader)} batches")
    
    
    # Create model
    print("\nCreating model...")
    num_classes = len(train_dataset.class_to_idx)
    model = DualCNNClassifier(num_classes=num_classes).to(DEVICE)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {list(train_dataset.class_to_idx.keys())}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    writer = SummaryWriter(LOG_DIR)
    
    print("\nTraining setup complete")
    
    
    # Main training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, 'coin_dual_cnn_best.pth')
    
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
    with open(os.path.join(OUTPUT_DIR, 'history_dual_cnn.json'), 'w') as f:
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
    
    with open(os.path.join(OUTPUT_DIR, 'config_dual_cnn.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train', marker='o')
    ax1.plot(history['val_loss'], label='Val', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train', marker='o')
    ax2.plot(history['val_acc'], label='Val', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history_dual_cnn.png'), dpi=300)
    print(f"\n✓ Training complete! Test accuracy: {test_acc:.2f}%")
    print(f"✓ Results saved to {OUTPUT_DIR}/")
    print(f"✓ Plot saved to {OUTPUT_DIR}/training_history_dual_cnn.png")

