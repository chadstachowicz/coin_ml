"""
Single-Image Custom CNN for Coin Grade Classification

Trains a custom CNN from scratch using ONLY obverse images
at full resolution. Simpler than dual-image version.

Features:
- Uses only obverse side of the coin
- Full resolution
- Custom VGG-style CNN architecture
- Training from scratch
- Class balancing support
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
DATA_DIR = 'davidlawrence_elements/Circulation'
IMAGE_SUFFIX = '_gradient_small.jpg'  # Use gradient images from element analysis
OUTPUT_DIR = 'models'
LOG_DIR = 'runs/single_cnn_gradient_' + datetime.now().strftime('%Y%m%d_%H%M%S')

# Model hyperparameters
IMAGE_SIZE = 1000  # Full resolution
BATCH_SIZE = 16     # Can use larger batch since only one image per sample
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

# Class balancing options (IMPORTANT for imbalanced datasets!)
MAX_SAMPLES_PER_CLASS = 200     # Limit max samples per class (e.g., 50, 100, None=no limit)
USE_CLASS_WEIGHTS = False       # Weight loss by inverse class frequency
USE_BALANCED_SAMPLING = False   # Use WeightedRandomSampler (slower but more balanced)

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
print("SINGLE-IMAGE CUSTOM CNN CLASSIFIER")
print("="*60)
print(f"Hardware: {DEVICE}")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Data: {DATA_DIR}")
print(f"Image type: {IMAGE_SUFFIX}")
print("="*60)


# ============================================================================
# DATASET CLASS - SINGLE IMAGE LOADER
# ============================================================================

class SingleCoinDataset(Dataset):
    """Dataset that loads only obverse images for each coin."""
    
    def __init__(self, data_dir, split='train', transform=None, max_samples_per_class=None, 
                 image_suffix='_gradient_small.jpg'):
        """
        Args:
            data_dir: Root directory with structure: <grade>/obverse/
            split: 'train', 'test', or 'val'
            transform: Optional transform to apply to images
            max_samples_per_class: Maximum samples per class (None = no limit)
            image_suffix: Suffix to filter images (e.g., '_gradient_small.jpg')
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_suffix = image_suffix
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Scan for grade folders
        grade_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        # First pass: collect samples by class
        samples_by_class = {}
        
        for idx, grade_folder in enumerate(grade_folders):
            grade_name = grade_folder.name
            self.class_to_idx[grade_name] = idx
            self.idx_to_class[idx] = grade_name
            
            obverse_dir = grade_folder / 'obverse'
            
            if not obverse_dir.exists():
                continue
            
            # Get all images with the specified suffix
            obverse_images = sorted([f for f in obverse_dir.glob('*' + image_suffix) if f.is_file()])
            
            class_samples = []
            for obverse_img in obverse_images:
                class_samples.append({
                    'image': obverse_img,
                    'label': idx,
                    'grade': grade_name
                })
            
            samples_by_class[idx] = class_samples
        
        # Apply class balancing if requested
        if max_samples_per_class is not None and split == 'train':
            print(f"\n📊 Class balancing (max {max_samples_per_class} per class):")
            for idx, class_samples in samples_by_class.items():
                before = len(class_samples)
                if before > max_samples_per_class:
                    np.random.seed(42 + idx)
                    indices = np.random.choice(before, max_samples_per_class, replace=False)
                    samples_by_class[idx] = [class_samples[i] for i in sorted(indices)]
                    print(f"  {self.idx_to_class[idx]:8s}: {before:4d} → {max_samples_per_class:4d}")
        
        # Flatten samples
        for class_samples in samples_by_class.values():
            self.samples.extend(class_samples)
        
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
        
        # Compute class distribution
        from collections import Counter
        label_counts = Counter([s['label'] for s in self.samples])
        self.class_counts = label_counts
        
        print(f"\n{split.upper()}: {len(self.samples)} samples, {len(self.class_to_idx)} classes")
        for grade_name in sorted(self.class_to_idx.keys()):
            idx = self.class_to_idx[grade_name]
            count = label_counts.get(idx, 0)
            if count > 0:
                print(f"  {grade_name:8s}: {count:4d} ({100*count/len(self.samples):5.1f}%)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label']


# ============================================================================
# DATA TRANSFORMS
# ============================================================================

print("\nConfiguring transforms...")

# Training transforms
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/test transforms
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================================================
# MODEL ARCHITECTURE - SINGLE IMAGE CNN
# ============================================================================

class SingleCNNClassifier(nn.Module):
    """Custom CNN for single-image classification."""
    
    def __init__(self, num_classes):
        super(SingleCNNClassifier, self).__init__()
        
        # VGG-style CNN encoder
        self.encoder = nn.Sequential(
            # Block 1: 512x512 -> 256x256
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 256x256 -> 128x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 128x128 -> 64x64
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: 64x64 -> 32x32
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5: 32x32 -> 16x16
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Process image
        features = self.encoder(x)  # [B, 512, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 512]
        
        # Classification
        output = self.classifier(features)  # [B, num_classes]
        
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
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
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
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
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
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Create datasets
    print("\nCreating datasets...")
    print(f"Using element analysis images: {IMAGE_SUFFIX}")
    train_dataset = SingleCoinDataset(DATA_DIR, split='train', transform=train_transform,
                                      max_samples_per_class=MAX_SAMPLES_PER_CLASS,
                                      image_suffix=IMAGE_SUFFIX)
    test_dataset = SingleCoinDataset(DATA_DIR, split='test', transform=val_transform,
                                     max_samples_per_class=MAX_SAMPLES_PER_CLASS,
                                     image_suffix=IMAGE_SUFFIX)
    val_dataset = SingleCoinDataset(DATA_DIR, split='val', transform=val_transform,
                                    max_samples_per_class=MAX_SAMPLES_PER_CLASS,
                                    image_suffix=IMAGE_SUFFIX)
    
    # Use weighted sampling for class balance if requested
    if USE_BALANCED_SAMPLING:
        class_sample_counts = [train_dataset.class_counts.get(i, 0) for i in range(len(train_dataset.class_to_idx))]
        class_weights = 1.0 / torch.tensor([max(c, 1) for c in class_sample_counts], dtype=torch.float)
        
        sample_weights = [class_weights[sample['label']].item() for sample in train_dataset.samples]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
        print("\n✓ Using WeightedRandomSampler")
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    print(f"\nTrain: {len(train_loader)} batches")
    print(f"Test: {len(test_loader)} batches")
    print(f"Val: {len(val_loader)} batches")
    
    # Create model
    print("\nCreating model...")
    num_classes = len(train_dataset.class_to_idx)
    model = SingleCNNClassifier(num_classes=num_classes)
    model = model.to(DEVICE)
    
    print(f"Classes: {num_classes}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss function with optional class weights
    if USE_CLASS_WEIGHTS:
        class_sample_counts = [train_dataset.class_counts.get(i, 0) for i in range(len(train_dataset.class_to_idx))]
        total_samples = sum(class_sample_counts)
        class_weights = torch.tensor(
            [total_samples / (len(class_sample_counts) * max(count, 1)) for count in class_sample_counts],
            dtype=torch.float32
        ).to(DEVICE)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("\n✓ Weighted loss:")
        for grade_name in sorted(train_dataset.class_to_idx.keys()):
            idx = train_dataset.class_to_idx[grade_name]
            if idx < len(class_weights):
                print(f"  {grade_name:8s}: {class_weights[idx]:.2f}x")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    # TensorBoard
    writer = SummaryWriter(LOG_DIR)
    
    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, 'coin_single_cnn_best.pth')
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Update LR
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
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
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
    
    writer.close()
    
    # Test evaluation
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
    
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE, split='Test')
    
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("="*60)
    
    # Save results
    with open(os.path.join(OUTPUT_DIR, 'history_single_cnn.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    config = {
        'architecture': 'SingleCNNClassifier',
        'image_type': IMAGE_SUFFIX,
        'data_source': DATA_DIR,
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'num_classes': num_classes,
        'classes': list(train_dataset.class_to_idx.keys()),
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'total_params': total_params
    }
    
    with open(os.path.join(OUTPUT_DIR, 'config_single_cnn.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history_single_cnn.png'), dpi=300)
    
    print(f"\n✓ All results saved!")
    print(f"  Model: {best_model_path}")
    print(f"  History: {OUTPUT_DIR}/history_single_cnn.json")
    print(f"  Plot: {OUTPUT_DIR}/training_history_single_cnn.png")

