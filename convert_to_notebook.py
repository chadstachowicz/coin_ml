#!/usr/bin/env python3
"""Convert coin_classifier_vit.py to a Jupyter notebook."""

import json

# Read the source file
with open('coin_classifier_vit.py', 'r') as f:
    lines = f.readlines()

# Define notebook cells
cells = []

def add_markdown(text):
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': text.strip().split('\n')
    })

def add_code(code):
    cells.append({
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': code.strip().split('\n')
    })

# Cell 1: Title and introduction
add_markdown("""# Coin Grade Classifier - Vision Transformer (ViT)

This notebook trains a Vision Transformer model to classify coin grades using **both obverse and reverse** images at full **1000x1000 resolution**.

## Key Features:
- 🪙 **Dual-Image Input**: Uses both sides of each coin
- 🔍 **Full Resolution**: 1000x1000 pixels (preserves fine details)
- 🤖 **Vision Transformer**: State-of-the-art architecture
- 🔄 **Feature Fusion**: Combines information from both sides

## Architecture:
```
Obverse (1000x1000)     Reverse (1000x1000)
        ↓                       ↓
   ViT Encoder             ViT Encoder
    (768 dim)               (768 dim)
        ↓                       ↓
        └──── Concatenate ──────┘
                 ↓
           Fusion Layer
             (1024)
                 ↓
       Classification Head
                 ↓
            Coin Grade
```""")

# Cell 2: Imports header
add_markdown("## 1. Setup and Imports")

# Cell 3: Imports code
add_code("""import torch
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

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")""")

# Cell 4: Configuration header
add_markdown("""## 2. Configuration

Adjust these settings based on your hardware and dataset:
- **IMAGE_SIZE**: Keep at 1000 for full resolution
- **BATCH_SIZE**: Increase if you have more GPU memory (8GB+ → use 4)
- **NUM_EPOCHS**: More epochs = better training (if not overfitting)""")

# Cell 5: Configuration code
add_code("""# Paths
DATA_DIR = 'images'  # Directory containing <grade>/obverse/ and <grade>/reverse/
OUTPUT_DIR = 'models'
LOG_DIR = 'runs/vit_dual_' + datetime.now().strftime('%Y%m%d_%H%M%S')

# Model hyperparameters
IMAGE_SIZE = 1000  # Keep full resolution
BATCH_SIZE = 2     # Small batch due to large images (increase if you have GPU memory)
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

# Training settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4
PIN_MEMORY = True if torch.cuda.is_available() else False

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")""")

# Cell 6: Dataset header
add_markdown("""## 3. Dataset Class - Dual Image Loader

Loads both obverse and reverse images for each coin.

**Important**: Obverse and reverse images must have matching filenames!""")

# Cell 7: Dataset class (from lines 72-155 approx)
dataset_code = """class DualCoinDataset(Dataset):
    \"\"\"Dataset that loads both obverse and reverse images for each coin.\"\"\"
    
    def __init__(self, data_dir, split='train', transform=None):
        \"\"\"
        Args:
            data_dir: Root directory with structure: <grade>/obverse/ and <grade>/reverse/
            split: 'train', 'test', or 'val'
            transform: Optional transform to apply to both images
        \"\"\"
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
        
        return obverse, reverse, sample['label']"""
add_code(dataset_code)

# Cell 8: Transforms header
add_markdown("## 4. Data Transforms\n\nMinimal transforms to preserve coin details.")

# Cell 9: Transforms code
add_code("""# Training transforms - minimal augmentation to preserve detail
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

print("✓ Transforms configured")""")

# Cell 10: Create datasets header
add_markdown("## 5. Create Datasets and DataLoaders")

# Cell 11: Create datasets code
add_code("""# Create datasets
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

print(f"\\nDataloaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")
print(f"  Val batches: {len(val_loader)}")""")

# Cell 12: Model architecture header
add_markdown("""## 6. Dual Vision Transformer Model

Two ViT encoders (one per side) with feature fusion.""")

# Cell 13: Model code
model_code = """class DualViTClassifier(nn.Module):
    \"\"\"Vision Transformer for dual-image (obverse + reverse) classification.\"\"\"
    
    def __init__(self, num_classes, image_size=1000, pretrained=True):
        super(DualViTClassifier, self).__init__()
        
        # Import ViT model
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        
        # Load pretrained ViT-B/16 models for both sides
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            obverse_vit = vit_b_16(weights=weights)
            reverse_vit = vit_b_16(weights=weights)
        else:
            obverse_vit = vit_b_16(weights=None)
            reverse_vit = vit_b_16(weights=None)
        
        # Extract feature extractors (remove classification head)
        self.obverse_encoder = nn.Sequential(
            obverse_vit.conv_proj,
            obverse_vit.encoder
        )
        self.reverse_encoder = nn.Sequential(
            reverse_vit.conv_proj,
            reverse_vit.encoder
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
        # Process obverse image
        obverse_feat = self.obverse_encoder(obverse)  # [B, N+1, 768]
        obverse_feat = obverse_feat[:, 0]  # Take CLS token [B, 768]
        
        # Process reverse image
        reverse_feat = self.reverse_encoder(reverse)  # [B, N+1, 768]
        reverse_feat = reverse_feat[:, 0]  # Take CLS token [B, 768]
        
        # Concatenate features from both sides
        combined = torch.cat([obverse_feat, reverse_feat], dim=1)  # [B, 1536]
        
        # Fusion
        fused = self.fusion(combined)  # [B, 1024]
        
        # Classification
        output = self.classifier(fused)  # [B, num_classes]
        
        return output


# Create model
num_classes = len(train_dataset.class_to_idx)
model = DualViTClassifier(num_classes=num_classes, image_size=IMAGE_SIZE, pretrained=True)
model = model.to(DEVICE)

print(f"\\nModel created:")
print(f"  Number of classes: {num_classes}")
print(f"  Classes: {list(train_dataset.class_to_idx.keys())}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")"""
add_code(model_code)

# Cell 14: Training setup header
add_markdown("## 7. Training Setup")

# Cell 15: Training setup code
add_code("""# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# TensorBoard writer
writer = SummaryWriter(LOG_DIR)

print("Training setup complete")
print(f"  Optimizer: AdamW")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Scheduler: CosineAnnealingLR")
print(f"  TensorBoard logs: {LOG_DIR}")""")

# Cell 16: Training functions header
add_markdown("## 8. Training and Validation Functions")

# Cell 17: Training functions code
training_funcs = """def train_epoch(model, loader, criterion, optimizer, device, epoch):
    \"\"\"Train for one epoch.\"\"\"
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
    \"\"\"Validate the model.\"\"\"
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

print("✓ Training functions defined")"""
add_code(training_funcs)

# Cell 18: Main training loop header
add_markdown("## 9. Main Training Loop\n\n**Run this cell to start training!**")

# Cell 19: Main training loop code
training_loop = """# Training history
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

best_val_acc = 0.0
best_model_path = os.path.join(OUTPUT_DIR, 'coin_vit_dual_best.pth')

print("\\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

for epoch in range(NUM_EPOCHS):
    print(f"\\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 60)
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, split='Val')
    
    # Update learning rate
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    
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
    print(f"\\nEpoch {epoch+1} Summary:")
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

print("\\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Best model saved to: {best_model_path}")

writer.close()"""
add_code(training_loop)

# Cell 20: Plot training history header
add_markdown("## 10. Plot Training History")

# Cell 21: Plot code
add_code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

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

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_history_vit.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"Training plots saved to: {OUTPUT_DIR}/training_history_vit.png")""")

# Cell 22: Test evaluation header
add_markdown("## 11. Test Set Evaluation")

# Cell 23: Test evaluation code
add_code("""# Load best model
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

# Evaluate on test set
print("\\nEvaluating on test set...")
test_loss, test_acc = validate(model, test_loader, criterion, DEVICE, split='Test')

print(f"\\n{'='*60}")
print("FINAL TEST RESULTS")
print("="*60)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print("="*60)""")

# Cell 24: TensorBoard info
add_markdown("""## 12. View TensorBoard Logs

Run this command in terminal to view training progress:

```bash
tensorboard --logdir=runs
```

Then open http://localhost:6006""")

# Cell 25: Save results header
add_markdown("## 13. Save Final Results")

# Cell 26: Save results code
add_code("""# Save training history
with open(os.path.join(OUTPUT_DIR, 'history_vit.json'), 'w') as f:
    json.dump(history, f, indent=2)

# Save configuration
config = {
    'image_size': IMAGE_SIZE,
    'batch_size': BATCH_SIZE,
    'num_epochs': NUM_EPOCHS,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'num_classes': num_classes,
    'classes': list(train_dataset.class_to_idx.keys()),
    'best_val_acc': best_val_acc,
    'test_acc': test_acc
}

with open(os.path.join(OUTPUT_DIR, 'config_vit.json'), 'w') as f:
    json.dump(config, f, indent=2)

print("\\nResults saved:")
print(f"  Model: {best_model_path}")
print(f"  History: {OUTPUT_DIR}/history_vit.json")
print(f"  Config: {OUTPUT_DIR}/config_vit.json")
print(f"  Plots: {OUTPUT_DIR}/training_history_vit.png")
print(f"\\n✓ All done! Best validation accuracy: {best_val_acc:.2f}%")
print(f"✓ Test accuracy: {test_acc:.2f}%")""")

# Create notebook structure
notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {
                'name': 'ipython',
                'version': 3
            },
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.8.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

# Save notebook
with open('coin_classifier_vit.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("✓ Created coin_classifier_vit.ipynb with 26 cells")
print("  - 13 markdown cells")
print("  - 13 code cells")
print("\\nReady to use in Jupyter!")

