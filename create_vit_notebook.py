#!/usr/bin/env python3
"""Convert coin_classifier_vit.py to Jupyter notebook."""

import json

cells = []

def add_md(text):
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': text.split('\n')
    })

def add_code(code):
    cells.append({
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': code.split('\n')
    })

# Title
add_md("""# Coin Grade Classifier - Vision Transformer (ViT)

Train a Vision Transformer model to classify coin grades using **both obverse and reverse** images at full **1000x1000 resolution**.

## Key Features:
- 🪙 **Dual-Image Input**: Uses both sides of each coin
- 🔍 **Full Resolution**: 1000x1000 pixels (preserves fine details)
- 🤖 **Vision Transformer**: State-of-the-art architecture
- 🔄 **Feature Fusion**: Combines information from both sides""")

# Imports
add_md("## 1. Setup and Imports")
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

# Config
add_md("## 2. Configuration")
add_code("""# Paths
DATA_DIR = 'images'
OUTPUT_DIR = 'models'
LOG_DIR = 'runs/vit_dual_' + datetime.now().strftime('%Y%m%d_%H%M%S')

# Hyperparameters
IMAGE_SIZE = 1000
BATCH_SIZE = 2
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4
PIN_MEMORY = True if torch.cuda.is_available() else False

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")""")

# Dataset
add_md("## 3. Dataset Class")
add_code("""class DualCoinDataset(Dataset):
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
            
            for obverse_img in obverse_images:
                reverse_img = reverse_dir / obverse_img.name
                if reverse_img.exists():
                    self.samples.append({
                        'obverse': obverse_img,
                        'reverse': reverse_img,
                        'label': idx,
                        'grade': grade_name
                    })
        
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
        
        return obverse, reverse, sample['label']""")

# Transforms
add_md("## 4. Data Transforms")
add_code("""train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("✓ Transforms configured")""")

# Create datasets
add_md("## 5. Create Datasets")
add_code("""train_dataset = DualCoinDataset(DATA_DIR, split='train', transform=train_transform)
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
print(f"Val: {len(val_loader)} batches")""")

# Model
add_md("## 6. Dual Vision Transformer Model")
add_code("""class DualViTClassifier(nn.Module):
    def __init__(self, num_classes, image_size=1000, pretrained=True):
        super().__init__()
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            obverse_vit = vit_b_16(weights=weights)
            reverse_vit = vit_b_16(weights=weights)
        else:
            obverse_vit = vit_b_16(weights=None)
            reverse_vit = vit_b_16(weights=None)
        
        self.obverse_encoder = nn.Sequential(obverse_vit.conv_proj, obverse_vit.encoder)
        self.reverse_encoder = nn.Sequential(reverse_vit.conv_proj, reverse_vit.encoder)
        self.feature_dim = 768
        
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, obverse, reverse):
        obverse_feat = self.obverse_encoder(obverse)[:, 0]
        reverse_feat = self.reverse_encoder(reverse)[:, 0]
        combined = torch.cat([obverse_feat, reverse_feat], dim=1)
        fused = self.fusion(combined)
        output = self.classifier(fused)
        return output

num_classes = len(train_dataset.class_to_idx)
model = DualViTClassifier(num_classes=num_classes, pretrained=True).to(DEVICE)

print(f"Classes: {num_classes}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")""")

# Training setup
add_md("## 7. Training Setup")
add_code("""criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
writer = SummaryWriter(LOG_DIR)

print(f"Optimizer: AdamW (LR={LEARNING_RATE})")
print(f"Scheduler: CosineAnnealingLR")""")

# Training functions
add_md("## 8. Training Functions")
add_code("""def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
    for obverse, reverse, labels in pbar:
        obverse, reverse, labels = obverse.to(device), reverse.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(obverse, reverse)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * obverse.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    return running_loss / total, 100 * correct / total

def validate(model, loader, criterion, device, split='Val'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f'{split}')
        for obverse, reverse, labels in pbar:
            obverse, reverse, labels = obverse.to(device), reverse.to(device), labels.to(device)
            outputs = model(obverse, reverse)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * obverse.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    return running_loss / total, 100 * correct / total

print("✓ Training functions defined")""")

# Main loop
add_md("## 9. Main Training Loop")
add_code("""history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0.0
best_model_path = os.path.join(OUTPUT_DIR, 'coin_vit_dual_best.pth')

print("\\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

for epoch in range(NUM_EPOCHS):
    print(f"\\nEpoch {epoch+1}/{NUM_EPOCHS}")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
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
print(f"\\nBest validation accuracy: {best_val_acc:.2f}%")""")

# Plots
add_md("## 10. Plot Training History")
add_code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

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
plt.savefig(os.path.join(OUTPUT_DIR, 'training_history_vit.png'), dpi=300)
plt.show()""")

# Test
add_md("## 11. Test Evaluation")
add_code("""checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

test_loss, test_acc = validate(model, test_loader, criterion, DEVICE, split='Test')

print(f"\\n{'='*60}")
print("FINAL TEST RESULTS")
print("="*60)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print("="*60)""")

# Save
add_md("## 12. Save Results")
add_code("""with open(os.path.join(OUTPUT_DIR, 'history_vit.json'), 'w') as f:
    json.dump(history, f, indent=2)

config = {
    'image_size': IMAGE_SIZE,
    'batch_size': BATCH_SIZE,
    'num_epochs': NUM_EPOCHS,
    'learning_rate': LEARNING_RATE,
    'num_classes': num_classes,
    'classes': list(train_dataset.class_to_idx.keys()),
    'best_val_acc': best_val_acc,
    'test_acc': test_acc
}

with open(os.path.join(OUTPUT_DIR, 'config_vit.json'), 'w') as f:
    json.dump(config, f, indent=2)

print("\\n✓ Results saved!")
print(f"  Model: {best_model_path}")
print(f"  History: {OUTPUT_DIR}/history_vit.json")
print(f"  Config: {OUTPUT_DIR}/config_vit.json")""")

# Create notebook
notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
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

with open('coin_classifier_vit.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✓ Created coin_classifier_vit.ipynb")
print(f"  Total cells: {len(cells)}")
print(f"  Markdown: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"  Code: {sum(1 for c in cells if c['cell_type'] == 'code')}")













