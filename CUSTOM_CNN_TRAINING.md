# Custom CNN Training for Coin Classification

Train a custom CNN from scratch on rectangular coin images (preserving 2:1 aspect ratio).

## 🎯 Why Custom CNN?

- **Preserves aspect ratio**: Your coin images are 6000x3000 (2:1 ratio) - rectangular, not square
- **Train from scratch**: No pre-trained weights, learns specifically from your coin data
- **Optimized architecture**: VGG-style CNN designed for rectangular image classification
- **Full control**: Customize every layer for your specific use case

## 🏗️ Architecture

Based on VGG-style architecture with modifications for rectangular inputs:

```
Input: 512x256 (2:1 aspect ratio, scaled down from 6000x3000)

Block 1:  Conv(3→64)  → Conv(64→64)   → MaxPool → 256x128
Block 2:  Conv(64→128) → Conv(128→128) → MaxPool → 128x64
Block 3:  Conv(128→256) → Conv(256→256) → Conv(256→256) → MaxPool → 64x32
Block 4:  Conv(256→512) → Conv(512→512) → Conv(512→512) → MaxPool → 32x16
Block 5:  Conv(512→512) → Conv(512→512) → MaxPool → 16x8

Flatten → FC(65536→2048) → Dropout(0.5) → FC(2048→1024) → Dropout(0.5) → FC(1024→num_classes)
```

**Total Parameters**: ~100M (varies by number of classes)

## 🚀 Quick Start

### Step 1: Prepare Dataset
```bash
python prepare_dataset.py
```

This creates the train/test/val split from your `images/<grade>/6000x3000/` folders.

### Step 2: Train the Model
```bash
python coin_classifier_custom.py
```

That's it! The script will:
- Load your rectangular coin images
- Train the custom CNN from scratch
- Save the best model
- Evaluate on test set

### Step 3: Monitor Training
```bash
tensorboard --logdir=runs
```

Open http://localhost:6006 to view:
- Training/validation loss
- Training/validation accuracy
- Learning rate schedule

## ⚙️ Configuration

Edit the top of `coin_classifier_custom.py`:

```python
# Image size (keeps 2:1 aspect ratio)
IMAGE_WIDTH = 512   # Original: 6000
IMAGE_HEIGHT = 256  # Original: 3000

# Training parameters
BATCH_SIZE = 4      # Reduce if out of memory
NUM_EPOCHS = 50     # Increase for more training
LEARNING_RATE = 0.001  # Adjust if not converging
```

### Recommended Settings by GPU

**CPU / Small GPU (< 8GB)**
```python
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 128
BATCH_SIZE = 2
```

**Medium GPU (8-16GB)**
```python
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 256
BATCH_SIZE = 4
```

**Large GPU (> 16GB)**
```python
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 512
BATCH_SIZE = 8
```

## 📊 Expected Results

### Training Time (per epoch)
- CPU: ~10-20 minutes
- GPU: ~1-3 minutes

### Target Accuracy
- **50+ images/class**: 70-85% test accuracy
- **100+ images/class**: 85-95% test accuracy
- **200+ images/class**: 90-98% test accuracy

### Signs of Good Training
- ✅ Training loss decreases steadily
- ✅ Validation accuracy within 10% of training
- ✅ No sudden spikes in loss
- ✅ Learning rate decreases over time

## 🎛️ Hyperparameter Tuning

### If Overfitting (train acc >> val acc)
```python
# 1. Increase dropout
self.dropout1 = nn.Dropout(0.7)  # was 0.5

# 2. Add weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 3. Use stronger augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # Increase rotation
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # Stronger
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### If Underfitting (both train and val acc low)
```python
# 1. Increase model capacity
self.fc1 = nn.Linear(512 * 16 * 8, 4096)  # was 2048
self.fc2 = nn.Linear(4096, 2048)  # was 1024

# 2. Train longer
NUM_EPOCHS = 100  # was 50

# 3. Increase learning rate
LEARNING_RATE = 0.003  # was 0.001
```

### If Training is Slow
```python
# 1. Reduce image size
IMAGE_WIDTH = 256  # was 512
IMAGE_HEIGHT = 128  # was 256

# 2. Reduce model size
# Comment out Block 5 in the model

# 3. Increase batch size (if GPU allows)
BATCH_SIZE = 8  # was 4
```

## 📁 Output Files

After training:

```
coin_classifier_custom_best.pth   # Saved model checkpoint
runs/coin_custom_TIMESTAMP/       # TensorBoard logs
```

The checkpoint contains:
- Model weights
- Optimizer state
- Best validation accuracy
- Class names (grades)

## 🔮 Using the Trained Model

```python
import torch
from PIL import Image
from torchvision import transforms

# Load checkpoint
checkpoint = torch.load('coin_classifier_custom_best.pth')
classes = checkpoint['classes']

# Recreate model
from coin_classifier_custom import CoinClassifier
model = CoinClassifier(num_classes=len(classes))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((256, 512)),  # Match training size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and predict
img = Image.open('path/to/coin.jpg')
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(img_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)
    
    grade = classes[predicted.item()]
    conf_pct = confidence.item() * 100
    
    print(f"Predicted Grade: {grade}")
    print(f"Confidence: {conf_pct:.2f}%")
    
    # Show top 3 predictions
    top3_conf, top3_idx = torch.topk(probabilities, 3)
    print("\nTop 3 Predictions:")
    for i in range(3):
        print(f"  {classes[top3_idx[0][i]]}: {top3_conf[0][i]*100:.2f}%")
```

## 🆘 Troubleshooting

### Out of Memory Error
```python
# Solution 1: Reduce image size
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 128

# Solution 2: Reduce batch size
BATCH_SIZE = 1

# Solution 3: Use gradient accumulation
# (Accumulate gradients over multiple batches)
```

### Model Not Learning (accuracy stuck at ~random)
- Check dataset: Are images in correct folders?
- Check labels: Do class names make sense?
- Increase learning rate: Try 0.01
- Train longer: 100+ epochs
- Visualize images: Make sure they're loading correctly

### Validation Accuracy Fluctuating Wildly
- Increase validation set size (adjust split ratios)
- Use learning rate scheduling (already included)
- Increase batch size for more stable gradients

### Training Loss Exploding (NaN)
- Reduce learning rate: Try 0.0001
- Check for corrupted images
- Add gradient clipping:
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```

## 📈 Advanced: Custom Architecture

Want to modify the architecture? Here's how:

```python
class CoinClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CoinClassifier, self).__init__()
        
        # Add more blocks for deeper network
        self.conv_block6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Or use residual connections
        self.residual = nn.Conv2d(256, 512, 1)  # Skip connection
        
        # Or try different activation functions
        self.activation = nn.LeakyReLU(0.2)  # Instead of ReLU
```

## 🎓 Tips for Best Results

1. **Collect Balanced Data**: Same number of images per grade
2. **Quality Over Quantity**: Clear, well-lit images beat many poor ones
3. **Use Data Augmentation**: Helps with small datasets
4. **Monitor Overfitting**: Watch the train/val accuracy gap
5. **Save Checkpoints**: Train can be interrupted and resumed
6. **Experiment**: Try different hyperparameters
7. **Use GPU**: 10-20x faster than CPU

## 📝 Comparison: Custom CNN vs ResNet-50

| Feature | Custom CNN | ResNet-50 |
|---------|-----------|-----------|
| **Aspect Ratio** | Preserves 2:1 | Forces square |
| **Training** | From scratch | Fine-tuning |
| **Parameters** | ~100M | ~25M |
| **Training Time** | Longer | Faster |
| **Data Required** | More | Less |
| **Flexibility** | Full control | Limited |
| **Best For** | Rectangular images | Square images |

Use **Custom CNN** when:
- Images are rectangular (like coins)
- You have 100+ images per class
- You want full architectural control
- You have GPU resources

Use **ResNet-50** when:
- Images can be cropped to square
- You have limited data (< 50 per class)
- You want faster training
- You want proven architecture

---

Happy training! Your custom coin classifier awaits! 🪙🤖













