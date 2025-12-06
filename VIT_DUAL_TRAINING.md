# Vision Transformer Training - Dual Image Classification

This guide explains how to train a Vision Transformer (ViT) model that uses **both obverse and reverse** coin images at **full 1000x1000 resolution** for accurate grade classification.

## 🌟 Why Vision Transformer + Dual Images?

### Advantages

1. **Complete Information**: Uses both sides of the coin
2. **Full Resolution**: 1000x1000 pixels - preserves all fine details
3. **State-of-the-Art**: ViT architecture excels at image understanding
4. **Attention Mechanism**: Learns which coin features matter most
5. **Transfer Learning**: Pretrained on ImageNet for better starting point

### Perfect For

- Coins where both sides matter for grading
- Datasets with high-resolution images
- Cases where fine details (scratches, luster, strikes) are important
- Professional grading that considers both obverse and reverse

## 📁 Required Dataset Structure

Your `images/` folder must have this structure:

```
images/
├── g04/
│   ├── obverse/
│   │   ├── coin1.jpg    ← 1000x1000 obverse
│   │   ├── coin2.jpg
│   │   └── ...
│   └── reverse/
│       ├── coin1.jpg    ← 1000x1000 reverse (same filename!)
│       ├── coin2.jpg
│       └── ...
├── ms65/
│   ├── obverse/
│   └── reverse/
├── au50/
│   ├── obverse/
│   └── reverse/
└── ...
```

**Important**: Obverse and reverse images must have **matching filenames** so the dataset loader can pair them correctly.

## 🚀 Quick Start

### 1. Prepare Your Data

Make sure your images are organized with matching obverse/reverse pairs:

```bash
# Check your data structure
ls images/*/obverse/ | head
ls images/*/reverse/ | head
```

### 2. Run Training

```bash
python coin_classifier_vit.py
```

That's it! The script will:
- Load paired obverse/reverse images
- Split data 70/20/10 (train/test/val)
- Train Vision Transformer model
- Save best model automatically
- Generate training plots
- Evaluate on test set

### 3. Monitor Training

In another terminal, run:

```bash
tensorboard --logdir=runs
```

Open http://localhost:6006 to see:
- Loss curves
- Accuracy metrics
- Learning rate schedule

## 🏗️ Model Architecture

```
Input: Obverse (1000x1000) + Reverse (1000x1000)
   ↓                            ↓
ViT Encoder                  ViT Encoder
(pretrained)                 (pretrained)
   ↓                            ↓
CLS Token [768]              CLS Token [768]
   ↓                            ↓
   └─────── Concatenate ────────┘
                ↓
           Fusion Layer
         (768*2 → 1024)
                ↓
         Classification
          (1024 → 512 → classes)
                ↓
            Output
```

### Key Components

1. **Dual ViT Encoders**: Two separate Vision Transformers process obverse and reverse
2. **Feature Extraction**: Each ViT outputs 768-dimensional features
3. **Fusion Layer**: Combines information from both sides (1536 → 1024)
4. **Classification Head**: Final layers predict coin grade

## ⚙️ Configuration

Edit these variables in `coin_classifier_vit.py`:

```python
# Model settings
IMAGE_SIZE = 1000      # Keep at 1000 for full resolution
BATCH_SIZE = 2         # Increase if you have more GPU memory
NUM_EPOCHS = 30        # More epochs = better training

# Training settings
LEARNING_RATE = 1e-4   # Lower for fine-tuning
WEIGHT_DECAY = 0.01    # Regularization
```

### GPU Memory

- **2GB VRAM**: `BATCH_SIZE = 1`
- **4GB VRAM**: `BATCH_SIZE = 2`
- **8GB VRAM**: `BATCH_SIZE = 4`
- **16GB+ VRAM**: `BATCH_SIZE = 8`

If you run out of memory:
1. Reduce `BATCH_SIZE`
2. Reduce `IMAGE_SIZE` (e.g., 800 or 768)
3. Use CPU (slower): `DEVICE = torch.device('cpu')`

## 📊 Expected Results

### Training Time

- **RTX 3090 (24GB)**: ~2-3 hours (batch size 8)
- **RTX 3060 (12GB)**: ~4-6 hours (batch size 4)
- **GTX 1080 (8GB)**: ~8-12 hours (batch size 2)
- **CPU**: ~48+ hours (not recommended)

### Accuracy

With good data:
- **100+ images per class**: 85-95% accuracy
- **50-100 images per class**: 75-85% accuracy
- **< 50 images per class**: 65-75% accuracy

Accuracy improves with:
- More training data
- Better image quality
- Consistent grading standards
- Balanced classes

## 📈 Output Files

After training, you'll have:

```
models/
├── coin_vit_dual_best.pth      ← Best model checkpoint
├── history_vit.json            ← Training metrics
├── config_vit.json             ← Model configuration
└── training_history_vit.png    ← Training plots

runs/
└── vit_dual_YYYYMMDD_HHMMSS/  ← TensorBoard logs
```

## 🎯 Using the Trained Model

### Load and Predict

```python
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load model
checkpoint = torch.load('models/coin_vit_dual_best.pth')
model = DualViTClassifier(num_classes=len(checkpoint['class_to_idx']))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and transform images
transform = transforms.Compose([
    transforms.Resize((1000, 1000)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

obverse = transform(Image.open('test_obverse.jpg').convert('RGB')).unsqueeze(0)
reverse = transform(Image.open('test_reverse.jpg').convert('RGB')).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(obverse, reverse)
    predicted_class = torch.argmax(output, dim=1).item()
    predicted_grade = checkpoint['idx_to_class'][str(predicted_class)]
    confidence = torch.softmax(output, dim=1)[0][predicted_class].item()

print(f"Predicted Grade: {predicted_grade}")
print(f"Confidence: {confidence:.2%}")
```

## 🔧 Troubleshooting

### Issue: CUDA out of memory

**Solution**: Reduce batch size
```python
BATCH_SIZE = 1  # Start with 1
```

### Issue: Training is too slow

**Solutions**:
1. Use GPU instead of CPU
2. Reduce `NUM_WORKERS`
3. Reduce `IMAGE_SIZE` (trade-off: less detail)

### Issue: Low accuracy

**Solutions**:
1. Collect more training data (aim for 100+ per class)
2. Balance your dataset (similar counts per grade)
3. Train for more epochs
4. Verify obverse/reverse pairs are matched correctly
5. Check image quality

### Issue: Missing obverse/reverse pairs

**Error**: `Warning: No matching reverse for xxx.jpg`

**Solution**: Ensure every obverse image has a matching reverse with the same filename:
```bash
# Check for unpaired images
for f in images/*/obverse/*.jpg; do
    base=$(basename "$f")
    grade=$(basename $(dirname $(dirname "$f")))
    if [ ! -f "images/$grade/reverse/$base" ]; then
        echo "Missing reverse for: $f"
    fi
done
```

### Issue: Not enough classes

**Error**: Need at least 2 classes

**Solution**: Make sure you have at least 2 different grade folders with obverse/reverse pairs

## 📚 Technical Details

### Why Vision Transformer?

Traditional CNNs process images with fixed receptive fields. ViT uses **self-attention** to:
1. Look at the entire image at once
2. Learn relationships between distant parts
3. Focus on relevant features (scratches, luster, details)
4. Handle large images efficiently

### Why Dual Images?

Professional coin grading considers:
- **Obverse**: Main design, major features, primary condition
- **Reverse**: Back design, complementary details, overall preservation

Both sides together provide:
- Complete condition assessment
- Detection of asymmetric wear
- Full understanding of coin preservation
- More accurate grade prediction

### Fusion Strategy

We concatenate features from both encoders because:
1. **Late fusion**: Allows each side to be processed independently first
2. **Full information**: Preserves all details from both sides
3. **Flexible**: Fusion layer learns optimal combination
4. **Proven**: Works well in multi-view classification tasks

## 🎓 Next Steps

1. **Experiment with augmentation**: Adjust ColorJitter, add rotation
2. **Try different fusion**: Average, attention-based, or multiplicative
3. **Fine-tune more layers**: Unfreeze earlier ViT layers for your specific domain
4. **Ensemble models**: Combine multiple models for better predictions
5. **Add more features**: Incorporate metadata (year, mint mark, variety)

## 📖 References

- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) - Original ViT paper
- [Vision Transformer (ViT)](https://github.com/google-research/vision_transformer) - Official repo
- [PyTorch Vision Transformers](https://pytorch.org/vision/stable/models/vision_transformer.html) - Documentation

---

Happy training! 🪙🤖




