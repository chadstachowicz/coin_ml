# 🚀 Quick Start Guide

Get from zero to trained coin classifier in minutes!

## 📦 Step 1: Install Dependencies (2 minutes)

```bash
pip install -r requirements.txt
```

Required packages:
- Flask (web scraper)
- PyTorch (deep learning)
- torchvision (image processing)
- transformers (models - only for ResNet)

## 🪙 Step 2: Collect Coins (10-30 minutes)

### Start the Web Scraper
```bash
python app.py
```

### Open Browser
Go to http://localhost:8000

### Bulk Upload Cert Numbers
1. Click **"Bulk Upload"** tab
2. Paste your PCGS cert numbers (one per line or comma-separated):
   ```
   12345678
   87654321
   11223344
   ...
   ```
3. Click **"Process All"**
4. Watch the progress bar as it downloads coins and images

### Collect at Least:
- **Minimum**: 20 images per grade
- **Recommended**: 50+ images per grade
- **Optimal**: 100+ images per grade

Images are saved to: `images/<grade>/6000x3000/`

## 📊 Step 3: Check Your Dataset (30 seconds)

```bash
python check_dataset.py
```

This shows:
- How many images per grade
- Dataset balance
- **Recommended training method** based on your data

Example output:
```
📊 DATASET OVERVIEW
Grade (Class)        Images     Status          Bar
──────────────────────────────────────────────────────
ms65                 45         ✅ Good         ████████████
au50                 32         ⚡ OK           ███████
g04                  28         ⚡ OK           ██████
```

## 🎓 Step 4: Prepare Training Data (1 minute)

```bash
python prepare_dataset.py
```

This creates:
- `coin_dataset/train/` (70% of images)
- `coin_dataset/test/` (20% of images)
- `coin_dataset/val/` (10% of images)

## 🤖 Step 5: Train Your Model

### Option A: Custom CNN (Recommended for Rectangular Images)

**Use if you have 100+ images per class**

```bash
python coin_classifier_custom.py
```

Features:
- ✅ Preserves rectangular 2:1 aspect ratio
- ✅ Trains from scratch on your coins
- ✅ Highest accuracy potential
- ⏱️ Takes 2-4 hours

### Option B: ResNet-50 (Faster for Small Datasets)

**Use if you have < 50 images per class**

```bash
jupyter notebook coin_classifier_full.ipynb
```

Then run all cells.

Features:
- ✅ Fast training (30-60 minutes)
- ✅ Works with limited data
- ✅ Pre-trained on ImageNet
- ⚠️ Converts to square images

### Still Unsure?

See [TRAINING_COMPARISON.md](TRAINING_COMPARISON.md) for detailed comparison.

## 📈 Step 6: Monitor Training (Optional)

In a new terminal:
```bash
tensorboard --logdir=runs
```

Open http://localhost:6006 to see:
- Training/validation loss curves
- Accuracy over time
- Learning rate schedule

## 🎯 Step 7: Use Your Trained Model

After training completes, you'll have:
- `coin_classifier_custom_best.pth` (Custom CNN), or
- `coin_classifier_best.pth` (ResNet-50)

### Quick Prediction Script

```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
checkpoint = torch.load('coin_classifier_custom_best.pth')
classes = checkpoint['classes']

# TODO: Load your model architecture here
# model = ... 
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
img = Image.open('coin.jpg')
transform = transforms.Compose([
    transforms.Resize((256, 512)),  # For custom CNN (512x256)
    # transforms.Resize((256, 256)),  # For ResNet-50
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_tensor = transform(img).unsqueeze(0)
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    print(f"Predicted grade: {classes[predicted.item()]}")
```

## 🆘 Troubleshooting

### "No images found"
- Make sure you scraped coins using the web UI first
- Images should be in `images/<grade>/6000x3000/` folders

### "Out of memory"
- Reduce batch size in the training script
- Use smaller image size
- Close other programs

### "CUDA not available"
- Training will use CPU (slower but works)
- Or install CUDA: https://pytorch.org/get-started/locally/

### "Low accuracy (< 60%)"
- Collect more images (aim for 50+ per class)
- Train for more epochs
- Check data quality (are images clear?)
- Balance your dataset (similar counts per class)

## 📚 Documentation

- **[TRAINING_COMPARISON.md](TRAINING_COMPARISON.md)** - Which method to choose
- **[CUSTOM_CNN_TRAINING.md](CUSTOM_CNN_TRAINING.md)** - Custom CNN details
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - ResNet-50 details

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Install dependencies | 2-5 min |
| Collect 100 coins | 10-20 min |
| Prepare dataset | 1 min |
| Train (Custom CNN) | 2-4 hours |
| Train (ResNet-50) | 30-60 min |
| **Total (Custom CNN)** | **3-5 hours** |
| **Total (ResNet-50)** | **1-2 hours** |

## 🎯 Success Checklist

- [ ] Installed dependencies
- [ ] Scraped coins using web UI
- [ ] Have 20+ images per grade
- [ ] Ran `check_dataset.py`
- [ ] Prepared dataset with `prepare_dataset.py`
- [ ] Chose training method (Custom CNN or ResNet-50)
- [ ] Started training
- [ ] Monitoring with TensorBoard (optional)
- [ ] Model achieves > 80% test accuracy
- [ ] Saved best model checkpoint

## 🚀 Next Level

Once you have a working model:

1. **Collect more data** - More = better accuracy
2. **Balance classes** - Equal images per grade
3. **Try data augmentation** - Helps with overfitting
4. **Experiment with hyperparameters** - Learning rate, epochs, etc.
5. **Build an API** - Deploy for real-time predictions
6. **Create a web app** - Upload coin images and get predictions
7. **Ensemble models** - Combine Custom CNN + ResNet-50

Happy coin classifying! 🪙🤖













