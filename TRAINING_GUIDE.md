# PCGS Coin Grade Classification - Training Guide

This guide explains how to prepare your coin image dataset and train a deep learning model to classify coin grades.

## 📁 Current Image Structure

Your images are currently organized as:
```
images/
  <grade>/              ← Grade folders (e.g., ms65, au50, g04)
    obverse/            ← Obverse images (not used for training)
    reverse/            ← Reverse images (not used for training)
    6000x3000/          ← High-resolution images (used for training)
      coin-image1.jpg
      coin-image2.jpg
      ...
```

The `<grade>` folders are your **class labels** for classification.

## 🚀 Quick Start

### Step 1: Prepare the Dataset

Run the dataset preparation script to organize images into train/test/val splits:

```bash
python prepare_dataset.py
```

This will:
- Scan all `images/<grade>/6000x3000/` folders
- Split images into 70% train, 20% test, 10% validation
- Create a new `coin_dataset/` folder with proper structure:

```
coin_dataset/
  train/
    ms65/
      image1.jpg
      image2.jpg
    au50/
      image3.jpg
    ...
  test/
    ms65/
      ...
  val/
    ms65/
      ...
```

### Step 2: Train the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook coin_classifier_full.ipynb
```

Or use JupyterLab:

```bash
jupyter lab coin_classifier_full.ipynb
```

**Run all cells in order** - the notebook will:
1. Load and preprocess images (resize to 256x256)
2. Load a pre-trained ResNet-50 model
3. Fine-tune it on your coin images
4. Save the best model as `coin_classifier_best.pth`
5. Evaluate on the test set

### Step 3: Monitor Training

While training, you can monitor progress with TensorBoard:

```bash
tensorboard --logdir=runs
```

Then open http://localhost:6006 to see:
- Training/validation loss curves
- Training/validation accuracy curves
- Real-time metrics during training

## 📊 Dataset Requirements

### Minimum Requirements
- **At least 10 images per grade** (class)
- **At least 2 different grades** (classes)
- **Recommended: 50+ images per grade** for good results

### Current Dataset Status

After collecting coins, run this to check your dataset:

```bash
python -c "
from pathlib import Path
from collections import defaultdict

images_by_grade = defaultdict(int)
for grade_folder in Path('images').iterdir():
    if grade_folder.is_dir() and (grade_folder / '6000x3000').exists():
        count = len(list((grade_folder / '6000x3000').glob('*.jpg')))
        images_by_grade[grade_folder.name] = count

print('Current Dataset:')
print('-' * 40)
for grade, count in sorted(images_by_grade.items()):
    print(f'{grade:20s}: {count:4d} images')
print('-' * 40)
print(f'Total Grades: {len(images_by_grade)}')
print(f'Total Images: {sum(images_by_grade.values())}')
"
```

## 🎯 Training Tips

### 1. Data Collection
- Collect coins with consistent grade distributions
- Aim for balanced classes (similar number of images per grade)
- More data = better model performance

### 2. Hyperparameters to Tune

**In the notebook, you can adjust:**

```python
# Image size (larger = more detail, slower training)
transforms.Resize(256)  # Try 224, 256, 384, 512

# Batch size (larger = faster, needs more GPU memory)
BATCH_SIZE = 4  # Try 4, 8, 16, 32

# Learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Try 0.0001, 0.001, 0.01

# Number of epochs
NUM_EPOCHS = 20  # Try 10, 20, 30, 50
```

### 3. Data Augmentation

Uncomment the augmentation transforms in the notebook to help prevent overfitting:

```python
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.RandomHorizontalFlip(),      # Flip coins horizontally
    transforms.RandomRotation(10),          # Slight rotation
    transforms.ColorJitter(brightness=0.2), # Adjust brightness
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 4. Model Selection

The notebook uses ResNet-50 by default. You can try other models:

```python
# ResNet-50 (default)
model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50', ...)

# ResNet-18 (faster, less accurate)
model = ResNetForImageClassification.from_pretrained('microsoft/resnet-18', ...)

# ResNet-152 (slower, more accurate)
model = ResNetForImageClassification.from_pretrained('microsoft/resnet-152', ...)
```

## 📈 Interpreting Results

### Training Metrics

- **Training Loss**: Should decrease over time
- **Validation Loss**: Should decrease (if increasing = overfitting)
- **Training Accuracy**: Should increase (target: >90%)
- **Validation Accuracy**: Should increase (gap from training = overfitting)
- **Test Accuracy**: Final performance on unseen data

### What's Good?

- ✅ Train accuracy > 85%
- ✅ Val accuracy within 5-10% of train accuracy
- ✅ Test accuracy similar to val accuracy
- ✅ Loss curves decreasing smoothly

### Warning Signs

- ⚠️ Train accuracy high (>95%) but val accuracy low (<70%) = **Overfitting**
  - Solution: More data, more augmentation, smaller model, dropout
  
- ⚠️ Both train and val accuracy low (<60%) = **Underfitting**
  - Solution: Larger model, more epochs, higher learning rate, check data quality
  
- ⚠️ Loss increasing after initial decrease = **Learning rate too high**
  - Solution: Decrease learning rate (e.g., 0.001 → 0.0001)

## 🔄 Collecting More Data

To improve your model, add more cert numbers to your scraper:

1. Use the bulk upload feature in the web UI
2. Paste lists of cert numbers (one per line or comma-separated)
3. The scraper will automatically organize new images
4. Re-run `prepare_dataset.py` to regenerate the training splits
5. Train again with more data!

## 💾 Using the Trained Model

After training, you'll have `coin_classifier_best.pth` - this is your trained model!

To use it for predictions:

```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50', num_labels=num_classes)
model.load_state_dict(torch.load('coin_classifier_best.pth'))
model.eval()

# Load and transform image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open('path/to/coin/image.jpg')
img_tensor = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    outputs = model(img_tensor).logits
    _, predicted = torch.max(outputs, 1)
    grade = classes[predicted.item()]
    print(f"Predicted grade: {grade}")
```

## 🆘 Troubleshooting

### "No module named 'transformers'"
```bash
pip install transformers torch torchvision
```

### "CUDA out of memory"
Decrease batch size in the notebook:
```python
BATCH_SIZE = 2  # or even 1
```

### "No images found"
Check that you have images in `images/<grade>/6000x3000/` folders

### "Not enough images for training"
Collect more coins using the bulk upload feature!

## 📚 Next Steps

1. **Collect more data** - More coins = better model
2. **Balance your dataset** - Similar number of images per grade
3. **Experiment with hyperparameters** - Try different settings
4. **Use data augmentation** - Helps with small datasets
5. **Try different models** - ResNet-18, ResNet-152, EfficientNet
6. **Build an inference API** - Deploy your model for real-time predictions

Happy training! 🪙🤖




