# PCGS Coin Scraper & Classifier

A complete system to scrape PCGS coin data, organize images, and train AI models to classify coin grades.

## ⚡ Quick Start

**New to this project?** Check out [QUICKSTART.md](QUICKSTART.md) for a step-by-step guide!

**Key Commands:**
```bash
python app.py                      # Scrape coins
python check_dataset.py            # Check your data
python prepare_dataset.py          # Prepare for training
python coin_classifier_custom.py   # Train (rectangular images)
```

## 🌟 Features

### 📥 Web Scrapers with Beautiful UI

**PCGS Scraper** (Primary)
- **Single & Bulk Upload** - Add cert numbers individually or in bulk
- **Automatic Download** - Fetches coin details and images from PCGS API
- **Smart Organization** - Images organized by grade and resolution
- **Modern Tailwind UI** - Beautiful, responsive interface
- **Duplicate Detection** - Skips already downloaded certs

**David Lawrence Scraper** (New!)
- **Selenium-powered** - Handles JavaScript-rendered content
- **Full-size images** - Downloads 1000x1000 images via thumbnail clicks
- **Auto-discovery** - Finds all coins from auction/category URLs
- **Bulk scraping** - Process entire listings automatically
- **Detailed extraction** - Grade, cert #, series, eye appeal, toning, etc.

### 🤖 AI Training System
- **Dataset Preparation** - Automatic train/test/val split (70/20/10)
- **Deep Learning** - Fine-tune ResNet models for grade classification
- **TensorBoard Integration** - Monitor training in real-time
- **Pre-configured Notebook** - Ready-to-use Jupyter notebook

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Scrapers

**Option A: PCGS Scraper (Primary)**
```bash
python app.py
```
Open http://localhost:8000 and start adding cert numbers!

**Option B: David Lawrence Scraper (Alternative)**
```bash
# Single coin by inventory ID
python davidlawrence_selenium_scraper.py --inventory-id 789518

# Bulk scraping from listing URL (discovers all coins automatically)
python davidlawrence_selenium_bulk.py "https://davidlawrence.com/auctions?coinCategory=25&soldInPastAuction=true"

# Discovery only (find IDs without scraping)
python davidlawrence_selenium_bulk.py "https://davidlawrence.com/auctions?coinCategory=25" --discover-only --save-ids found.txt
```
See [SELENIUM_SETUP.md](SELENIUM_SETUP.md) for setup details.

### 3. Prepare Dataset for Training
```bash
python prepare_dataset.py
```
This organizes your images into proper training structure.

### 4. Train the Model

**Option A: Vision Transformer - Dual Image (⭐ RECOMMENDED for coins)**

```bash
python coin_classifier_vit.py
```

- **Uses BOTH obverse AND reverse** images
- **Full 1000x1000 resolution** (no downsampling)
- **Vision Transformer** architecture (state-of-the-art)
- **Preserves ALL coin details** for accurate grading
- Best for datasets with paired obverse/reverse images

**Option B: Custom CNN (Rectangular Images)**

```bash
# Python Script (Automated)
python coin_classifier_custom.py

# Jupyter Notebook (Interactive)
jupyter notebook coin_classifier_custom_cnn.ipynb
```

- Trains from scratch on rectangular (2:1) images
- Preserves full coin detail
- Best for 100+ images per class (single side)

**Option C: ResNet-50 (Faster - Square Images)**
```bash
jupyter notebook coin_classifier_full.ipynb
```
- Fine-tunes pre-trained model
- Converts to square images
- Best for < 50 images per class

See [TRAINING_COMPARISON.md](TRAINING_COMPARISON.md) for detailed comparison!

## 📁 Project Structure

```
coin_scrape/
├── app.py                             # Flask web application (PCGS)
├── pcgs_api.py                        # PCGS API client
├── davidlawrence_selenium_scraper.py  # David Lawrence single coin scraper (Selenium)
├── davidlawrence_selenium_bulk.py     # David Lawrence bulk scraper (Selenium)
├── prepare_dataset.py                 # Dataset preparation script
├── coin_classifier_custom.py       # Custom CNN training script (rectangular)
├── coin_classifier_custom_cnn.ipynb # Custom CNN notebook (rectangular)
├── coin_classifier_full.ipynb      # ResNet-50 training (square images)
├── CUSTOM_CNN_TRAINING.md          # Custom CNN guide
├── TRAINING_COMPARISON.md          # Compare training methods
├── TRAINING_GUIDE.md               # General training guide
├── templates/                      # Web UI templates
├── static/                         # JavaScript and assets
├── data/                          # Cert number database
│   └── cert_numbers.json
├── images/                        # Downloaded coin images
│   └── <grade>/
│       ├── obverse/              # Obverse images
│       ├── reverse/              # Reverse images  
│       └── 6000x3000/            # High-res (used for training)
└── coin_dataset/                 # Training dataset (after prepare_dataset.py)
    ├── train/
    ├── test/
    └── val/
```

## 💡 Usage Workflow

### Step 1: Collect Coins
1. Open the web UI at http://localhost:8000
2. Use **Bulk Upload** to paste lists of PCGS cert numbers
3. Images are automatically downloaded and organized by grade
4. Each grade becomes a classification class

### Step 2: Prepare Training Data
```bash
python prepare_dataset.py
```
- Scans `images/<grade>/6000x3000/` folders
- Creates train/test/validation splits
- Generates `coin_dataset/` folder

### Step 3: Train Model
```bash
jupyter notebook coin_classifier_full.ipynb
```
- Run all cells in the notebook
- Model trains to classify coin grades
- Best model saved as `coin_classifier_best.pth`

### Step 4: Monitor & Evaluate
```bash
tensorboard --logdir=runs
```
- View training metrics at http://localhost:6006
- Check accuracy and loss curves
- Evaluate on test set

## 📊 Image Organization

Images are organized in a hierarchical structure:

```
images/
  ms65/                    ← Grade = Class label
    obverse/              ← Not used for training
    reverse/              ← Not used for training
    6000x3000/            ← Used for training (highest resolution)
    5757x2905/            ← Stored but not used
    3000x1500/            ← Stored but not used
  au50/
    6000x3000/
  g04/
    6000x3000/
  ...
```

## 🎯 Training Requirements

- **Minimum**: 10 images per grade, 2+ grades
- **Recommended**: 50+ images per grade for good results
- **Best**: 100+ images per grade for excellent results

## 🔧 Configuration

### Web Scraper (`app.py`)
- **Port**: 8000 (change in `app.py`)
- **Image resolution**: Uses highest available (6000x3000 preferred)
- **API**: Official PCGS Public API with authentication

### Training (`coin_classifier_full.ipynb`)
- **Model**: ResNet-50 (pre-trained on ImageNet)
- **Image size**: 256x256 (adjustable)
- **Batch size**: 4 (adjust based on GPU memory)
- **Epochs**: 20 (increase for more training)
- **Learning rate**: 0.001 (tune if needed)

## 📚 Documentation

### Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - ⭐ Start here! Complete walkthrough
- **[DAVIDLAWRENCE_SCRAPER.md](DAVIDLAWRENCE_SCRAPER.md)** - David Lawrence scraper guide

### Training Guides
- **[TRAINING_COMPARISON.md](TRAINING_COMPARISON.md)** - Choose your training method
- **[CUSTOM_CNN_TRAINING.md](CUSTOM_CNN_TRAINING.md)** - Custom CNN guide (rectangular)
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - ResNet-50 guide (square)

### Scripts & Notebooks
- **[coin_classifier_custom.py](coin_classifier_custom.py)** - Custom CNN script (automated)
- **[coin_classifier_custom_cnn.ipynb](coin_classifier_custom_cnn.ipynb)** - Custom CNN notebook (interactive)
- **[coin_classifier_full.ipynb](coin_classifier_full.ipynb)** - ResNet-50 notebook
- **[prepare_dataset.py](prepare_dataset.py)** - Dataset splitter
- **[check_dataset.py](check_dataset.py)** - Dataset analyzer

## 🆘 Troubleshooting

### No images in dataset
- Make sure you've scraped coins using the web UI first
- Images must be in `images/<grade>/6000x3000/` folders

### CUDA out of memory
- Reduce `BATCH_SIZE` in the notebook (try 2 or 1)
- Or use CPU training (slower but works)

### Low accuracy
- Collect more images per grade (aim for 50+)
- Balance your dataset (similar counts per grade)
- Try data augmentation (uncomment in notebook)
- Train for more epochs

### Module not found
```bash
pip install transformers torch torchvision tensorboard jupyter matplotlib
```

## 🎓 What You Get

After training, you'll have:
1. **A trained model** (`coin_classifier_best.pth`) that can classify coin grades
2. **Training metrics** (accuracy, loss curves)
3. **Test results** showing model performance
4. **Organized dataset** ready for future experiments

## 🚀 Next Steps

1. **Collect more data** - Use bulk upload to add hundreds of certs
2. **Balance classes** - Aim for similar counts per grade
3. **Experiment** - Try different models, hyperparameters
4. **Deploy** - Build an API or web app for predictions
5. **Expand** - Add more features like price prediction, variety detection

## 📝 API Integration

Uses official PCGS Public API:
- **Base URL**: `https://api.pcgs.com/publicapi`
- **Authentication**: Bearer token (included)
- **Endpoints**:
  - `GetCoinFactsByCertNo/{certNo}` - Coin details
  - `GetImagesByCertNo?certNo={certNo}` - Coin images

Happy coin collecting and training! 🪙🤖

