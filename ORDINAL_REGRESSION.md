# Ordinal Regression for Coin Grading

## 🎯 Why Use Regression Instead of Classification?

Coin grades are **inherently ordered**. The Sheldon scale (MS60, MS61, MS62, ..., MS70) represents a continuum, not independent categories.

### Problem with Classification

With classification, the model treats all errors equally:

```
True Grade: MS64

Prediction: MS63 → ❌ Wrong (1 grade off)
Prediction: MS54 → ❌ Wrong (10 grades off)

Both get the same penalty! 😱
```

This doesn't match reality. Being off by 1 grade is **much better** than being off by 10 grades.

### Solution: Ordinal Regression

Treat grades as continuous/ordered values:

```
True Grade: MS64 (normalized: 0.65)

Prediction: MS63 (0.64) → Small error: 0.01
Prediction: MS54 (0.54) → Large error: 0.11

Larger distances = larger penalties ✅
```

---

## 📊 Three Approaches Implemented

### 1. **MSE (Mean Squared Error)**
```python
loss = (prediction - target)²
```
- **Pros**: Standard regression, heavily penalizes outliers
- **Cons**: Can be too aggressive on large errors
- **Use**: When you want to strongly discourage wild predictions

### 2. **MAE (Mean Absolute Error)**
```python
loss = |prediction - target|
```
- **Pros**: More robust to outliers, easier to interpret
- **Cons**: Less penalty for moderate errors
- **Use**: When your data has some noisy/uncertain labels

### 3. **Ordinal Loss** (Recommended)
```python
loss = (prediction - target)²
```
- Same as MSE, but explicitly designed for ordered data
- **Pros**: Natural for ordinal scales, balanced penalty
- **Cons**: None really!
- **Use**: **Best choice for coin grading** 🏆

---

## 🔢 Evaluation Metrics

### MAE in Grade Numbers
```
Predictions: [64.2, 65.1, 63.8, 66.5]
True:        [64.0, 65.0, 64.0, 65.0]

MAE = (0.2 + 0.1 + 0.2 + 1.5) / 4 = 0.5 grades
```

**Interpretation**: On average, predictions are within **0.5 grades** of truth.

### Accuracy Within ±N Grades
```
±1 Accuracy: % of predictions within 1 grade
±2 Accuracy: % of predictions within 2 grades
```

Example:
```
True: MS64

Prediction: MS64 → ✓ Within ±1
Prediction: MS63 → ✓ Within ±1
Prediction: MS65 → ✓ Within ±1
Prediction: MS62 → ✓ Within ±2 (but not ±1)
Prediction: MS61 → ❌ Outside ±2
```

---

## 🆚 Classification vs Regression

| Metric | Classification | Regression |
|--------|---------------|------------|
| **Loss Function** | Cross-Entropy | MSE/MAE/Ordinal |
| **Output** | Discrete class | Continuous value |
| **Off by 1 grade** | Same penalty as off by 10 | Small penalty |
| **Off by 10 grades** | Same penalty as off by 1 | Large penalty |
| **Evaluation** | Accuracy (binary) | MAE (continuous) |
| **Interpretability** | "Correct" or "Wrong" | "Off by X grades" |
| **Generalization** | Can overfit to specific grades | Smooths across grade spectrum |

### Example Comparison

**True Grade: MS64**

#### Classification Approach:
```
Model outputs:
  MS63: 0.15 (15%)
  MS64: 0.45 (45%) ← Highest
  MS65: 0.30 (30%)
  MS66: 0.10 (10%)

Prediction: MS64 ✅ (Correct)

But if it had predicted MS63, it would be 100% wrong!
```

#### Regression Approach:
```
Model outputs: 64.3

Prediction: MS64 (after rounding)
Error: 0.3 grades

Even if we round to MS65, we're only off by 1 grade.
The model "knows" it's between MS64 and MS65.
```

---

## 🚀 Quick Start

### Train the Model

```bash
python coin_classifier_ordinal_regression.py
```

### Configuration Options

Edit these in the script:

```python
# Loss function
REGRESSION_TYPE = 'ordinal'  # 'mse', 'mae', or 'ordinal'

# Company conditioning
USE_COMPANY_CONDITIONING = True  # Include company as input feature

# Hyperparameters
IMAGE_SIZE = 448
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
```

### Expected Output

```
Epoch 30:
  Train: MAE=2.15 grades, ±1=45.3%, ±2=68.7%
  Val:   MAE=2.48 grades, ±1=42.1%, ±2=65.3%

Epoch 45:
  Train: MAE=1.82 grades, ±1=52.8%, ±2=75.4%
  Val:   MAE=2.21 grades, ±1=48.5%, ±2=71.2%
  ✓ New best! MAE: 2.21 grades
```

**Interpretation**:
- Model is typically off by **~2 grades**
- **48.5%** of predictions are exactly right or within 1 grade
- **71.2%** are within 2 grades

---

## 📈 Expected Performance

### Baseline (Classification)
```
Accuracy: 60%
Problem: Treats MS63 vs MS64 same as MS54 vs MS64
```

### Ordinal Regression
```
MAE: ~2-3 grades
±1 Accuracy: 45-55%
±2 Accuracy: 65-75%
±5 Accuracy: 85-95%

Better evaluation of "how wrong" predictions are!
```

---

## 🧠 Why This Works Better

### 1. **Smooth Learning**
Classification has hard boundaries between classes. Regression learns smooth transitions.

```
Classification:
MS63 | MS64 | MS65
  ❌  |  ✅  |  ❌
Hard boundaries

Regression:
63.0 ... 63.5 ... 64.0 ... 64.5 ... 65.0
         ↑ Model can output any value
```

### 2. **Better Gradients**
In classification, a wrong prediction gives the same gradient regardless of how wrong.

In regression, gradient magnitude ∝ distance from truth.

```
Classification:
True=MS64, Pred=MS63 → gradient magnitude: 1.0
True=MS64, Pred=MS54 → gradient magnitude: 1.0

Regression:
True=MS64, Pred=MS63 → gradient: 0.1 × (64-63) = 0.1
True=MS64, Pred=MS54 → gradient: 0.1 × (64-54) = 1.0
```

### 3. **Handles Uncertainty**
Some coins are borderline between grades. Regression can output "64.5" (between MS64 and MS65).

Classification must commit to one discrete class.

### 4. **Data Efficiency**
Regression uses the ordinal structure, so it can learn from fewer examples.

A coin graded MS64 provides information about both MS63 and MS65 (it's between them).

---

## 🔬 Advanced: Company-Conditioned Ordinal Regression

Enable this with `USE_COMPANY_CONDITIONING = True`.

The model learns:
- PCGS might consistently grade 0.5 points lower than NGC
- Company embeddings capture these systematic biases

At inference:
```python
# Predict for different companies
pcgs_grade = model(obverse, reverse, company='PCGS')  # → 64.2
ngc_grade = model(obverse, reverse, company='NGC')    # → 64.7
```

This combines the benefits of:
- **Ordinal regression** (smooth predictions, proper error penalties)
- **Company conditioning** (models company-specific biases)

---

## 📊 Visualization Example

After training, you can plot:

```python
import matplotlib.pyplot as plt

# Scatter plot: Predicted vs True
plt.scatter(true_grades, predicted_grades, alpha=0.5)
plt.plot([60, 70], [60, 70], 'r--')  # Perfect prediction line
plt.xlabel('True Grade')
plt.ylabel('Predicted Grade')
plt.title('Ordinal Regression Results')

# Points should cluster around the diagonal
# Distance from diagonal = prediction error
```

---

## 💡 Key Takeaways

1. ✅ **Use ordinal regression for coin grading** (it's inherently ordered data)
2. ✅ **Evaluate with MAE in grade numbers** (more meaningful than accuracy)
3. ✅ **Report ±1, ±2 accuracy** (shows practical usefulness)
4. ✅ **Consider company conditioning** (captures grading biases)
5. ✅ **Expect continuous predictions** (model outputs 64.3, not just MS64)

---

## 🎓 Further Reading

- **Ordinal Regression**: Treating rankings properly in ML
- **Label Smoothing**: Related technique for softening hard class boundaries
- **Ranking Loss Functions**: Other ways to penalize ordering violations

---

## 🚀 Next Steps

After training:

1. **Compare to baseline classification**:
   - Is MAE better than just looking at top-1 accuracy?
   - Do we get more nuanced predictions?

2. **Analyze error patterns**:
   - Which grades are hardest to predict?
   - Are we systematically over/under-predicting?

3. **Ensemble with classification**:
   - Use regression for "how good" (continuous)
   - Use classification for "which bucket" (discrete)
   - Average their predictions

4. **Per-company analysis**:
   - Does PCGS MAE differ from NGC MAE?
   - Can we learn systematic biases?

Good luck! 🎯



