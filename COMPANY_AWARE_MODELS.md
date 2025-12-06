# Company-Aware Coin Grading Models

Two sophisticated approaches to leverage grading company information (PCGS, NGC, ANACS, etc.) to improve grade predictions.

## 🎯 Why This Matters

Different grading companies have **different grading standards and tendencies**:
- PCGS might be stricter on certain defects
- NGC might weight surface quality differently
- Some companies are known to be more generous with certain coin types

By incorporating this knowledge, we can:
1. **Improve accuracy**: Learn company-specific biases
2. **Better generalization**: Understand what makes a "PCGS MS65" vs "NGC MS65"
3. **Flexible inference**: Predict "What would PCGS call this?" vs "What would NGC call this?"

---

## 📊 Approach A: Multi-Task Learning

**File**: `coin_classifier_multitask.py`

### How It Works

```
Input: Coin Images (Obverse + Reverse)
         ↓
   ResNet Backbone (Shared)
         ↓
    Shared Features
       ↙     ↘
   Head 1    Head 2
   Grade     Company
```

### Key Idea

Train with **two objectives simultaneously**:
1. **Main task**: Predict coin grade (MS64, MS65, etc.)
2. **Auxiliary task**: Predict grading company (PCGS, NGC, etc.)

**Combined Loss**:
```python
loss = 1.0 * grade_loss + 0.3 * company_loss
```

### Benefits

✅ Forces model to learn features useful for **both** tasks  
✅ Company prediction acts as **regularization**  
✅ Model implicitly learns company biases  
✅ Simple to implement and train  

### Usage

```bash
# Train
python coin_classifier_multitask.py

# The model will output:
# - Grade predictions: MS64 (45%), MS65 (30%), ...
# - Company predictions: PCGS (60%), NGC (35%), ...
```

### Training Output Example

```
Epoch 30:
  Train: Grade=68.3%, Company=85.2%
  Val:   Grade=65.1%, Company=82.7%
```

Notice the company prediction is easier (higher accuracy), which helps guide the harder grade prediction task!

---

## 🎚️ Approach B: Company-Conditioned Model

**File**: `coin_classifier_company_conditioned.py`

### How It Works

```
Input: [Coin Images, Company ID]
              ↓
    ResNet Backbone (Shared)
              ↓
       Image Features
              ↓
    [Features + Company Embedding]
              ↓
         Grade Head
              ↓
    "What would PCGS call this?"
```

### Key Idea

**At training**: Learn "Given these coin features + company style → grade"

**At inference**: You control which company to mimic!

```python
# Same coin, different companies
pcgs_prediction = model.predict_with_company(obv, rev, 'PCGS', company_to_idx)
ngc_prediction  = model.predict_with_company(obv, rev, 'NGC', company_to_idx)
anacs_prediction = model.predict_with_company(obv, rev, 'ANACS', company_to_idx)
```

### Benefits

✅ **Explicit control** at inference time  
✅ Can compare "What would different companies call this?"  
✅ Model learns **company embeddings** (their grading tendencies)  
✅ More interpretable: company is an explicit input  

### Usage

```bash
# Train
python coin_classifier_company_conditioned.py

# Demo
python demo_company_aware_models.py conditioned obverse.jpg reverse.jpg
```

### Inference Example

```
🏢 PCGS:
   Top Prediction: MS64 (55.3%)
   Top 3:
     • MS64     - 55.3%
     • MS63     - 25.1%
     • MS65     - 15.8%

🏢 NGC:
   Top Prediction: MS65 (48.7%)
   Top 3:
     • MS65     - 48.7%
     • MS64     - 35.2%
     • MS66     - 10.1%
```

See how NGC might call it MS65 while PCGS calls it MS64! This captures real-world grading differences.

---

## 🆚 Comparison

| Feature | Multi-Task (A) | Company-Conditioned (B) |
|---------|---------------|------------------------|
| **Training** | Two loss functions | Single loss function |
| **Company Info** | Auxiliary task | Input feature |
| **Inference** | Predicts company automatically | You specify company |
| **Flexibility** | Less flexible | More flexible |
| **Interpretability** | Implicit biases learned | Explicit company embeddings |
| **Use Case** | "What is this coin + who graded it?" | "What would X company call this?" |

---

## 🚀 Quick Start

### 1. Train Both Models

```bash
# Approach A: Multi-Task
python coin_classifier_multitask.py

# Approach B: Company-Conditioned
python coin_classifier_company_conditioned.py
```

Both will create models in `models/`:
- `models/coin_multitask_best.pth`
- `models/coin_company_conditioned_best.pth`

### 2. Demo Inference

```bash
# Demo multi-task model
python demo_company_aware_models.py multitask \
    davidlawrence_dataset/Proof/ms64/obverse/ms64-PCGS-1908-5d-38516252.jpg \
    davidlawrence_dataset/Proof/ms64/reverse/ms64-PCGS-1908-5d-38516252.jpg

# Demo company-conditioned model
python demo_company_aware_models.py conditioned \
    davidlawrence_dataset/Proof/ms64/obverse/ms64-PCGS-1908-5d-38516252.jpg \
    davidlawrence_dataset/Proof/ms64/reverse/ms64-PCGS-1908-5d-38516252.jpg

# Compare both approaches
python demo_company_aware_models.py compare \
    davidlawrence_dataset/Proof/ms64/obverse/ms64-PCGS-1908-5d-38516252.jpg \
    davidlawrence_dataset/Proof/ms64/reverse/ms64-PCGS-1908-5d-38516252.jpg
```

---

## 🧠 Advanced: How Company Embeddings Work

In Approach B, each company gets a **learned embedding vector**:

```python
# During training
company_embedding = nn.Embedding(num_companies, embedding_dim=32)

# PCGS might learn: [0.2, -0.5, 0.8, ..., 0.1]  (32 dims)
# NGC might learn:  [0.1, -0.3, 0.6, ..., -0.2]
# ANACS might learn: [-0.1, 0.4, 0.3, ..., 0.5]
```

These embeddings capture:
- How strict the company is overall
- Which defect types they penalize more
- Their general grading philosophy

### Visualizing Company Embeddings

After training, you could:
1. Extract the embeddings for each company
2. Use t-SNE or PCA to visualize in 2D
3. See which companies have similar grading styles!

```python
# Extract embeddings
embeddings = model.company_embedding.weight.data
# embeddings.shape = [num_companies, 32]

# Companies with similar embeddings have similar grading tendencies
```

---

## 📈 Expected Performance Gains

Based on similar multi-task learning approaches:

| Metric | Baseline | Multi-Task (A) | Conditioned (B) |
|--------|----------|----------------|-----------------|
| **Overall Accuracy** | 60% | **63-65%** 🎯 | **64-66%** 🎯 |
| **Interpretability** | Low | Medium | **High** ✨ |
| **Company-Specific Accuracy** | N/A | N/A | **70-75%** 🚀 |

The company-conditioned model tends to perform best when you know the target company at inference time.

---

## 🔬 Experimental Ideas

### 1. **Ensemble Both Approaches**
```python
pred_a = multitask_model(obv, rev)[0]  # Grade output
pred_b = conditioned_model(obv, rev, company)
final_pred = 0.5 * pred_a + 0.5 * pred_b
```

### 2. **Company Bias Analysis**
Track prediction differences:
```python
for coin in test_set:
    pcgs_grade = model_b.predict_with_company(..., 'PCGS')
    ngc_grade = model_b.predict_with_company(..., 'NGC')
    
    if pcgs_grade != ngc_grade:
        print(f"Disagreement: PCGS={pcgs_grade}, NGC={ngc_grade}")
```

### 3. **Hard Mining by Company**
Some companies might be harder to predict. Use company accuracy to weight training samples.

---

## 🎓 References

This implementation is inspired by:
- Multi-task learning in computer vision (Caruana, 1997)
- Conditional image generation (cGAN, Mirza & Osindero, 2014)
- Meta-learning with company as task context

---

## 📊 Monitoring Training

Both scripts log to TensorBoard:

```bash
# View training progress
tensorboard --logdir runs/

# You'll see:
# - Approach A: Grade loss, Company loss, Grade acc, Company acc
# - Approach B: Loss, Grade acc (conditioned on company)
```

---

## ❓ FAQ

**Q: Which approach should I use?**  
A: 
- Use **Multi-Task (A)** if you want a single model that does everything
- Use **Company-Conditioned (B)** if you want to explicitly control which company to mimic at inference

**Q: Can I combine them?**  
A: Yes! Train both and ensemble their predictions.

**Q: What if I don't know the company at inference?**  
A: Use Multi-Task (A), which predicts both grade and company. Or use Model B with the most common company (e.g., PCGS).

**Q: How much data do I need?**  
A: For best results, have at least 50-100 examples per company. Rare companies might not learn good embeddings.

---

## 🎉 Summary

You now have **two powerful ways** to leverage grading company information:

1. **Multi-Task**: Learn grade + company simultaneously, implicit biases
2. **Company-Conditioned**: Explicit company input, flexible inference

Both approaches should **outperform** a baseline model that ignores company information, because they capture real grading differences between PCGS, NGC, and others!

**Try both, compare results, and use what works best for your use case!** 🚀



