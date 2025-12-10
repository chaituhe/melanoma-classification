# Melanoma Classification – Image + Metadata Deep Learning Pipeline

This project builds a **clinical-grade melanoma detection model** by combining dermoscopic images with patient metadata.  
It demonstrates a full **end-to-end ML pipeline** including data exploration, preprocessing, metadata engineering, patient-grouped cross-validation, deep learning using EfficientNet, and image–metadata fusion.

This work aligns with real-world healthcare ML workflows where **patient safety, leakage prevention, reproducibility, and model robustness** are essential.

---

## 🔍 Project Objective

Build a binary classifier to predict melanoma using:

- High-resolution dermoscopy images  
- Patient features such as **age**, **sex**, **anatomical site**  
- Balanced training methods to address severe class imbalance  
- Patient-level grouped folds to prevent leakage  
- EfficientNet-based feature extractor with metadata fusion  

This pipeline is modeled after the **SIIM-ISIC Melanoma Challenge** dataset.

---

## 📁 Repository Structure

melanoma-classification/
│
├── eda/
│ └── eda.ipynb # Clean, structured EDA notebook (recommended for reviewers)
│
├── src/
│ ├── dataset.py # Dataset class + Albumentations image transforms
│ ├── model_meta.py # Metadata-only MLP model
│ ├── model_imgmeta.py # EfficientNet + metadata fusion architecture
│ ├── train_cv.py # 5-fold cross-validation training pipeline
│ └── utils.py # Helper functions (metrics, seeding, weight computation)
│
├── data/
│ ├── train.csv # Metadata file
│ └── jpeg/train # Dermoscopy images (image_name.jpg)
│
└── README.md # This document

markdown
Copy code

---

## 🧠 Key Pipeline Features

### **1. Reproducible Setup**
- Global seeding  
- Deterministic dataloaders  
- CUDA-aware training initialization  
- Automatic Mixed Precision (AMP) for performance  

### **2. Patient-Grouped 5-Fold Cross-Validation**
Ensures **no image from the same patient** leaks into both train and validation.

### **3. Metadata Engineering**
- One-hot encoding  
- Normalization of age features  
- Combined into **11-dimensional metadata vector**  
- Used both standalone (MLP) and as fusion input to CNN  

### **4. Data Augmentation (Albumentations 2.x)**
Used to improve robustness:

- RandomResizedCrop  
- Horizontal / vertical flips  
- Brightness/contrast  
- CLAHE  
- Coarse dropout  
- CenterCrop for validation  

Supports GPU-ready tensor output via `ToTensorV2`.

### **5. Deep Learning Models**

#### **📌 A) Metadata MLP Baseline**
Lightweight neural network for metadata-only modeling:

- 64 → 32 hidden layers  
- BatchNorm + Dropout  
- Trained with **class-balanced BCEWithLogitsLoss**  

#### **📌 B) EfficientNet + Metadata Fusion Model**
Deep architecture combining image and patient metadata:

- EfficientNet-B4 backbone (`tf_efficientnet_b4_ns`)  
- Global average pooling  
- Metadata processed through MLP  
- Concatenation of both embeddings  
- Fully connected fusion head for classification  
- Supports mixed precision (AMP)  

---

## 📊 Evaluation Metrics

Metrics computed per fold and overall out-of-fold predictions:

- **ROC-AUC** (primary)
- **Average Precision (PR-AUC)**  

Useful for imbalanced medical datasets.

---

## 🚀 How to Run the Pipeline

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
Core libraries:

torch / torchvision

timm

albumentations>=2.0

scikit-learn

opencv-python

numpy, pandas, matplotlib

2. Configure Data Paths
Update paths in notebook or scripts:

python
Copy code
CSV_PATH = "path/to/train.csv"
IMG_DIR  = "path/to/jpeg/train"
3. Run Cross-Validation Training
bash
Copy code
python src/train_cv.py
This will:

Generate 5 patient-grouped folds

Train metadata and fusion models per fold

Produce out-of-fold predictions

Print fold-wise and overall AUC/AP

📈 Sample Output (Example)
yaml
Copy code
=== Fold 0 ===
Meta-Only AUC: ~0.75
Img+Meta AUC: ~0.89

Overall OOF:
Meta-Only AUC: ~0.76 | AP: ~0.18
Img+Meta AUC: ~0.90 | AP: ~0.32
🧪 EDA Highlights (from eda2.ipynb)
Missing value analysis

Target imbalance visualization

Metadata distributions

Anatomical site frequency

Age histogram

Example dermoscopic images

Positive vs negative sampling

Identification of rare sites & edge-case samples

🎯 Summary
This repository demonstrates:

📌 End-to-end ML engineering for medical imaging

📌 Clean PyTorch architecture design

📌 Metadata fusion with CNNs

📌 Correct handling of patient-level leakage

📌 Advanced Albumentations augmentation

📌 Efficient training with AMP and AdamW

📌 Strong model baselines + reproducible validation

It is designed to showcase practical machine learning skills applicable to healthcare, medical imaging, and clinical decision-support systems.
