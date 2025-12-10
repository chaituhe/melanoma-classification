# Melanoma Classification – Image + Metadata Deep Learning Pipeline

This project builds a **clinical-grade melanoma detection model** using dermoscopic images and patient metadata.  
It implements a **complete end-to-end pipeline** including EDA, preprocessing, patient-grouped cross-validation, metadata modeling, CNN feature extraction, and image–metadata fusion using EfficientNet.

---

## 🔍 Project Objective

To develop a **robust binary classifier** that predicts melanoma likelihood using:

- Dermoscopy images (`jpeg/train`)
- Patient metadata (`sex`, `age_approx`, `anatom_site_general_challenge`)
- Patient-level grouping to avoid leakage
- Balanced loss functions for heavy class imbalance

This pipeline follows the SIIM-ISIC Melanoma Classification dataset structure.

