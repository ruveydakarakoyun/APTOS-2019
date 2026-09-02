# APTOS 2019 Diabetic Retinopathy Grading

A PyTorch-based Deep Learning pipeline to predict Diabetic Retinopathy severity (ICDRSS grades 0-4) using the APTOS 2019 Blindness Detection dataset.

##  Project Overview
- **Dataset:** 3,662 retinal fundus images (Kaggle APTOS 2019).
- **Classes:** 5 ordinal severity levels (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR).
- **Primary Metric:** Quadratic Weighted Kappa (QWK) to properly account for class imbalance and ordinal error distance.
- **Infrastructure:** Trained via Google Colab (T4 GPU) directly pulling images from Google Cloud Storage (GCS).

##  Pipeline & Preprocessing
1. **Auto-Crop:** Removes non-informative black borders (~10.4% area reduction) around the retina.
2. **Albumentations Augmentation:** Random resized crop, horizontal/vertical flips, and mild spatial rotations.
3. **On-the-Fly Processing:** Dynamic batch loading directly from GCS (`aptos_train_images` / `aptos_test_images`) using custom PyTorch `Dataset` & `DataLoader`.

##  Model Architectures
- **Baseline:** Custom 3-Layer Convolutional Neural Network (CNN) as a benchmark.
- **Transfer Learning (Planned):** EfficientNet-B0 and ResNet50 fine-tuning using `timm`/`torchvision`.
- **Loss Function:** Weighted Cross-Entropy Loss to handle heavy class imbalance (9.3:1 ratio between Class 0 and Class 3).

##  Repository Structure
```text
APTOS-2019/
├── notebooks/
│   └── aptos_2019.ipynb   # Complete Colab end-to-end training notebook
├── tests/                 # Unit tests for preprocessing pipeline
├── .gitignore
├── README.md
└── requirements.txt
