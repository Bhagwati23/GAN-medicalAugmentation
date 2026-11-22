# GAN-medicalAugmentation

Evaluating the impact of GAN-generated synthetic medical images on deep learning classification performance.

## Project Overview

This project explores the use of Generative Adversarial Networks (GANs) to generate synthetic medical images for data augmentation. We investigate whether GAN-generated synthetic images can improve the performance of deep learning classifiers on medical imaging tasks.

### Research Question

Can GAN-generated synthetic medical images effectively augment training data and improve classification performance compared to traditional augmentation techniques?

## Project Status

**Issue #1 - Dataset Selection, Preparation and Exploration** - COMPLETED

- Explored BreastMNIST and PneumoniaMNIST from MedMNIST v2
- Analyzed dataset characteristics, class distributions, and image properties
- Selected **PneumoniaMNIST** as the primary dataset for GAN experiments
- Established strict data leakage prevention protocols
- All 38 unit tests passing on Python 3.12

## Dataset: PneumoniaMNIST

We selected **PneumoniaMNIST** from the MedMNIST v2 collection for this project.

### Dataset Characteristics

- **Total Samples**: 5,856 images
  - Training: 4,708 images
  - Validation: 524 images
  - Test: 624 images
- **Task**: Binary classification (Normal vs Pneumonia)
- **Image Size**: 28 × 28 pixels
- **Channels**: 1 (grayscale)
- **Modality**: Chest X-ray images
- **Pixel Range**: [0, 255]

### Why PneumoniaMNIST?

PneumoniaMNIST was selected primarily for practical reasons:

1. **Larger Dataset Size**: 5,856 samples vs 780 in BreastMNIST (7.5× more data)
2. **More Training Samples**: 4,708 training images enable better GAN training
3. **Project Requirements**: Sufficient data for GAN experiments and evaluation
4. **Binary Classification**: Same task complexity as BreastMNIST (2 classes)
5. **Established Baselines**: Well-documented dataset with existing literature

**Important Caveats**:
- Clinical suitability for GAN augmentation is not established prior to experiments
- Class balance and visual pattern quality will be evaluated during the project
- Dataset selection is based on size and availability, not proven GAN performance

### Dataset Comparison

| Metric | BreastMNIST | PneumoniaMNIST |
|--------|-------------|----------------|
| Total Samples | 780 | 5,856 |
| Train Samples | 546 | 4,708 |
| Val Samples | 78 | 524 |
| Test Samples | 156 | 624 |
| Number of Classes | 2 | 2 |
| Modality | Breast Ultrasound | Chest X-Ray |
| Task | Benign vs Malignant | Normal vs Pneumonia |

## Project Structure

```
GAN-medicalAugmentation/
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb    # Dataset analysis and exploration
│
├── src/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       └── dataset.py                  # Dataset loading utilities
│
├── tests/
│   ├── __init__.py
│   └── test_dataset.py                 # Unit tests for dataset module
│
├── data/
│   └── .gitkeep                        # Downloaded datasets (not committed)
│
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── .gitignore                          # Git ignore rules
```

## Installation

### Prerequisites

- Python 3.12 (tested and recommended)
- pip (Python package manager)

**Note**: This project has been developed and tested exclusively with Python 3.12.10 on Windows. Other Python versions are not verified.

### Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/Bhagwati23/GAN-medicalAugmentation.git
cd GAN-medicalAugmentation
```

2. **Create a virtual environment (Python 3.12)**

```powershell
# Windows PowerShell (tested environment)
python -m venv .venv312
```

3. **Install dependencies (no activation required)**

```powershell
# Windows PowerShell - Direct executable path
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

4. **Verify installation**

```powershell
.\.venv312\Scripts\python.exe verify_installation.py
```

**Alternative (with activation)**:

```powershell
# Activate virtual environment
.\.venv312\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python verify_installation.py
```

### Required Packages

- **Data Science**: numpy, pandas, matplotlib, seaborn, scikit-learn
- **Medical Imaging**: medmnist (v2.2.0+)
- **Deep Learning**: torch, torchvision (CPU version)
- **Development**: jupyter, ipykernel, pytest

See `requirements.txt` for complete list with version specifications.

## Usage

### Dataset Loading

```python
from src.data.dataset import (
    load_pneumonia_mnist,
    load_full_dataset,
    get_training_data_only,
    get_dataset_statistics
)

# Load a specific split
train_images, train_labels = load_pneumonia_mnist(split='train', download=True)

# Load all splits at once
all_data = load_full_dataset('pneumoniamnist', download=True)
train_images, train_labels = all_data['train']
val_images, val_labels = all_data['val']
test_images, test_labels = all_data['test']

# Get ONLY training data (for GAN training - excludes test data)
gan_train_images, gan_train_labels = get_training_data_only(
    'pneumoniamnist',
    download=True,
    include_val=False  # Set to True to include validation data
)

# Get dataset statistics
stats = get_dataset_statistics(train_images, train_labels)
print(f"Number of samples: {stats['num_samples']}")
print(f"Class distribution: {stats['class_distribution']}")
```

### Running the Exploration Notebook

**Note**: Notebook execution has not been verified in the current environment.

1. **Start Jupyter**

```powershell
.\.venv312\Scripts\jupyter.exe notebook
```

2. **Open the exploration notebook**

Navigate to `notebooks/01_dataset_exploration.ipynb`

3. **Expected notebook contents**

The notebook is designed to:
- Download both BreastMNIST and PneumoniaMNIST datasets
- Display comprehensive statistics and visualizations
- Compare both datasets
- Demonstrate reproducibility
- Verify test-set isolation

### Running Tests

```powershell
# Windows PowerShell (tested environment)
# Run all tests
.\.venv312\Scripts\python.exe -m pytest tests/ -v

# Run specific test file
.\.venv312\Scripts\python.exe -m pytest tests/test_dataset.py -v

# Run with coverage report (if pytest-cov is installed)
.\.venv312\Scripts\python.exe -m pytest tests/ --cov=src --cov-report=html
```

All 38 tests pass on Python 3.12.10.

## Data Leakage Prevention

**CRITICAL**: This project implements strict protocols to prevent data leakage:

### Test Data Isolation

- Test data is ONLY used for final model evaluation
- Test data is NEVER used for:
  - GAN training
  - Synthetic image generation
  - Model selection
  - Hyperparameter tuning
  - Training set augmentation

### Official Split Usage

We preserve MedMNIST's official train/validation/test splits:

- **Training Split**: Used for GAN training and generating synthetic images
- **Validation Split**: Used for model selection and hyperparameter tuning
- **Test Split**: Reserved exclusively for final performance evaluation

### Split Separation Verification

Our verification checks use SHA-256 hashing to detect pixel-identical images across splits:

- Zero train-test overlaps detected
- 8 pixel-identical train-val images found (0.17% of training data)

**Important Limitation**: SHA-256 pixel-level duplicate checking can only detect identical images. It cannot verify patient-level separation (i.e., whether different images come from the same patient). MedMNIST documentation should be consulted for patient-level de-identification guarantees.

### Code Safeguards

The `get_training_data_only()` function explicitly excludes test data:

```python
# This function ensures test data is never accidentally included
training_data = get_training_data_only('pneumoniamnist', include_val=False)
```

## Reproducibility

### Random Seeds

All experiments use a fixed random seed for reproducibility:

```python
from src.data.dataset import set_random_seed

RANDOM_SEED = 42
set_random_seed(RANDOM_SEED)
```

### Dataset Versions

- **MedMNIST Version**: 2.2.0+
- **Dataset Version**: MedMNIST v2 (official release)

### Verification

Run the reproducibility demonstration:

```python
from src.data.dataset import demonstrate_reproducibility

result = demonstrate_reproducibility('pneumoniamnist', seed=42, n_samples=5)
print(f"Reproducible: {result['identical']}")  # Should print True
```

## Next Steps

1. **Baseline Classifier** (Issue #2)
   - Train a CNN classifier on original PneumoniaMNIST data
   - Establish baseline performance metrics
   - Analyze classification errors

2. **GAN Implementation** (Issue #3)
   - Implement DCGAN or similar architecture
   - Train GAN on PneumoniaMNIST training data only
   - Generate synthetic pneumonia images

3. **Augmented Training** (Issue #4)
   - Combine real and synthetic training images
   - Train classifier on augmented dataset
   - Compare with baseline performance

4. **Evaluation & Analysis** (Issue #5)
   - Evaluate on held-out test set
   - Compare baseline vs augmented model
   - Analyze synthetic image quality
   - Document findings and conclusions

## Contributing

This is a research project. Contributions are welcome through:

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feat/new-feature`)
5. Open a Pull Request

## License

This project is for educational and research purposes.

## References

- **MedMNIST v2**: Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., ... & Ni, B. (2023). MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data, 10(1), 41.
- **MedMNIST Paper**: https://doi.org/10.1038/s41597-022-01721-8
- **MedMNIST Website**: https://medmnist.com/

## Contact

For questions or issues, please open a GitHub issue in this repository.

---

**Last Updated**: November 21, 2026
**Status**: Dataset exploration completed, ready for baseline classifier implementation
