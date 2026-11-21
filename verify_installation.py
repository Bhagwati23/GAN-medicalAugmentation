"""
Verification script to test basic functionality.
Run this script to verify that the dataset module is properly set up.
"""

import sys

print("=" * 70)
print("Verifying Installation and Basic Functionality")
print("=" * 70)

# Test 1: Check Python version
print(f"\n1. Python Version: {sys.version}")
print("   [OK] Python is working")

# Test 2: Import numpy
try:
    import numpy as np
    print(f"\n2. NumPy Version: {np.__version__}")
    print("   [OK] NumPy is installed")
except ImportError as e:
    print(f"\n2. [FAIL] NumPy not installed: {e}")
    sys.exit(1)

# Test 3: Import medmnist
try:
    import medmnist
    print(f"\n3. MedMNIST Version: {medmnist.__version__}")
    print("   [OK] MedMNIST is installed")
except ImportError as e:
    print(f"\n3. [FAIL] MedMNIST not installed: {e}")
    print("   Please install: pip install medmnist")
    sys.exit(1)

# Test 4: Import torch
try:
    import torch
    print(f"\n4. PyTorch Version: {torch.__version__}")
    print("   [OK] PyTorch is installed")
except ImportError as e:
    print(f"\n4. [FAIL] PyTorch not installed: {e}")
    print("   Please install: pip install torch torchvision")
    sys.exit(1)

# Test 5: Import our dataset module
try:
    from src.data.dataset import get_medmnist_info
    print("\n5. Dataset Module")
    print("   [OK] src.data.dataset module loads successfully")

    # Test getting dataset info
    info = get_medmnist_info('pneumoniamnist')
    print(f"   [OK] PneumoniaMNIST info retrieved: {info['task']}")

except Exception as e:
    print(f"\n5. [FAIL] Dataset module error: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("[OK] ALL VERIFICATION CHECKS PASSED!")
print("=" * 70)
print("\nYou can now:")
print("1. Run the exploration notebook: jupyter notebook notebooks/01_dataset_exploration.ipynb")
print("2. Run tests: pytest tests/ -v")
print("3. Use the dataset module in your code")
print("=" * 70)
