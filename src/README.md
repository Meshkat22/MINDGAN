# MINDGAN Package

This folder contains all the necessary files to run the MINDGAN experiments.

## Files Included

1. **MINDGAN_runner.py** - Main runner for BCI-IV 2A dataset (4-class)
2. **MINDGAN_2B.py** - Main runner for BCI-IV 2B dataset (2-class)
3. **utils.py** - Utility functions for data loading, metrics calculation, etc.

## Prerequisites

- Python 3.9+
- PyTorch 1.10+
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn
- scikit-learn

## Setup

1. **Install dependencies**:
   ```bash
   pip install torch torchvision numpy pandas scipy matplotlib seaborn scikit-learn
   ```

2. **Download datasets**:
   - BCI-IV 2A dataset: Place in `./Preprocessed_BCI_IV_2A_dataset/`
   - BCI-IV 2B dataset: Place in `./Preprocessed_BCI_IV_2B_dataset/`

## Usage

### For BCI-IV 2A dataset (4-class):
```bash
python MINDGAN_runner.py
```

### For BCI-IV 2B dataset (2-class):
```bash
python MINDGAN_2B.py
```

## Configuration

Edit the user config section at the top of each file to adjust:
- `DATA_DIR`: Path to dataset
- `QUICK_TEST`: Set to `True` for fast testing
- `TARGET_SUBS`: Specify which subjects to run
- `FULL_EPOCHS`: Number of training epochs
- And other hyperparameters

## Output

Results will be saved in a timestamped folder with:
- Training logs and figures
- Confusion matrices
- PSD analysis for real vs synthetic data
- Ablation study results
- Experiment summary

## Note

This package is self-contained and includes all necessary modules. You can copy this entire folder to another machine and run the experiments without additional dependencies (except for the required Python packages).