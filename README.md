# Neuroscience Data Analysis Problem Sets

This repository contains Python scripts for solving various problems in neuroscience data analysis, covering topics from basic Python operations to advanced signal processing and statistical testing.

## Overview

- **Problem Sets**: 5 Python scripts (`Problem_set_1.py` to `Problem_set_5.py`).
- **Additional Scripts**: 
  - `Bootstrap_permutation.py`: Demonstrates bootstrap and permutation tests for statistical analysis.
  - `Classification.py`: Implements logistic regression and SVM classifiers for binary classification tasks.
  - `Final_Project.py`: A comprehensive final project covering array manipulations, statistical tests, bootstrapping, and signal filtering.
- **Topics**: List/dictionary manipulations, NumPy array operations, Matplotlib visualizations, statistical testing, signal filtering, fMRI/EEG data analysis, regression models, and classification.
- **Collaborators**: Kevin Seth, Jared Lim, Raiyan Osman (credited per problem set).

## Dependencies

- Python 3.x
- Libraries:
  - NumPy
  - Matplotlib
  - SciPy
  - Pandas
  - Scikit-learn (for classification tasks)
  - IPython (for audio playback in Problem Set 5)
- Data files (not included):
  - `text_file.txt` (Problem Set 1)
  - `ps3_neural_data.npz`, `labels.txt` (Problem Set 3)
  - `ps4_dataset_2.npz`, `ps4_dataset_3.npz` (Problem Set 4)
  - `ps5_dataset_1.npz`, `ps5_EEG_data.npz`, `ps5_fmri_data.npz` (Problem Set 5)

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```
2. Install dependencies:
   ```bash
   pip install numpy matplotlib scipy pandas ipython scikit-learn
   ```

## Problem Sets Description

### Problem Set 1 (`Problem_set_1.py`)
- **Topics**: List/dictionary operations, matrix multiplication, word frequency analysis, neural data parsing.
- **Functions**: 
  - Basic list manipulations (elements, length, squares).
  - Word frequency counter, student records dictionary.
  - Dot product, matrix multiplication, neural label indexing.

### Problem Set 2 (`Problem_set_2.py`)
- **Topics**: NumPy array indexing, checkerboard pattern, statistical operations (subtract/negate values), matrix creation.
- **Key Functions**:
  - `checkerboard`, `subtract_from_between`, `row_index_matrix`, circular mask via Euclidean distance.

### Problem Set 3 (`Problem_set_3.py`)
- **Topics**: Matplotlib plotting, neurophysiology data analysis (tuning curves, PSTHs), binomial/Bonferroni correction.
- **Includes**:
  - Visualization of audio/neural data, peri-stimulus time histograms, multiple comparison corrections.

### Problem Set 4 (`Problem_set_4.py`)
- **Topics**: Statistical tests (t-test, bootstrap, permutation tests) for group comparisons.
- **Datasets**: Simulated group data (unpaired and paired designs).

### Problem Set 5 (`Problem_set_5.py`)
- **Topics**: Signal processing (audio/EEG filtering), fMRI data analysis with FIR models.
- **Features**:
  - Low-pass filter design, spectrogram/Power Spectral Density (PSD) plots.
  - Regression for fMRI voxel response prediction.

### Additional Scripts

#### `Bootstrap_permutation.py`
- **Topics**: Bootstrap and permutation tests for estimating standard errors and p-values.
- **Key Features**:
  - Bootstrap resampling for SEM estimation.
  - Permutation tests for hypothesis testing.

#### `Classification.py`
- **Topics**: Binary classification using logistic regression and SVM.
- **Key Features**:
  - Visualization of decision boundaries.
  - ROC analysis for classifier evaluation.

#### `Final_Project.py`
- **Topics**: Comprehensive problem set covering array manipulations, statistical tests, bootstrapping, and signal filtering.
- **Key Features**:
  - Array operations (correlation, reshaping, sorting).
  - Statistical tests (t-test, binomial test, bootstrapping).
  - Signal filtering (low-pass and high-pass filters).

---

## Collaborators
- **Sonu Yadav** (Primary Author)
- **Kevin Seth** (Problem Sets 1–5, Bootstrap_permutation.py, Classification.py, Final_Project.py)
- **Jared Lim** (Problem Sets 3–5, Final_Project.py)
- **Raiyan Osman** (Problem Set 3)

## Notes
- Data files referenced in the scripts are not included in this repository.
- Some results (e.g., audio output in Problem Set 5) require Jupyter/IPython environments.
- The `Final_Project.py` script serves as a comprehensive review of the concepts covered in the problem sets.
