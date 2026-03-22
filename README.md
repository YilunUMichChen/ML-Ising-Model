# Machine Learning Classification of Magnetic Phases in the Ising Model

## Overview
This repository contains the computational physics project analyzing magnetic phase transitions in the 2D Ising Model using various machine learning techniques. We generated spin configurations using Markov Chain Monte Carlo (MCMC) with the Metropolis-Hastings algorithm and applied deep learning and traditional machine learning models to classify the physical phases (Ferromagnetic vs. Paramagnetic).

## Key Features
- **Data Generation:** Simulated 2D Ising model spin configurations across different temperatures using multiprocessing for computational efficiency.
- **Deep Learning:** Implemented a Convolutional Neural Network (CNN) in `PyTorch` to capture spatial correlations of spin lattices.
- **Traditional Machine Learning:** Evaluated Support Vector Machine (SVM) and Random Forest (RF) classifiers using `scikit-learn`.
- **Dimensionality Reduction:** Investigated the effect of Principal Component Analysis (PCA) on the loss of crucial spatial features before neural network training.
- **Robustness Testing:** Systematically added noise (random spin flips) to evaluate and compare the robustness of CNNs, SVMs, and RFs under realistic experimental noise conditions.

## Repository Contents
- `ising_ml_classification.py` (formerly *Final edition.py*): The main Python script containing the Monte Carlo data generation, data preprocessing, and the implementation of CNN, SVM, and RF models.
- `2650 final report.pdf`: The comprehensive final academic report detailing the theoretical background, model architectures, and in-depth analysis of the results. 

## Tech Stack & Libraries
- **Language:** Python 3
- **Deep Learning:** PyTorch
- **Machine Learning & Data Processing:** Scikit-learn, NumPy
- **Visualization:** Matplotlib
- **Parallel Computing:** Python `multiprocessing` module

## Quick Start
To run the simulation and model training locally:
1. Ensure you have the required libraries installed (`torch`, `scikit-learn`, `numpy`, `matplotlib`).
2. Run the main script:
    ```bash
    python ising_ml_classification.py
    ```
*(Note: Data generation via Monte Carlo steps might take a few minutes depending on your CPU due to the multiprocessing pool).*

## Results & Discussion
For detailed mathematical theory, loss curves, confusion matrices, and the complete comparative analysis of how noise impacts different algorithms, please refer to the included **`2650 final report.pdf`**.

## Authors
- Yilun Chen
- Shuzhe Li

*This project was completed for the PHY 2650 Computational Physics and AI Tools course at CUHK-Shenzhen.*
