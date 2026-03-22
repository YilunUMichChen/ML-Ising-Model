"""
=============================================================================
Project: 2D Ising Model Phase Transition Classification via Machine Learning
Author: Alan Chen, Shuzhe Li
Description:
    This script simulates the 2D Ising model using the Monte Carlo 
    (Metropolis-Hastings) algorithm. It generates lattice spin configurations 
    at different temperatures (ordered phase T < Tc, disordered phase T > Tc).
    
    The generated data is then evaluated using multiple machine learning 
    architectures to benchmark their classification performance and robustness:
        1. Convolutional Neural Networks (CNN) using PyTorch
        2. Principal Component Analysis (PCA) combined with Fully Connected NNs
        3. Traditional ML models: Support Vector Machines (SVM) & Random Forests (RF)
    
    Finally, the script injects artificial spatial noise to analyze the 
    robustness of both raw-feature models and PCA-reduced models.
=============================================================================
"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import multiprocessing as mp   # <— Added for parallel data generation


def monte_carlo_ising(size=32, temp=2.0, steps=10000):
    """
    Simulates a 2D Ising model using the Metropolis-Hastings algorithm.
    
    Args:
        size (int): The dimension of the 2D square lattice (size x size).
        temp (float): The simulated temperature. Critical temp Tc is ~2.269.
        steps (int): Number of Monte Carlo steps to reach thermal equilibrium.
        
    Returns:
        numpy.ndarray: A 2D array of spins (-1 or 1) representing the state.
    """
    # Initialize a random spin configuration
    spins = np.random.choice([-1, 1], size=(size, size))
    
    # Perform Monte Carlo updates
    for _ in range(steps):
        i, j = np.random.randint(0, size, 2)
        
        # Calculate the sum of nearest neighbors (with periodic boundary conditions)
        neighbors = (
            spins[(i+1)%size, j] + 
            spins[(i-1)%size, j] + 
            spins[i, (j+1)%size] + 
            spins[i, (j-1)%size]
        )
        
        # Calculate energy difference if we flip the spin at (i, j)
        energy_diff = 2 * spins[i, j] * neighbors
        
        # Metropolis acceptance criterion
        if energy_diff <= 0 or (np.random.random() < np.exp(-energy_diff/temp)):
            spins[i, j] *= -1
            
    return spins


def _gen_single_sample(args):
    """
    Pool worker helper function: Generates a single Ising configuration.
    
    Args:
        args (tuple): A tuple containing the temperature value.
        
    Returns:
        numpy.ndarray: A single simulated Ising lattice.
    """
    temp = args[0]
    return monte_carlo_ising(temp=temp)


def generate_data(n_samples, temp, label, description, n_workers: int | None = None):
    """
    Generates a dataset of Ising model configurations, supporting parallel execution.
    
    Args:
        n_samples (int): Number of lattice samples to generate.
        temp (float): Temperature parameter for the Monte Carlo simulation.
        label (int): Ground truth label (e.g., 0 for disordered, 1 for ordered).
        description (str): String identifier for logging purposes.
        n_workers (int, optional): Number of CPU cores to use. 
                                   None/1 uses single process.
                                   >=2 uses multiprocessing.Pool.
                                   
    Returns:
        tuple: (X_data, y_labels) as numpy arrays.
    """
    start_time = time.time()

    # ==========================================
    # Parallel Processing Path
    # ==========================================
    if n_workers and n_workers > 1:
        results = []
        with mp.Pool(processes=n_workers) as pool:
            # imap_unordered produces and consumes simultaneously, saving memory
            for i, spins in enumerate(
                pool.imap_unordered(_gen_single_sample, [(temp,)] * n_samples)
            ):
                results.append(spins)
                if i % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Generating {description} data "
                          f"{i}/{n_samples}, elapsed {elapsed:.2f}s")
        X_data = np.array(results)
        
    # ==========================================
    # Single Process Path
    # ==========================================
    else:
        X_data = []
        for i in range(n_samples):
            if i % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Generating {description} data "
                      f"{i}/{n_samples}, elapsed {elapsed:.2f}s")
            X_data.append(monte_carlo_ising(temp=temp))
        X_data = np.array(X_data)

    print(f"{description} data generation completed, "
          f"Total time: {time.time() - start_time:.2f} seconds")
          
    return X_data, np.full(n_samples, label)


def main():
    """
    Main execution pipeline:
    1. Data Generation
    2. Data Preprocessing & Splitting
    3. CNN Training & Evaluation
    4. PCA + Fully Connected NN Training & Evaluation
    5. Traditional ML (SVM, RF) Tuning & Evaluation
    6. Spatial Noise Robustness Testing across all models
    """
    
    # =========================================================================
    # Section 1: Data Generation & Preprocessing
    # =========================================================================
    n_samples = 300
    T_critical = 2.27  # Theoretical critical temperature for 2D Ising Model
    
    # Generate high-temperature (disordered/paramagnetic) phase data
    X_disordered, y_disordered = generate_data(
        n_samples, 
        temp=3.0, 
        label=0,
        description="Disordered phase",
        n_workers=mp.cpu_count()
    )
    
    # Generate low-temperature (ordered/ferromagnetic) phase data
    X_ordered, y_ordered = generate_data(
        n_samples, 
        temp=1.5, 
        label=1,
        description="Ordered phase",
        n_workers=mp.cpu_count()
    )
    
    # Combine datasets
    X = np.concatenate([X_disordered, X_ordered])
    y = np.concatenate([y_disordered, y_ordered])
    
    # Normalize spin values from {-1, 1} to {0, 1} for better NN convergence
    X = (X + 1) / 2  
    # Add channel dimension for PyTorch Conv2D: (Batch, Channel, Height, Width)
    X = X[:, np.newaxis, :, :]

    # Train/Test Split with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.25, 
        random_state=0, 
        stratify=y
    )

    # Convert numpy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create PyTorch DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # =========================================================================
    # Section 2: Convolutional Neural Network (CNN) Approach
    # =========================================================================
    class ProjectCNN(nn.Module):
        """
        A simple Convolutional Neural Network designed to extract spatial 
        correlation features from the 2D Ising lattice for phase classification.
        """
        def __init__(self):
            super(ProjectCNN, self).__init__()
            # Feature extraction layer
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2)
            self.pool = nn.MaxPool2d(kernel_size=2)
            # Classification layers
            self.fc1 = nn.Linear(64 * 15 * 15, 64)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(64, 2)

        def forward(self, x):
            # Forward pass through CNN
            x = self.pool(F.relu(self.conv1(x)))
            x = x.view(-1, 64 * 15 * 15)  # Flatten spatial dimensions
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    # Initialize CNN model, loss function, and optimizer
    model = ProjectCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop for CNN
    num_epochs = 10
    print("\n--- Starting CNN Training ---")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Loss: {running_loss/len(train_loader):.4f}")

    # Evaluation loop for CNN
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = 100 * correct / total
    print(f"\nTest Accuracy (CNN): {accuracy:.2f}%")

    # Plot Confusion Matrix for CNN
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure()
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", 
        xticklabels=["Disordered", "Ordered"], 
        yticklabels=["Disordered", "Ordered"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (CNN)")
    plt.show()

    # =========================================================================
    # Section 3: PCA Dimensionality Reduction + Fully Connected NN
    # =========================================================================
    # Flatten spatial data for PCA (since PCA expects 1D feature vectors)
    X_train_flat = X_train.view(X_train.size(0), -1).numpy()
    X_test_flat = X_test.view(X_test.size(0), -1).numpy()

    # Apply PCA to extract dominant features
    n_components = 100
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)

    # Convert PCA outputs back to PyTorch tensors
    X_train_pca_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
    X_test_pca_tensor = torch.tensor(X_test_pca, dtype=torch.float32)

    # Create new DataLoaders for PCA-reduced data
    train_loader_pca = DataLoader(
        TensorDataset(X_train_pca_tensor, y_train), 
        batch_size=16, shuffle=True
    )
    test_loader_pca = DataLoader(
        TensorDataset(X_test_pca_tensor, y_test), 
        batch_size=16, shuffle=False
    )

    class ImprovedNN(nn.Module):
        """
        A deep Fully Connected Neural Network tailored to process 
        the dense representations generated by PCA.
        """
        def __init__(self, input_dim=n_components, dropout_rate=0.3):
            super(ImprovedNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 256)
            self.bn1 = nn.BatchNorm1d(256)
            self.fc2 = nn.Linear(256, 128)
            self.bn2 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128, 64)
            self.bn3 = nn.BatchNorm1d(64)
            self.fc4 = nn.Linear(64, 2)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = F.relu(self.bn3(self.fc3(x)))
            x = self.dropout(x)
            return self.fc4(x)

    # Initialize PCA+NN model, optimizer, and learning rate scheduler
    model_pca = ImprovedNN().to(device)
    optimizer_pca = optim.Adam(model_pca.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_pca, mode='min', patience=3
    )
    
    # Training configuration for PCA+NN
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    print("\n--- Starting PCA + NN Training ---")
    for epoch in range(30):
        model_pca.train()
        running_loss = 0.0
        for inputs, labels in train_loader_pca:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_pca.zero_grad()
            outputs = model_pca(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_pca.step()
            running_loss += loss.item()

        # Validation phase for early stopping
        val_loss = 0.0
        model_pca.eval()
        with torch.no_grad():
            for inputs, labels in test_loader_pca:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model_pca(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        val_loss /= len(test_loader_pca)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}, "
              f"Train Loss: {running_loss/len(train_loader_pca):.4f}, "
              f"Val Loss: {val_loss:.4f}")

        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model_pca.state_dict(), 'best_model_pca.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered. Reverting to best model.")
                break

    # Evaluation on PCA+NN
    model_pca.eval()
    all_preds_pca, all_labels_pca = [], []
    with torch.no_grad():
        for inputs, labels in test_loader_pca:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_pca(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds_pca.extend(predicted.cpu().numpy())
            all_labels_pca.extend(labels.cpu().numpy())
            
    accuracy_pca = accuracy_score(all_labels_pca, all_preds_pca)
    print(f"\nTest Accuracy (PCA + NN): {accuracy_pca*100:.2f}%")

    # Plot Confusion Matrix for PCA+NN
    cm_pca_nn = confusion_matrix(all_labels_pca, all_preds_pca)
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm_pca_nn, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Disordered", "Ordered"],
        yticklabels=["Disordered", "Ordered"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (PCA + NN)")
    plt.tight_layout()
    plt.show()  

    # =========================================================================
    # Section 4: Traditional ML Baselines (SVM & Random Forest)
    # =========================================================================
    print("\n--- Training Traditional ML Baselines ---")
    
    # Fit PCA for Scikit-learn models, preserving 95% variance
    pca_sklearn = PCA(n_components=0.95)
    X_train_pca_sklearn = pca_sklearn.fit_transform(X_train_flat)
    X_test_pca_sklearn = pca_sklearn.transform(X_test_flat)

    # Standardize data for Support Vector Machine
    scaler = StandardScaler().fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # --- SVM Grid Search Optimization ---
    param_svm = {
        "C":      [0.5, 2, 3, 6, 7, 10],
        "gamma":  [0.001, 0.004, 0.005, 0.006, 0.007, 0.01, "scale"],
        "kernel": ["rbf"],
    }
    svm_opt = GridSearchCV(
        SVC(),
        param_grid=param_svm,
        cv=StratifiedKFold(5, shuffle=True, random_state=0),
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    svm_opt.fit(X_train_scaled, y_train.numpy())
    print(f"\nSVM best params: {svm_opt.best_params_}")
    
    acc_svm_opt = accuracy_score(
        y_test.numpy(), svm_opt.predict(X_test_scaled)
    )
    print(f"==> Optimized SVM test acc: {acc_svm_opt:.4f}")

    # --- Random Forest Grid Search Optimization ---
    param_rf = {
        "n_estimators": [900, 1000],
        "max_features": [None],
        "min_samples_split": [2, 3],
        "max_depth": [None, 15]
    }
    rf_opt = GridSearchCV(
        RandomForestClassifier(n_jobs=-1, random_state=0),
        param_grid=param_rf, 
        cv=5, 
        scoring="accuracy",
        n_jobs=-1
    )
    rf_opt.fit(X_train_scaled, y_train.numpy())
    
    acc_rf_raw = accuracy_score(
        y_test.numpy(), rf_opt.predict(X_test_scaled)
    )
    print(f"Opt RF params: {rf_opt.best_params_}")
    print(f"==> Optimized RF test acc: {acc_rf_raw:.4f}")

    # --- Re-train "PCA features" models with optimal parameters ---
    svm_pca_opt = SVC(**svm_opt.best_params_)
    rf_pca_opt  = RandomForestClassifier(
        n_jobs=-1, random_state=0, **rf_opt.best_params_
    )

    svm_pca_opt.fit(X_train_pca_sklearn, y_train.numpy())
    rf_pca_opt.fit(X_train_pca_sklearn, y_train.numpy())

    acc_svm_pca_opt = accuracy_score(
        y_test.numpy(), svm_pca_opt.predict(X_test_pca_sklearn)
    )
    acc_rf_pca_opt  = accuracy_score(
        y_test.numpy(), rf_pca_opt.predict(X_test_pca_sklearn)
    )

    print("\n=== PCA Features + Optimal Parameters ===")
    print(f"SVM acc (with PCA): {acc_svm_pca_opt:.4f}")
    print(f"RF  acc (with PCA): {acc_rf_pca_opt:.4f}")
    
    # Map raw models for future use
    svm_raw = svm_opt
    rf_raw  = rf_opt
    acc_svm_raw = acc_svm_opt

    print("\n=== Traditional Model Accuracy (Raw vs PCA) ===")
    print(f"SVM  : Raw {acc_svm_raw:.4f} | PCA {acc_svm_pca_opt:.4f}")
    print(f"RF   : Raw {acc_rf_raw :.4f}  | PCA {acc_rf_pca_opt :.4f}")

    # --- Plot Bar Chart: Raw vs PCA ---
    models_list = ["SVM", "Random Forest"]
    raw_scores  = [acc_svm_raw, acc_rf_raw]
    pca_scores  = [acc_svm_pca_opt, acc_rf_pca_opt]

    x = np.arange(len(models_list))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, raw_scores, width, label="Raw Features", edgecolor="black")
    plt.bar(x + width/2, pca_scores, width, label="PCA Features", edgecolor="black")
    plt.xticks(x, models_list)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy: Before vs After PCA Dimensionality Reduction")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot Confusion Matrices for PCA Models ---
    pca_sklearn_models = {
        "SVM (PCA)": svm_pca_opt, 
        "Random Forest (PCA)": rf_pca_opt
    }
    for name, clf in pca_sklearn_models.items():
        cm = confusion_matrix(y_test.numpy(), clf.predict(X_test_pca_sklearn))
        plt.figure(figsize=(4, 3))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Disordered", "Ordered"], 
            yticklabels=["Disordered", "Ordered"]
        )
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

    # --- Plot Confusion Matrices for Raw Models ---
    raw_sklearn_models = {
        "SVM (Raw)": svm_raw,
        "Random Forest (Raw)": rf_raw,
    }
    for name, clf in raw_sklearn_models.items():
        y_pred_raw = clf.predict(X_test_scaled)
        cm_raw = confusion_matrix(y_test.numpy(), y_pred_raw)

        plt.figure(figsize=(4, 3))
        sns.heatmap(
            cm_raw, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Disordered", "Ordered"],
            yticklabels=["Disordered", "Ordered"]
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix: {name}")
        plt.tight_layout()
        plt.show() 

    # =========================================================================
    # Section 5: Spatial Noise Robustness Testing
    # =========================================================================
    print("\n--- Commencing Noise Robustness Testing ---")

    def add_noise(X, noise_ratio=0.1):
        """
        Injects spatial noise into the Ising lattice by randomly flipping
        a specified percentage of the individual spins.
        
        Args:
            X (numpy.ndarray): The original feature data.
            noise_ratio (float): The fraction of spins to flip (0.0 to 1.0).
            
        Returns:
            numpy.ndarray: The corrupted lattice data.
        """
        X_noisy = copy.deepcopy(X)
        num_spins = X_noisy.shape[1]
        flip_count = int(noise_ratio * num_spins)
        
        for i in range(X_noisy.shape[0]):
            flip_indices = np.random.choice(
                num_spins, flip_count, replace=False
            )
            X_noisy[i, flip_indices] *= -1
            
        return X_noisy

    # Define noise parameters
    noise_max = 0.15
    n_points = 301
    noise_levels = np.linspace(0.0, noise_max, n_points)
    
    # Storage arrays for performance tracking
    acc_cnn_list = []
    acc_pca_nn_list = []
    acc_svm_noise = []
    acc_rf_noise = []

    # Ensure neural networks are in evaluation mode
    model.eval()
    model_pca.eval()

    for nr in noise_levels:
        # Generate noisy test set
        if nr == 0:
            X_test_noisy_flat = X_test_flat.copy()
        else:
            X_test_noisy_flat = add_noise(X_test_flat.copy(), noise_ratio=nr)

        # 1. Evaluate CNN
        X_test_noisy_tensor = torch.tensor(
            X_test_noisy_flat, dtype=torch.float32
        ).view(-1, 1, 32, 32).to(device)
        with torch.no_grad():
            outputs_cnn = model(X_test_noisy_tensor)
            _, pred_cnn = torch.max(outputs_cnn, 1)
            y_pred_cnn = pred_cnn.cpu().numpy()

        # 2. Evaluate PCA + NN
        X_test_noisy_pca = pca.transform(X_test_noisy_flat)
        inputs_pca = torch.tensor(X_test_noisy_pca, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs_pca = model_pca(inputs_pca)
            _, pred_pca = torch.max(outputs_pca, 1)
            y_pred_pca = pred_pca.cpu().numpy()
            
        # 3. Evaluate Traditional Models (Raw features)
        # Ensure consistent feature scaling before prediction
        X_test_noisy_scaled = scaler.transform(X_test_noisy_flat)  
        y_pred_svm = svm_opt.predict(X_test_noisy_scaled)
        y_pred_rf  = rf_opt.predict(X_test_noisy_scaled)
        
        # Calculate current accuracies
        acc_cnn = accuracy_score(y_test.numpy(), y_pred_cnn)
        acc_pca_nn = accuracy_score(y_test.numpy(), y_pred_pca)
        acc_svm_now = accuracy_score(y_test.numpy(), y_pred_svm)
        acc_rf_now = accuracy_score(y_test.numpy(), y_pred_rf)

        # Append to tracking lists
        acc_cnn_list.append(acc_cnn)
        acc_pca_nn_list.append(acc_pca_nn)
        acc_svm_noise.append(acc_svm_now)
        acc_rf_noise.append(acc_rf_now)

        # Log intermittent progress
        print(f"Noise {nr*100:05.2f}% | "
              f"CNN: {acc_cnn:.3f} | PCA+NN: {acc_pca_nn:.3f} | "
              f"SVM: {acc_svm_now:.3f} | RF: {acc_rf_now:.3f}")

    # --- Plot Final Robustness Curve ---
    plt.figure(figsize=(7, 4))
    x_percent = noise_levels * 100
    plt.plot(x_percent, acc_cnn_list, label="Deep CNN", linestyle="-")
    plt.plot(x_percent, acc_pca_nn_list, label="PCA + Dense NN", linestyle="-")
    plt.plot(x_percent, acc_svm_noise, label="SVM (Raw Features)", linestyle="-")
    plt.plot(x_percent, acc_rf_noise, label="RF (Raw Features)", linestyle="-")
    
    plt.xlabel("Spatial Noise Level (%)")
    plt.ylabel("Classification Accuracy")
    plt.title("Model Robustness Under Injected Spatial Noise")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =============================================================================
# Execution Entry Point
# =============================================================================
if __name__ == "__main__":
    # Safe multiprocess launch setup (Required for Windows architecture)
    mp.freeze_support()
    
    # "spawn" method ensures a fresh python interpreter starts for each worker, 
    # preventing memory leakage and deadlocks during parallel data generation.
    mp.set_start_method("spawn", force=True)
    
    # Deferred plotting imports to prevent GUI context errors in spawned workers
    import matplotlib.pyplot as plt
    import seaborn as sns   
    
    # Set PyTorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initialized Process. Hardware Accelerator using device: {device}")

    # Launch main pipeline
    main()
