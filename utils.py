import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import torch
import matplotlib.pyplot as plt

def remove_price_col(train_data, val_data, test_data):
    price_cols = [col for col in train_data.columns if col.startswith('price')]
    train_data.drop(columns=price_cols, inplace=True)
    val_data.drop(columns=price_cols, inplace=True)
    test_data.drop(columns=price_cols, inplace=True)
    return train_data, val_data, test_data

def createX_and_y(data,scaler,torch_tensor=False):
    X = data.drop(columns='target').to_numpy()
    y = data['target'].to_numpy() 
    if scaler is not None:
        X = scaler.fit_transform(X)
    if torch_tensor:
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
    return X, y

def permutation_importance(model, X, y, metric, n_repeats=30, random_state=42):
    # Calculate the baseline performance
    baseline_score = metric(y, model.predict(X))
    importances = np.zeros(X.shape[1])

    for i in range(X.shape[1]):
        permuted_scores = np.zeros(n_repeats)
        print(i)
                
        for n in range(n_repeats):
            X_permuted = X.copy()
            np.random.seed(random_state + n)
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            permuted_score = metric(y, model.predict(X_permuted))
            permuted_scores[n] = permuted_score
        
        # Calculate the importance as the average change in performance
        importances[i] = np.mean(permuted_scores) - baseline_score

    return importances

def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def permutation_importance_NN(model, X, y, metric=rmspe, n_repeats=30, random_state=42):
    # Calculate the baseline performance
    baseline_score = metric(y.detach().cpu().numpy(), model(X).detach().cpu().numpy().squeeze())
    importances = np.zeros(X.shape[1])

    for i in range(X.shape[1]):
        permuted_scores = np.zeros(n_repeats)
        print(i)
        
        for n in range(n_repeats):
            X_permuted = X.clone()
            np.random.seed(random_state + n)
            X_permuted[:, i] = X_permuted[:, i][torch.randperm(X_permuted.size(0))]
            permuted_score = metric(y.detach().cpu().numpy(), model(X_permuted).detach().cpu().numpy().squeeze())
            permuted_scores[n] = permuted_score
        
        # Calculate the importance as the average change in performance
        importances[i] = np.mean(permuted_scores) - baseline_score

    return importances

def permutation_importance_LSTM(model, X, y, metric, n_repeats=30, random_state=42):
    model.eval()  # Set the model to evaluation mode
    
    if X.dim() == 2:  
        X = X.unsqueeze(1) 

    with torch.no_grad():
        baseline_output = model(X)
        baseline_score = metric(y.detach().cpu().numpy(), baseline_output.detach().cpu().numpy().squeeze())
    
    importances = np.zeros(X.shape[2])

    for i in range(X.shape[2]):
        permuted_scores = np.zeros(n_repeats)
        print(i)
        
        for n in range(n_repeats):
            X_permuted = X.clone()
            np.random.seed(random_state + n)
            perm_indices = torch.randperm(X_permuted.size(0))
            X_permuted[:, :, i] = X_permuted[perm_indices, :, i]
            
            with torch.no_grad():  
                permuted_output = model(X_permuted)
                permuted_score = metric(y.detach().cpu().numpy(), permuted_output.detach().cpu().numpy().squeeze())
            permuted_scores[n] = permuted_score
        
        importances[i] = np.mean(permuted_scores) - baseline_score

    return importances

def calculate_r_squared(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared.item()

# Enhanced plotting function
def plot_feature_importances(values, keys,model_name):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plotting
    bars = ax.barh(keys, values, color='steelblue', edgecolor='black')
    
    # Adding values on the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.01, bar.get_y() + bar.get_height() / 2, 
                f'{width:.4f}', ha='left', va='center')
    
    # Titles and labels
    ax.set_title(model_name, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Importance', fontsize=14, labelpad=15)
    ax.set_ylabel('Feature', fontsize=14, labelpad=15)
    
    # Inverting y-axis to have the highest importance on top
    ax.invert_yaxis()
    
    # Grid and styling
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    # Report text
    report_text = (
        "This report visualizes the aggregated importances of different features. "
        "Each feature's importance was summed up based on its initial characters, "
        "and the aggregated values were plotted to identify the most significant features."
    )
    plt.figtext(0.1, -0.1, "", wrap=True, horizontalalignment='left', fontsize=12)
    
    # Show plot
    plt.tight_layout()
    plt.show()