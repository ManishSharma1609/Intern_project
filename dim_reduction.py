import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 
from data_utils import ingest_data, preprocess_data

def run_pca(preprocessed_df: pd.DataFrame, target: str = None, random_state: int = 42):
    """
    Applies PCA with n_components=0.95 to retain 95% of the variance.

    Parameters:
        preprocessed_df (pd.DataFrame): The preprocessed DataFrame (e.g., scaled spectral data).
        target (str, optional): Name of the target column to exclude from PCA and use for color-coding.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple of:
         - X_pca (np.ndarray): The PCA-transformed data.
         - explained_variance (np.ndarray): Explained variance ratios for each component.
         - y (pd.Series or None): Target values if provided; otherwise, None.
    """
    # Exclude target column from PCA if it exists.
    if target and target in preprocessed_df.columns:
        X = preprocessed_df.drop(columns=[target])
        y = preprocessed_df[target]
    else:
        X = preprocessed_df.copy()
        y = None

    # Use PCA to retain 95% of variance.
    pca = PCA(n_components=0.95, random_state=random_state)
    X_pca = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    
    return X_pca, explained_variance, y

def visualize_2d(X_pca, y=None, target: str = None):
    """
    Visualizes the first two principal components in a 2D scatter plot.
    """
    if X_pca.shape[1] < 2:
        print("Not enough components for 2D visualization.")
        return

    plt.figure(figsize=(8, 6))
    if y is not None:
        sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
        plt.colorbar(sc, label=target)
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("2D PCA Projection (95% Variance Retained)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_3d(X_pca, y=None, target: str = None):
    """
    Visualizes the first three principal components in a 3D scatter plot.
    """
    if X_pca.shape[1] < 3:
        print("Not enough components for 3D visualization.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    if y is not None:
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', alpha=0.7)
        fig.colorbar(sc, ax=ax, label=target)
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.7)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.set_title("3D PCA Projection (95% Variance Retained)")
    plt.tight_layout()
    plt.show()

def main():

    raw_filepath = 'TASK-ML-INTERN.csv'
    target = 'vomitoxin_ppb' 

    # Ingest the raw data and drop the 'hsi_id' identifier.
    data = ingest_data(raw_filepath, id_col='hsi_id')
    
    # Preprocess the data (imputation and scaling), excluding the target column.
    data_scaled, scaler_used = preprocess_data(data, scaling_method='standard', exclude_cols=[target])
    
    # Run PCA to retain 95% variance.
    print("Running PCA to retain 95% variance...")
    X_pca, explained_variance, y = run_pca(data_scaled, target=target)
    
    # Report the number of components and their explained variance ratios.
    print(f"Number of components retained: {X_pca.shape[1]}")
    print("Explained Variance Ratio:")
    for i, var in enumerate(explained_variance):
        print(f"  PC {i+1}: {var:.4f}")
    
    # Visualize the first two principal components (2D).
    visualize_2d(X_pca, y, target)
    
    # Visualize the first three principal components (3D) if available.
    visualize_3d(X_pca, y, target)

if __name__ == "__main__":
    main()
