import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def ingest_data(filepath: str, id_col: str = 'hsi_id') -> pd.DataFrame:
    """
    Reads the CSV file and removes the identifier column if it exists.
    
    Parameters:
        filepath (str): Path to the CSV file.
        id_col (str): Name of the identifier column to drop.
    
    Returns:
        pd.DataFrame: DataFrame with the identifier column removed.
    """
    df = pd.read_csv(filepath)
    if id_col in df.columns:
        df = df.drop(columns=[id_col])
    return df

def inspect_data(df: pd.DataFrame) -> None:
    """
    Checks the dataset for missing values, provides summary statistics,
    and reports potential outliers using the IQR method.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    """
    print("Data Shape:", df.shape)
    print("\nMissing Values per Column:\n", df.isnull().sum())
    print("\nBasic Statistical Summary:\n", df.describe())
    
    # Identify outliers for each numeric column using the IQR rule.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print("\nOutlier Detection (IQR Method):")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        print(f"Column '{col}': {outlier_count} outliers (Lower: {lower_bound:.2f}, Upper: {upper_bound:.2f})")

def preprocess_data(df: pd.DataFrame, scaling_method: str = 'standard', exclude_cols: list = None):
    """
    Applies missing value imputation and normalization or standardization to numeric spectral features.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        scaling_method (str): 'standard' for StandardScaler or 'minmax' for MinMaxScaler.
        exclude_cols (list): List of columns (e.g., target columns) to exclude from scaling.
    
    Returns:
        df_scaled (pd.DataFrame): DataFrame with scaled numeric features.
        scaler: Fitted scaler instance.
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Select only numeric columns excluding those in exclude_cols.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude_cols)
    df_numeric = df[numeric_cols].copy()
    
    # Handle missing values using SimpleImputer with the 'mean' strategy.
    imputer = SimpleImputer(strategy='mean')
    df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric),
                                      columns=numeric_cols, 
                                      index=df.index)
    
    # Choose the appropriate scaler.
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaling_method must be either 'standard' or 'minmax'")
    
    # Fit the scaler on the imputed data and transform.
    scaled_values = scaler.fit_transform(df_numeric_imputed)
    df_scaled = pd.DataFrame(scaled_values, columns=numeric_cols, index=df.index)
    
    # Add back any columns that were excluded from scaling if they exist.
    cols_to_add = [col for col in exclude_cols if col in df.columns]
    if cols_to_add:
        df_scaled = pd.concat([df_scaled, df[cols_to_add]], axis=1)
    
    return df_scaled, scaler

def visualize_spectral_data(df: pd.DataFrame, target: str = None):
    """
    Visualizes the spectral bands by creating:
      1. A line plot of the average reflectance per spectral band.
      2. A heatmap comparing a subset of samples.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing spectral data.
        target (str): Optional; name of the target column to exclude from the spectral visualization.
    """
    # Exclude target column if provided.
    if target is not None and target in df.columns:
        spectral_df = df.drop(columns=[target])
    else:
        spectral_df = df.copy()
    
    # Keep only numeric features (assumed to be spectral reflectance values).
    spectral_df = spectral_df.select_dtypes(include=[np.number])
    
    # Compute the average reflectance per spectral band.
    avg_reflectance = spectral_df.mean()
    
    # Plot average reflectance as a line plot.
    plt.figure(figsize=(10, 6))
    plt.plot(avg_reflectance.index, avg_reflectance.values, marker='o', linestyle='-')
    plt.title("Average Reflectance Across Spectral Bands")
    plt.xlabel("Spectral Band")
    plt.ylabel("Average Reflectance")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Generate a heatmap for a subset of samples (first 20 samples).
    subset_samples = spectral_df.head(20)
    plt.figure(figsize=(12, 8))
    sns.heatmap(subset_samples, cmap='viridis', annot=False)
    plt.title("Heatmap of Spectral Data (First 20 Samples)")
    plt.xlabel("Spectral Band")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    filepath = 'TASK-ML-INTERN.csv'
    
    # 1. Ingest the data
    data = ingest_data(filepath, id_col='hsi_id')
    
    # 2. Inspect the dataset for missing values, summary stats, and outliers.
    inspect_data(data)
    
    # 3. Preprocess the data by standardizing spectral features (excluding the target 'vomitoxin_ppb').
    data_scaled, scaler_used = preprocess_data(data, scaling_method='standard', exclude_cols=['vomitoxin_ppb'])
    
    # 4. Visualize the spectral data:
    #    - Plot average reflectance across spectral bands.
    #    - Plot a heatmap for the first 20 samples.
    visualize_spectral_data(data_scaled, target='vomitoxin_ppb')
