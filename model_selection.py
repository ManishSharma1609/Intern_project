import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from data_utils import ingest_data, preprocess_data
from dim_reduction import run_pca

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Trains the model and evaluates its performance using Adjusted R² and RMSE.

    Returns:
        adjusted_r2: Adjusted R² score.
        rmse: Root Mean Squared Error.
        y_pred: Predictions on X_test.
    """
    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute R²
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = X_train.shape[1]
    
    # Calculate Adjusted R²
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return adjusted_r2, rmse, y_pred

def model_selection_pipeline():
    """
    Loads the data, preprocesses it, applies PCA to retain 95% variance, splits the data,
    trains multiple models, and selects the best model based on a composite score (Adjusted R² - RMSE).
    """
   
    raw_filepath = 'TASK-ML-INTERN.csv'
    target = 'vomitoxin_ppb' 
    
    # 1. Ingest the raw data (dropping the identifier 'hsi_id')
    data = ingest_data(raw_filepath, id_col='hsi_id')
    
    # 2. Preprocess the data: impute missing values and standardize the spectral features,
    #    excluding the target column.
    data_scaled, scaler_used = preprocess_data(data, scaling_method='standard', exclude_cols=[target])
    
    # 3. Apply PCA (n_components=0.95) to retain 95% variance.
    #    This function returns the PCA-transformed features and the explained variance ratios.
    X_pca, explained_variance, y = run_pca(data_scaled, target=target)
    
    print(f"Number of PCA components retained to explain 95% variance: {X_pca.shape[1]}")
    print("Explained Variance Ratios:")
    for i, var in enumerate(explained_variance):
        print(f"  PC {i+1}: {var:.4f}")
    
    # 4. Split the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    # 5. Define the models to evaluate.
    models = {
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42),
        "LightGBM": lgb.LGBMRegressor(random_state=42)
    }
    
    results = {}
    
    # 6. Evaluate each model.
    for name, model in models.items():
        print(f"Training and evaluating {name}...")
        adj_r2, rmse, _ = evaluate_model(model, X_train, y_train, X_test, y_test)

        # Here we define: score = Adjusted R² - RMSE
        score = adj_r2 - rmse
        results[name] = {
            "Adjusted R2": adj_r2,
            "RMSE": rmse,
            "Score": score
        }
    
    # 7. Display results and select the best model.
    print("\nModel Evaluation Results:")
    for name, metrics in results.items():
        print(f"{name}: Adjusted R2 = {metrics['Adjusted R2']:.4f}, RMSE = {metrics['RMSE']:.4f}, Score = {metrics['Score']:.4f}")
    
    best_model_name = max(results, key=lambda k: results[k]["Score"])
    best_score = results[best_model_name]["Score"]
    print(f"\nBest Model: {best_model_name} with a composite score of {best_score:.4f}")
    
    return best_model_name, results

def main():
    best_model_name, results = model_selection_pipeline()
    print("Model selection pipeline completed.")

if __name__ == "__main__":
    main()
