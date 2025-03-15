import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from data_utils import ingest_data, preprocess_data

from dim_reduction import run_pca

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Fits the model on training data and computes performance metrics on the test set.

    Returns:
        r2: R² score on test data.
        rmse: Root Mean Squared Error on test data.
        mae: Mean Absolute Error on test data.
        y_pred: Predictions on the test set.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    return r2, rmse, mae, y_pred

def train_and_tune_model():

    raw_filepath = 'TASK-ML-INTERN.csv'
    target = 'vomitoxin_ppb'  
    
    # 1. Ingest the raw data (dropping the identifier 'hsi_id')
    data = ingest_data(raw_filepath, id_col='hsi_id')
    
    # 2. Preprocess the data: impute missing values and standardize spectral features,
    #    excluding the target column.
    data_scaled, scaler = preprocess_data(data, scaling_method='standard', exclude_cols=[target])
    
    # 3. Apply PCA to the preprocessed data to retain 95% of the variance.
    #    This returns the PCA-transformed data and the target values.
    X_pca, explained_variance, y = run_pca(data_scaled, target=target)
    
    print(f"Number of PCA components retained: {X_pca.shape[1]}")
    print("Explained Variance Ratios:")
    for i, var in enumerate(explained_variance):
        print(f"  PC {i+1}: {var:.4f}")
    
    # 4. Split the PCA-reduced data into training (80%) and testing (20%) sets.
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    # 5. Define the model to tune.
    # Here, we're using RandomForest as it the best model based on findings of model_selection.py.
    model = RandomForestRegressor(random_state=42)
    
    # 6. Set up a parameter grid for hyperparameter tuning.
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    # 7. Perform GridSearchCV with 5-fold cross validation.
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print("Best Hyperparameters:", grid_search.best_params_)
    
    # 8. Evaluate the tuned model on the test set.
    r2, rmse, mae, y_pred_test = evaluate_model(best_model, X_train, y_train, X_test, y_test)
    
    print("\nTest Set Performance:")
    print(f"  R²  : {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE : {mae:.4f}")
    
    # 9. Scatter plot: Actual vs. Predicted values.
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.7, edgecolor='k')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values")
    # Identity line for reference.
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 10. Save the best model using pickle.
    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    
    return best_model, grid_search.best_params_, r2, rmse, mae

def main():
    best_model, best_params, r2, rmse, mae = train_and_tune_model()
    print("\nModel training and hyperparameter tuning completed.")
    print("Best Hyperparameters:", best_params)
    print(f"Test R² : {r2:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE : {mae:.4f}")

if __name__ == "__main__":
    main()
