# Hyperspectral Data Analysis and Predictive Modeling Pipeline

This repository provides a comprehensive pipeline for analyzing hyperspectral corn data to predict vomitoxin concentration. The pipeline includes data ingestion, preprocessing (with missing value imputation, normalization/standardization, and visualization), dimensionality reduction using PCA (retaining 95% variance), model selection and hyperparameter tuning, and an interactive Streamlit app for making predictions.

## Repository Structure

- **data_utils.py**  
  Contains functions for:
  - Data ingestion (reading CSV, dropping identifier columns)
  - Data inspection (missing values, outlier detection, summary statistics)
  - Data preprocessing (imputation, normalization/standardization)
  - Visualization of spectral bands (line plots and heatmaps)

- **dim_reduction.py**  
  Implements PCA (configured to retain 95% of the variance) on preprocessed data and visualizes the PCA-transformed data in both 2D and 3D.

- **model_selection.py**  
  Trains multiple models (XGBoost, RandomForest, LightGBM) on PCA-reduced data, calculates evaluation metrics (Adjusted R² and RMSE), and selects the best model based on a composite score.

- **model_trainer.py**  
  Uses the best model from the model selection step to perform hyperparameter tuning via GridSearchCV. The script:
  - Splits the PCA-reduced data into training (80%) and testing (20%) sets.
  - Trains the best model using the optimal hyperparameters.
  - Evaluates the model (RMSE, R², MAE) on the test set.
  - Generates a scatter plot of actual vs. predicted values.
  - Saves the trained model with pickle.


- **requirements.txt**  
  Lists all the dependencies required to run the project.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ManishSharma1609/Intern_project.git
   cd Intern_project 
   ```


2. **Set Up a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```


3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```


## Repository Structure

**Data Ingestion, Preprocessing, and Visualization**
Run the data_utils.py script to load, inspect, and preprocess the data, as well as to visualize spectral bands:
   ```bash
   python data_utils.py
   ```

**Dimensionality Reduction with PCA**
Run the dim_reduction.py script to apply PCA (retaining 95% variance) and view both 2D and 3D projections:
   ```bash
   python dim_reduction.py
   ```

**Model Selection**
Run the model_selection.py script to train and evaluate multiple models on the PCA-reduced data:
   ```bash
   python model_selection.py
   ```

**Model Training and Hyperparameter Tuning**
Run the model_trainer.py script to perform hyperparameter tuning on the best model, evaluate its performance on a test set, and save the model:
   ```bash
   python model_trainer.py
   ```

## Additional Notes

- **Data File Path:**  
    Ensure that the file paths specified in the code (e.g., TASK-ML-INTERN.csv) are updated to match your environment.

- **PCA Configuration:**  
    PCA is configured to retain 95% of the variance. This threshold can be adjusted depending on your needs.

- **Graphical User Interface (GUI):**  
    Important: This code is designed to run in an environment with a built-in GUI for displaying images (e.g., a local machine with a desktop environment). If you are using a headless environment like Codespaces or a remote server without a GUI, consider saving plots to image files (using plt.savefig()) or running the code in a local environment with GUI support.


