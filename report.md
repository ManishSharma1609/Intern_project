# Short Report: Analysis and Modeling of Hyperspectral Corn Data

This document summarizes the workflow and key findings from the project on predicting vomitoxin concentration from hyperspectral corn data. The report covers the data preprocessing, dimensionality reduction, model selection and training, and offers recommendations for further improvement.

---

## 1. Data Preprocessing and Rationale

### Data Loading and Cleaning
- **Dataset Import:** The hyperspectral corn data is loaded from a CSV file.
- **Removing Non-essential Identifiers:** The unique sample identifier (e.g., `hsi_id`) is removed since it does not contribute to predictive insights.
- **Separating the Target:** The outcome variable, `vomitoxin_ppb`, is separated from the input features to ensure that only relevant predictors are used in model training.

### Handling Missing Data and Feature Scaling
- **Imputation:** Any missing values in the spectral data are filled using a mean imputation method, ensuring a complete dataset.
- **Normalization:** The spectral features are standardized so that each variable contributes equally during the model training, which prevents features with larger scales from dominating the learning process.

---

## 2. Dimensionality Reduction Insights

### Application of PCA
- **Objective:** Principal Component Analysis (PCA) is applied with a configuration to retain 95% of the overall variance.
- **Outcome:** Despite the original dataset having 448 spectral features, PCA reduces this number drastically—down to only 3 principal components. This indicates that the spectral data is highly redundant.
- **Visualization Benefits:** The reduced dimensions allow for straightforward 2D and 3D visualizations, which help reveal underlying clusters and patterns that might not be apparent in the high-dimensional space.

---

## 3. Model Selection, Training, and Evaluation

### Modeling Process
- **Initial Model Comparison:** Several regression models (including XGBoost, RandomForest, and LightGBM) were compared using evaluation metrics such as Adjusted R² and RMSE.
- **Selection Strategy:** Although XGBoost achieved an almost perfect training score for non-dimension reduced data, which might lead to overfitting in future applications, RandomForest was chosen for its more balanced performance and lower risk of overfitting on unseen data, as well as its superior performance when using dimension-reduced data.


### Hyperparameter Tuning and Final Training
- **Tuning Method:** GridSearchCV was used to fine-tune the RandomForest(The Best model found) parameters, such as the number of trees, maximum depth, and minimum samples required for splitting.
- **Data Splitting:** The data was divided into an 80% training set and a 20% test set.
- **Performance Metrics:** On the test set, key performance metrics like RMSE, R², and MAE were computed. A scatter plot of actual vs. predicted values further illustrated the model’s performance.

---

## 4. Key Observations and Recommendations

### Main Findings
- **Effective Data Preparation:** Imputation and standardization helped ensure that all spectral features were on a level playing field, while removing extraneous identifiers clarified the dataset.
- **Significant Dimensionality Reduction:** The ability to compress 448 features into just 3 principal components (while maintaining 95% of the data variance) demonstrates the presence of high correlation among spectral bands.
- **Model Robustness:** RandomForest provided a robust balance between fitting the training data and generalizing to new, unseen data, despite other models showing near-perfect training metrics.

### Recommendations
- **Enhance Feature Engineering:** Explore deriving additional features (such as domain-specific indices) that may further improve model accuracy.
- **Alternative Reduction Techniques:** Consider experimenting with other dimensionality reduction methods (like t-SNE or UMAP) to capture both global and local structures in the data.
- **Model Ensembling:** Investigate the possibility of combining predictions from multiple models to potentially enhance overall predictive performance.
- **Extended Validation:** Utilize cross-validation and external test sets to ensure that the selected model generalizes well.
- **Interactive Visualizations:** Implement interactive tools for a more in-depth exploration of data patterns and model outputs.

---

This report outlines the steps taken in data preprocessing, dimensionality reduction, model selection, and training, along with key insights and recommendations for future improvements in the hyperspectral data modeling pipeline.
