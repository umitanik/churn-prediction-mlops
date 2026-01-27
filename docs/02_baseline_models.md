# Baseline Model Selection

## 1. Objective
The goal of this stage is to evaluate a wide range of machine learning algorithms using their **default hyperparameters**. This establishes a performance baseline and helps identify which algorithms are best suited for the specific characteristics of our customer churn dataset.

## 2. Tested Algorithms
We benchmarked the following classification models:
| Model | Type | ROC-AUC Mean | ROC-AUC Std |
| :--- | :--- | :--- | :--- |
| **Gradient Boosting** | Tree-Based Ensemble | 0.8656 | 0.0070 |
| **CatBoost** | Tree-Based Ensemble | 0.8643 | 0.0092 |
| **LightGBM** | Tree-Based Ensemble | 0.8569 | 0.0066 |
| **XGBoost** | Tree-Based Ensemble | 0.8396 | 0.0097 |
| **Support Vector Machine** | Kernel-Based (SVM) | 0.8005 | 0.0150 |
| **Naive Bayes** | Probabilistic | 0.7354 | 0.0106 |
| **Decision Tree** | Single Tree | 0.6834 | 0.0130 |
| **K-Nearest Neighbors** | Distance-Based | 0.6540 | 0.0108 |

## 3. Methodology
To ensure a fair and robust comparison:
*   **Cross-Validation:** A **5-Fold Stratified Cross-Validation** strategy was used. This ensures that each fold maintains the same proportion of churners as the original dataset, which is crucial for our imbalanced data.
*   **Metric:** **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)** was chosen as the primary evaluation metric. It provides a better measure of the model's ability to distinguish between classes than simple accuracy.
*   **Preprocessing:** All models were trained on the same preprocessed training set. Scaling (MinMaxScaler) was applied within the pipeline explicitly for distance-based models (KNN, SVM) that require normalized features.

## 4. Results & Key Findings
Based on the benchmarking results in [04_model_selection.ipynb](../notebooks/04_model_selection.ipynb):
*   **Ensemble Methods Dominate:** Gradient Boosting type algorithms (CatBoost, Gradient Boosting, XGBoost, LightGBM) significantly outperformed simpler models like Naive Bayes and KNN.
*   **Stability:** The tree-based models showed consistent performance across different folds.

## 5. Conclusion & Selection
The top two performing models selected for the **Hyperparameter Tuning** phase are:
1.  **Gradient Boosting Classifier**
2.  **CatBoost Classifier**

These models were chosen because they achieved the highest mean ROC-AUC scores and demonstrated good stability.

## 6. References
*   **[04_model_selection.ipynb](../notebooks/04_model_selection.ipynb)**: Code for model benchmarking and visualization.
