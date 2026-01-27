# Data Overview & Project Context

## 1. Dataset Source & Purpose
The dataset used in this project is the **Bank Customer Churn** dataset. It contains comprehensive information about bank customers and their churn status (whether they left the bank or not). The primary goal of this project is to build a machine learning model to predict customer churn.

*   **Source:** [Kaggle - Bank Customer Churn Dataset](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn/data)
*   **Original Dataset Size:** 10,000 records.
*   **Split Strategy:**
    *   **Training Set (90%):** 9,000 records used for analysis, training, and validation.
    *   **Test Set (10%):** 1,000 records reserved as "unseen data" to simulate real-world performance.

## 2. Feature Dictionary
The dataset consists of **18 variables**. Below is the description of key features:

| Feature | Type | Description |
| :--- | :--- | :--- |
| **RowNumber** | Metadata | Row index (Removed in preprocessing). |
| **CustomerId** | Metadata | Unique customer identifier (Removed in preprocessing). |
| **Surname** | Metadata | Customer's last name (Removed in preprocessing). |
| **CreditScore** | Numerical | Credit score of the customer. |
| **Geography** | Categorical | Country of residence (e.g., France, Spain, Germany). |
| **Gender** | Categorical | Gender of the customer (Male/Female). |
| **Age** | Numerical | Age of the customer. |
| **Tenure** | Numerical | Number of years the customer has been with the bank. |
| **Balance** | Numerical | Account balance. |
| **NumOfProducts** | Numerical | Number of bank products used by the customer. |
| **HasCrCard** | Categorical | Whether the customer holds a credit card (1=Yes, 0=No). |
| **IsActiveMember** | Categorical | Whether the customer is an active member (1=Yes, 0=No). |
| **EstimatedSalary** | Numerical | Estimated annual salary. |
| **Satisfaction Score**| Numerical | Score provided by the customer for their satisfaction. |
| **Card Type** | Categorical | Type of credit card held (e.g., Diamond, Gold, Silver). |
| **Point Earned** | Numerical | Points earned from using the card. |
| **Complain** | Categorical | Complaint status. **Note:** Found to have data leakage (100% correlation with target) and removed. |
| **Exited** | **Target** | Churn status (1=Churned/Left, 0=Retained/Stayed). |


## 3. Key Insights (EDA)
Based on the analysis in [02_data_analyze.ipynb](../notebooks/02_data_analyze.ipynb):

*   **Class Imbalance:** The dataset is imbalanced, with a churn rate of approximately **20.5%**. This requires careful metric selection (e.g., ROC-AUC over Accuracy).
*   **Age Factor:** There is a positive correlation between Age and Churn. Older customers are more likely to leave.
*   **Geography:** The dataset is dominated by customers from France (~50%).
*   **Financial Status:** A significant portion (~25%) of customers have a **zero account balance**.
*   **Data Leakage Warning ! :** The `Complain` variable had a perfect correlation (1.0) with `Exited`. It was identified as a source of data leakage and removed to ensure the model learns real patterns, not just "who complained".

## 4. Preprocessing & Feature Engineering
As detailed in [03_pre-processing.ipynb](../notebooks/03_pre-processing.ipynb), the following steps were taken to prepare the data:

*   **Cleaning:** Removed non-predictive features (`RowNumber`, `CustomerId`, `Surname`) and the leaky feature (`Complain`).
*   **Encoding:** Applied One-Hot Encoding to categorical variables (`Geography`, `Gender`, `Card Type`).
*   **Feature Engineering:** Created new features to capture deeper insights:
    *   **`BalanceSalaryRatio`**: Ratio of balance to salary.
    *   **`TenureByAge`**: Normalized loyalty metric.
    *   **`HasBalance`**: Binary flag to handle the zero-inflated balance distribution.
    *   **`CreditScoreGivenAge`**: Credit behavior relative to age.

## 5. References & Notebooks
For the complete analysis and code, please refer to the project notebooks:
1.  **[01_split_dataset.ipynb](../notebooks/01_split_dataset.ipynb)**: Data splitting strategy.
2.  **[02_data_analyze.ipynb](../notebooks/02_data_analyze.ipynb)**: Detailed Exploratory Data Analysis (EDA).
3.  **[03_pre-processing.ipynb](../notebooks/03_pre-processing.ipynb)**: Feature engineering and data preparation pipeline.
