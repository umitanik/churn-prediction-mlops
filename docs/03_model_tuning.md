# Hyperparameter Tuning & Final Optimization

## 1. Objective
After filtering the best candidates in the Model Selection phase, the next goal was to maximize their performance. In this stage, we focused on fine-tuning the hyperparameters of the top two models: **Gradient Boosting** and **CatBoost**.

Instead of using brute-force methods like Grid Search, we utilized **Optuna**, an automatic hyperparameter optimization framework, to efficiently search for the optimal set of parameters.

## 2. Methodology

| Component | Description |
| :--- | :--- |
| **Optimization Framework** | **Optuna** (Tree-structured Parzen Estimator - TPE sampler). |
| **Evaluation Metric** | **ROC-AUC** (Receiver Operating Characteristic - Area Under Curve). |
| **Validation Strategy** | **5-Fold Stratified Cross-Validation**. |
| **Search Budget** | **50 Trials** were executed for each model to explore the search space. |

## 3. Search Space
We defined dynamic search spaces for key hyperparameters:

### Gradient Boosting Search Space
| Hyperparameter | Range | Description |
| :--- | :--- | :--- |
| `n_estimators` | [100 - 1000] | Number of boosting stages to perform. |
| `learning_rate` | [0.001 - 0.1] | Step size shrinkage used in update to prevent overfitting. |
| `max_depth` | [3 - 10] | Maximum depth of the individual regression estimators. |
| `subsample` | [0.5 - 1.0] | The fraction of samples to be used for fitting the individual base learners. |

### CatBoost Search Space
| Hyperparameter | Range | Description |
| :--- | :--- | :--- |
| `iterations` | [100 - 1000] | The maximum number of trees that can be built. |
| `learning_rate` | [0.001 - 0.1] | The learning rate used for gradient descent optimization. |
| `depth` | [4 - 10] | Depth of the tree. |
| `l2_leaf_reg` | [0.001 - 10.0]| Coefficient at the L2 regularization term of the cost function. |
| `border_count` | [32 - 255] | The number of splits for numerical features. |

## 4. Results & Selection
Both models showed improvements after tuning. However, **CatBoost** was selected as the final production model due to its superior handling of categorical features and robustness.

| Final Model | Optimized ROC-AUC (CV) |
| :--- | :--- |
| **CatBoost Classifier** | **~0.87** |

## 6. References
*   **[05_model_tuning.ipynb](../notebooks/05_model_tuning.ipynb)**: Optuna implementation codes.
*   **[06_train_model.ipynb](../notebooks/06_train_model.ipynb)**: Final training with the best found parameters.