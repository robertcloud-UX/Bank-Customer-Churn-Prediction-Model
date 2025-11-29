# Improved Customer Churn Prediction Model

This project implements an advanced machine learning pipeline to predict customer churn with high accuracy. It goes beyond baseline models by incorporating robust feature engineering, handling class imbalance, and utilizing ensemble methods.

## Project Overview

Customer churn is a critical metric for businesses. This project aims to predict which customers are likely to leave the service, allowing for proactive retention strategies.

## Key Improvements

- **Feature Engineering**: Created 10+ new features including:
    - **Age Groups**: Categorizing customers into Young, Middle, Senior, Elder.
    - **Balance Groups**: Segmenting by account balance.
    - **Interaction Features**: `BalancePerProduct`, `BalanceSalaryRatio`, `TenureAgeRatio`.
    - **Engagement Score**: A weighted score based on product usage and activity.
- **Class Imbalance Handling**: Utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the training dataset, ensuring the model learns to identify churners effectively.
- **Advanced Modeling**: Implemented and compared multiple algorithms:
    - **Logistic Regression** (with class weights)
    - **Random Forest Classifier**
    - **Gradient Boosting Classifier**
    - **Ensemble Model** (Weighted average of the above)

## Results

The models were evaluated on a stratified validation set. The **Gradient Boosting Classifier** and the **Ensemble Model** achieved the best performance.

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| **Gradient Boosting** | **88.97%** | **0.9152** |
| Ensemble Model | 88.30% | 0.9151 |
| Random Forest | 87.20% | 0.9083 |
| SMOTE + Logistic Regression | 85.47% | 0.8597 |
| Improved Logistic Regression | 82.10% | 0.8794 |
| *Baseline (Original)* | *~83.00%* | *-* |

## Files

- `improved_model_notebook.ipynb`: The main Jupyter Notebook containing the complete analysis, code, and visualizations.
- `train.csv`: Training dataset.
- `test.csv`: Test dataset.
- `submission_improved_robert.csv`: The final predictions generated for the test dataset.

## Dependencies

To run this project, you will need the following Python libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
```

## Usage

1.  Ensure all dependencies are installed.
2.  Open `improved_model_notebook.ipynb` in Jupyter Notebook or JupyterLab.
3.  Run all cells to reproduce the analysis, training, and prediction generation.
4.  The final submission file will be saved as `submission_improved_robert.csv`.
