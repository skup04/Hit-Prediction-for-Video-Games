# Hit Prediction for Video Games

## Overview

This project investigates whether a video game can be classified as a **commercial hit** using only structured metadata available at or around release time.

The task is formulated as a **binary classification problem**:
- `Hit = 1` if a game exceeds a sales threshold,
- `Hit = 0` otherwise.

The goal is not just to maximize overall accuracy, but to determine whether video game metadata contains a meaningful predictive signal for identifying successful titles.

---

## Dataset

The analysis is based on the `vgsales.csv` dataset, which includes:

- `Name`
- `Platform`
- `Year`
- `Genre`
- `Publisher`
- `NA_Sales`
- `EU_Sales`
- `JP_Sales`
- `Other_Sales`
- `Global_Sales`

### Data cleaning
Rows with missing values in:
- `Year`
- `Publisher`

were removed before modeling.

---

## Problem Definition

A binary target variable `Hit` is created using a sales threshold based on the **80th percentile of `Global_Sales` within the training period**.

This threshold is then applied to both train and test sets so that the target definition remains consistent across time.

---

## Features Used

The model uses only the following predictors:

- `Year`
- `Genre`
- `Platform`
- `Publisher`

The following columns are **not used as features**:
- `Global_Sales` (target source)
- regional sales columns (`NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`)
- `Rank`

This prevents direct target leakage.

---

## Train/Test Split

A **time-based split** is used:

- **Train:** 1995–2007
- **Test:** after 2007

This setup is more realistic than a random split because it evaluates model performance on future releases rather than on randomly mixed observations.

---

## Modeling Pipeline

For `Logistic Regression` and `Random Forest`:
- `Year` is treated as a numerical feature
- `Genre`, `Platform`, and `Publisher` are treated as categorical features
- preprocessing is handled with `ColumnTransformer`
- categorical encoding is performed using `OneHotEncoder`

Models tested:
- `DummyClassifier`
- `Logistic Regression`
- `Random Forest`
- `Logistic Regression (class_weight='balanced')`
- `Random Forest (class_weight='balanced')`
- `CatBoostClassifier`

---

## Evaluation Metrics

Because the dataset is imbalanced, model quality is evaluated with:

- **Accuracy**
- **Precision (Hit)**
- **Recall (Hit)**
- **F1-score (Hit)**
- **ROC-AUC**

Confusion matrices and ROC curves are also used to compare models visually.

---

## Main Results

| Model | Accuracy | Precision (Hit) | Recall (Hit) | F1 (Hit) | ROC-AUC |
|------|----------:|----------------:|-------------:|---------:|--------:|
| LogReg | 0.826 | 0.618 | 0.039 | 0.073 | 0.765 |
| RandomForest | 0.804 | 0.411 | 0.254 | 0.314 | 0.700 |
| DummyClassifier | 0.693 | 0.175 | 0.199 | 0.187 | 0.495 |
| LogReg balanced | 0.795 | 0.424 | 0.456 | 0.440 | 0.756 |
| RandomForest balanced | 0.777 | 0.377 | 0.398 | 0.387 | 0.702 |
| CatBoost | 0.832 | 0.558 | 0.233 | 0.329 | 0.776 |

---

## Interpretation

The results show that:

- a standard **Logistic Regression** model achieves high overall accuracy, but almost completely fails to detect hit games;
- **class balancing** substantially improves performance on the positive class;
- **Balanced Logistic Regression** provides the best balance between precision and recall for hit detection;
- **CatBoost** achieves the best overall **Accuracy** and **ROC-AUC**, making it the strongest model in terms of overall predictive quality.

This means that the best model depends on the goal:
- if the task is to **identify hit games**, **Balanced Logistic Regression** is the most useful;
- if the task is to **rank games by success probability**, **CatBoost** is the strongest option.

---

## Key Takeaways

1. Video game metadata contains a real predictive signal for commercial success.
2. Class imbalance strongly affects model behavior.
3. High accuracy can be misleading in hit prediction tasks.
4. Balanced models are more useful when the goal is to detect successful titles.
5. Stronger prediction would likely require richer features beyond simple tabular metadata.

---

## Tools and Libraries

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- catboost

---

## Future Improvements

Possible next steps:

- tune hyperparameters for tree-based models,
- test threshold optimization for decision-making,
- compare performance across different hit definitions,
- add feature importance analysis,
- include richer metadata such as franchise or publisher history,
- build an interactive dashboard for model interpretation.
