## Garment Employees Productivity Prediction

### Project Overview

This project aims to predict the **productivity of garment employees** based on various factors such as quarter, department, day, team, targeted productivity, and operational metrics. The goal is to develop a machine learning model that can accurately forecast employee productivity, enabling garment factories to optimize their production processes and resource allocation.

-----

### Technical Highlights

  * **Dataset**: [Kaggle - Productivity Prediction of Garment Employees](https://www.kaggle.com/datasets/ishadss/productivity-prediction-of-garment-employees)
  * **Size**: 1197 entries, 15 columns
  * **Key Features**:
      * date, quarter, department, day, team, targeted\_productivity, smv, wip, over\_time, incentive, idle\_time, idle\_men, no\_of\_style\_change, no\_of\_workers
  * **Approach**:
      * Data Cleaning (Handling Missing Values by dropping 'wip' column due to high nulls, No Duplicates Found). The notebook shows `wip` having 506 nulls out of 1197. Instead of dropping, filling with mean/median would be a better approach for the future.
      * Preprocessing of 'department' column (stripping whitespace and converting to lowercase).
      * Exploratory Data Analysis (Histograms, Boxplots, Heatmaps).
      * Label Encoding for all features (including numerical, as per the notebook).
      * Data Standardization using `StandardScaler`.
      * Classification Task (Predicting `department` - based on the code provided, where 'department' is dropped from X and becomes y). *However, the project title "Productivity Prediction" usually implies predicting `actual_productivity` as a regression task. The provided code classifies 'department'. This README will reflect the code's action for now, but note the discrepancy with the project title.*
      * Models Used:
          * Logistic Regression, Ridge Classifier, SVC, Random Forest, XGBoost, AdaBoost, Gradient Boosting, Bagging, Decision Tree
  * **Best Accuracy**:
      * \~100% across all models (Logistic Regression, Ridge Classifier, XGBoost, Random Forest, AdaBoost, Gradient Boosting, Bagging, Decision Tree, SVC). This high accuracy might indicate potential data leakage or a very separable dataset for the chosen target variable (`department`).

-----

### Purpose and Applications

  * Enable garment factories to **predict and optimize employee productivity** per department (based on code's target).
  * Facilitate better resource allocation and workforce planning.
  * Identify factors influencing productivity for process improvement.
  * Support data-driven decision-making in garment manufacturing operations.

-----

### Installation

Clone the repository:

```bash
git clone https://github.com/BhaveshBhakta/Garment-Employees-Productivity-Prediction-Using-ML.git
cd Garment-Employees-Productivity-Prediction-Using-ML
```

Install the necessary libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost
```

-----

### Collaboration

We welcome contributions to improve the project. You can help by:

  * Revisiting the target variable: If "Productivity Prediction" refers to `actual_productivity`, the task should be converted to regression.
  * Investigating the unusually high 100% accuracy to ensure no data leakage or overfitting is present.
  * Exploring more robust missing value imputation strategies for `wip` instead of dropping the column.
  * Implementing advanced feature engineering techniques.
  * Adding explainability (e.g., SHAP or LIME) to understand the drivers of productivity/department classification.
