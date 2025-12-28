# Reducing Missed Diagnoses: Machine Learning for Diabetes Screening
Machine learning project to predict diabetes risk using Python and multiple models. The workflow includes data understanding, exploratory analysis, preprocessing, model construction, hyperparameter tuning, cross validation, and evaluation using healthcare focused performance metrics.


## Overview
This project builds an end-to-end machine learning pipeline in Python to predict diabetes risk using medical and demographic data. The primary objective is to minimise missed diagnoses by prioritising recall and false negative rate, which are critical considerations in healthcare screening tasks.

The project replicates and extends a prior statistical analysis by implementing multiple supervised learning models and comparing their performance under healthcare-focused evaluation criteria.

---

## Dataset
The dataset was sourced from Kaggle and contains approximately 100,000 records with the following variables:

- Response variable: Diabetes status (binary)
- Quantitative features: Age, BMI, HbA1c level, Blood glucose level
- Categorical features: Gender, Hypertension, Heart disease, Smoking history

The dataset exhibits class imbalance, with approximately 8.5 percent of individuals diagnosed with diabetes.

---

## Project Workflow

### 1. Data Understanding and Exploration
- Analysed feature distributions and class imbalance
- Used boxplots for quantitative variables
- Used contingency tables and odds ratios for categorical variables

### 2. Data Preprocessing
- Standardised numerical variables
- One hot encoded categorical variables where appropriate
- Applied stratified train test splitting to preserve class proportions
- Integrated preprocessing into pipelines to prevent data leakage

### 3. Feature Selection
Feature inclusion was informed by exploratory analysis and model assumptions:
- K Nearest Neighbours excluded categorical variables due to Euclidean distance sensitivity
- Decision Tree retained strongly associated categorical features
- Logistic Regression removed weak predictors to improve interpretability
- Random Forest retained all features due to robustness against redundancy

---

## Models Implemented
- K Nearest Neighbours
- Decision Tree
- Logistic Regression
- Random Forest

---

## Hyperparameter Tuning
Hyperparameters were optimised using cross validation with an emphasis on minimising false negative rate:

- K Nearest Neighbours: neighbourhood size k
- Decision Tree: maximum depth
- Logistic Regression: classification threshold
- Random Forest: maximum depth

Five fold cross validation was used to ensure robustness.

---

## Model Evaluation
Models were evaluated using multiple metrics:
- Accuracy
- Precision
- Recall
- False Negative Rate
- False Positive Rate
- F1 score
- ROC AUC

Metric importance was prioritised based on healthcare relevance, with false negative rate and recall ranked highest.

---

## Key Findings
- Logistic Regression achieved the lowest false negative rate and highest recall, making it most suitable for diabetes screening
- Random Forest achieved the highest overall accuracy and ROC AUC but at the cost of higher false negative rate
- The results highlight the trade off between interpretability and predictive power in medical applications

---

## Files
- `diabetes_ml.ipynb`: Full Python implementation including EDA, preprocessing, model training, tuning, and evaluation
- `Diabetes_Report.pdf`: Detailed statistical report with results and discussion
- `figures/`: Visualisations used in analysis and reporting

---

## Limitations
- The dataset may not fully represent real world populations
- Logistic Regression assumes linear relationships between predictors and log odds
- Interaction effects were not explicitly modelled
- More complex models may achieve higher accuracy at the cost of interpretability

---

## Conclusion
This project demonstrates how machine learning models can be applied to healthcare screening problems with careful consideration of evaluation metrics. It shows that simpler, interpretable models can outperform more complex alternatives when domain specific priorities are correctly defined.

---

## Acknowledgements
Dataset provided by Mohammed Mustafa via Kaggle.
