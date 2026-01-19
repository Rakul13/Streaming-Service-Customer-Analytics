# Streaming Service Customer Analytics

This project was completed as part of the **MSc Artificial Intelligence and Data Science**
programme. It applies supervised and unsupervised learning techniques to analyse customer
spending behaviour, churn, and segmentation in a streaming service dataset.

---

## Project Overview

Understanding customer behaviour is critical for subscription-based services.
This project explores how machine learning models can predict **monthly spending**,
identify **customer churn**, and uncover **behavioural clusters** using real-world style data.

---

## Objectives

- Predict **Monthly Spend** using regression models
- Compare linear, non-linear, and ensemble regression approaches
- Classify customers as churn / non-churn using classification models
- Segment customers using clustering algorithms
- Evaluate models using appropriate metrics and visualisations

---

## Methods Used

### Supervised Learning
- Linear Regression
- Polynomial Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Artificial Neural Network (ANN)
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

### Unsupervised Learning
- k-Means Clustering
- Hierarchical Clustering

---

## Model Evaluation

Models were evaluated using:
- R² and RMSE (regression)
- Accuracy, Precision, Recall, F1-score
- ROC-AUC for classification
- Elbow method and silhouette score for clustering

---

## Key Findings

- **Ridge Regression** performed best among regression models for predicting monthly spend.
- Including multiple numerical and categorical features significantly improved prediction accuracy.
- **Gradient Boosting Classifier** achieved the strongest churn prediction performance
  (ROC-AUC close to 1.0).
- Customer clustering revealed meaningful behavioural segments related to spending
  and subscription usage.

---

## Tools & Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Repository Contents

- `AI_Component_1.ipynb` – Full machine learning analysis notebook
- `README.md` – Project overview and insights


---

## Notes

This project demonstrates end-to-end machine learning workflows, including preprocessing,
model training, evaluation, and interpretation.
