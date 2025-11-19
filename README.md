# Stroke Prediction Model
---

This model predicts whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status.

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Dataset Description](#dataset-description)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Data Preprocessing](#data-preprocessing)
- [Modeling Approach](#modeling-approach)
- [Model Evaluation Metrics](#model-evaluation-metrics)
- [Results & Insights](#results--insights)
- [How to Use the Model](#how-to-use-the-model)
- [Future Improvements](#future-improvements)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview
Stroke remains one of the leading causes of long‑term disability and death worldwide. Early prediction can enable preventive care and save lives. 
This project uses machine learning to predict stroke risk based on demographic, clinical and lifestyle factors.

---

## Problem Statement
According to the World Health Organization (WHO), stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.
Hospitals and healthcare providers struggle to identify high‑risk patients early due to limited analytics capabilities. 
A data‑driven predictive model can support decision making by flagging individuals at risk for further clinical evaluation.

---

## Project Objectives
- Clean and preprocess medical patient data
- Engineer features that enhance predictive performance
- Train multiple machine learning models for benchmarking
- Identify the best model based on evaluation metrics
- Deploy the model for real‑time prediction

---

## Dataset Description
| Column | Description |
|--------|------------|
| Gender | Patient gender |
| Age | Patient age |
| Hypertension | 1 = Hypertension, 0 = No Hypertension |
| Heart_disease | Existing heart condition, 1 = has heart disease, 0 = No heart disease |
| ever_married | Marital history: Yes or No |
| work_type | Patient work type |
| residence_type | Patient residence type |
| avg_glucose_level | average glucose level |
| bmi | Patient BMI |
| Smoking_status | Smoking behavior |
| Stroke | Target variable (0 = No stroke, 1 = Stroke) |

## Exploratory Data Analysis
- Feature Understanding shows the distribution of patients with stroke was HIGHLY IMBALANCED, hence the dataset would require special handling in modeling.
- Distribution plots
- Correlation analysis
- Risk group comparisons

## Feature Engineering
Examples:
- Age groups
- Comorbidity flags
- Risk factor count features

## Data Preprocessing
- Handling missing values
- Scaling numerical features
- Encoding categorical variables
- SMOTE balancing for class imbalance

## Modeling Approach
Models trained include:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- KNN (SMOTE)

## Model Evaluation Metrics
Metrics used for comparison:
- Accuracy
- Precision
- Recall
- F1‑Score
- AUC‑ROC

## Results & Insights
The Gradient Boosting model delivered the best performance for detecting true positive stroke cases, demonstrating highest recall and F1‑score.

## How to Use the Model
```python
from joblib import load
model = load("stroke_model.joblib")
prediction = model.predict(input_data)
