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
  - No Stroke (0): 4,861 (95.13%)
  - Stroke (1): 249 (4.87%)
  - Imbalance Ratio: 19.5:1
- Correlation analysis showed age was highly correlated with the occurence of Stroke.

## Feature Engineering
- Created binary flags and categorical bins for Age, glucose level, BMI, and comorbidity
- `Key Findings:`
  - Patients within the age 50 - 65 yrs (senior) make up the highest number (1,183) of participants
  - Majority of the patients (1,506) were obesed (BMI between 30 - 40).
  - The glucose level of most of the patients (3,131) was within the normal range (0 - 100)
  - Patients with comorbidity (i.e either Hypertension OR heart disease) had a higher risk (14.08 %) of having stroke than those who dont have these conditions (3.39 %)

## Data Preprocessing
- Imputed missing BMI values (201) with median: 28.10
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
