# System Threat Forecaster

*Machine Learning Practice Course - IITM Online Degree*

*Based on the Kaggle Competition: System Threat Forecaster*

---

## Introduction

This project was developed as part of the Machine Learning Practice Course under the IITM Online Degree Program, and is based on the Kaggle competition - System Theat Forecaster. 

The objective of the competition is to predict the probability of a system getting infected by various families of malware, using telemetry data that describes the system's hardware, software, and antivirus configurations. 

The data was generated using threat reports collected by antivirus software installed on machines, capturing infection patterns and system characteristics. 

---

## Dataset Overview

Each row in the dataset represents a unique machine identified by `MachineID`. The target variable, `target`, indicates whether malware was detected on the machine (`1` for infected, `0` for not infected). 

- Total columns (features): 76
- Total rows (before cleaning): 100,000
- Target variable: `target`
- Data type: Mixed (numerical and categorical)

The dataset includes features such as operating system details, processor type, antivirus configurations, and user activity metrics. 

The primary task was to use these features to predict the likelihood of infections for each system.

---

## Project Workflow

The following steps were carried out during model development:

1. Data Loading & Exploration
     - Imported data and displayed key statistics.
     - Conducted Exploratory Data Analysis (EDA) to identify missing values, outliers and variable correlations.
2. Data Cleaning & Preprocessing
     - Removed duplicate records.
     - Handled missing values appropriately.
     - Encoded categorical variables using Label/One-Hot Encoding.
     - Scaled numerical features using StandardScaler.
3. Model Training & Evaluation
     - Split data into training and testing sets.
     - Trained and compared multiple models.
     - Evaluated performance using accuracy and cross-validation metrics.
  
---

## Models Implemented

| # | Model Description |
---
| 1 | Logistic Regression |
| 2 | Stochastic Gradient Descent (SGD Classifier) | 
| 3 | Logistic Regression (Hyperparameter Tuned) |
| 4 | SGD Classifier (Hyperparameter Tune) |
| 5 | Naive Bayes ( on PCA-Reduced Data) |
| 6 | K-Nearest Neighbours (KNN) |
| 7 | Bagging - Random Forest Classifier |
| 8 | Boosting - XGBoost, LightGBM |
| 9 | Stacking Ensemble (Random Forest + XGBoost + LightGBM) |
| 10 | Multi-Layer Perceptron (MLP) |

---

## Results & Observations

1. The dataset is balanced across target classes.
2. Features such as **Touch Enabled, Virtual Device, Optical Disk Drive Presence,** and **Firewall Enabled** showed little correlation with the target variable.
3. Features like **Antivirus Configuration, Being a Gamer,** and **Number of Antivirus Products Installed** exhibited noticeable differences between infected and non-infected systems.

### Model Performance Comparison

| Model | Accuracy |
---
| XGBoost | 0.6192 | 
---
| LightGBM | 0.6174 |
| Random Forest | 0.6097 |
| Logistic Regression | 0.5928 |
| Logistic Regression (tuned) | 0.5924 |
| SGD Classifier (tuned) | 0.5913 |

**XGBoost** delivered the best performance due to its gradient boosting framework, which is highly effective for structured data. It handles missing values efficiently, captures non-linear feature interactions, and optimises tree-based learning for classification tasks.

---

## Key Learnings

- Handling high-dimensional categorical data and performing feature encoding effectively.
- Understanding the importance of hyperparameter tuning for model improvement.
- Implementing ensemble techniques (bagging, boosting, stacking) to enhance model generalisation.
- Gaining hands-on experience with LighGBM and XGBoost for tabular data classification.

---

## Tools & Library Used

- Programming Language - Python
- Libraries - `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `lightgbm`

---

## How to Run the Code

1. Clone the repository:
   ```
   git clone https://github.com/<your-username>/System-Threat-Forecaster.git
   cd System-Threat-Forecaster
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the notebook or Python script:
   ```
   jupyter notebook
   ```
Open and execute the file `notebook.ipynb`

---

## Conclusion

This project demonstrates how machine learning can be applied to cybersecurity threat detection using structured system telemetry data. Through experimentation with multiple algorithms and ensemble methods, XGBoost emerged as the most effective model in predicting system vulnerability to malware attacks.
