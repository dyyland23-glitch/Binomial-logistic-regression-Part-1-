# Binomial-logistic-regression-Part-1-


# Binomial Logistic Regression (Part 1): Model Construction & Diagnostics

This repository contains practical machine learning workflows and dataset resources focusing on exploratory data analysis (EDA), statistical assumptions checking, and binary classification using **Binomial Logistic Regression** implemented in Python.

---

## 📂 Repository Contents

### **1. Jupyter Notebooks**

#### [`Annotated follow-along guide_ Construct a logistic regression model with Python.ipynb`](https://github.com/dyyland23-glitch/Binomial-logistic-regression-Part-1-/blob/main/Annotated%20follow-along%20guide_%20Construct%20a%20logistic%20regression%20model%20with%20Python.ipynb)
* **Project Overview**: Demonstrates an end-to-end binomial logistic regression pipeline to predict binary categorical outcomes from multi-attribute tabular data.
* **Key Tasks & Methods**:
  * **Exploratory Data Analysis (EDA)**: Inspects variable types, calculates central tendency and dispersion metrics, and identifies class distribution balances.
  * **Assumption Checking**: Evaluates linear relationships between continuous independent predictors and the logit of the outcome, assesses multicollinearity via correlation analysis, and screens for high-leverage outliers.
  * **Data Preprocessing & Encoding**: Formats feature sets, handles missing values, and applies one-hot or dummy encoding to categorical explanatory variables.
  * **Model Training & Diagnostics**: Fits a logistic regression model, outputs estimated log-odds coefficients ($\beta$), and converts log-odds to odds ratios ($e^\beta$) for real-world interpretation.
  * **Evaluation Metrics**: Assesses classification performance across decision thresholds using confusion matrices, precision, recall, specificity, and ROC-AUC curves.
* **Core Libraries & Key Functions**:
  * `pandas`: `read_csv()`, `info()`, `describe()`, `isna()`, `get_dummies()`
  * `numpy`: Mathematical functions, log transformations, array shaping (`.ravel()`)
  * `sklearn.model_selection`: `train_test_split()`
  * `sklearn.linear_model`: `LogisticRegression()`
  * `sklearn.metrics`: `confusion_matrix()`, `classification_report()`, `roc_auc_score()`, `roc_curve()`
  * `statsmodels.api`: `Logit()`, summary table inspection (`.summary()`), and p-value inference
  * `seaborn` & `matplotlib.pyplot`: `regplot()`, `heatmap()`, `boxplot()`

---

### **2. Datasets (`.csv`)**
* [`activity binomial logistic regression.csv`](https://github.com/dyyland23-glitch/Binomial-logistic-regression-Part-1-/blob/main/activity%20binomial%20logistic%20regression.csv): Tabular observation dataset containing categorical target outcomes and numerical/categorical features used to fit and validate the binomial logistic regression model.

---
