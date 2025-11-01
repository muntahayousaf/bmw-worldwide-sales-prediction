# 🏎️ BMW Sales Classification — Machine Learning Project

## 📘 Overview
This project focuses on classifying **BMW car sales performance** into different categories using machine learning techniques.  
By analyzing various features such as model type, year, price, and engine size, the model predicts the **Sales Classification** (target variable) with **100% accuracy**.

---

## 🎯 Objective
To build a predictive model that classifies BMW vehicles based on their sales performance using data-driven insights.

---

## 🧠 Features Used

| Feature | Description |
|----------|--------------|
| **Model** | BMW car model (e.g., 3 Series, 5 Series, etc.) |
| **Year** | Manufacturing or sales year |
| **Region** | Region where the car was sold |
| **Color** | Exterior color of the vehicle |
| **Fuel_Type** | Fuel category (Petrol, Diesel, Hybrid, etc.) |
| **Transmission** | Transmission type (Automatic / Manual) |
| **Engine_Size_L** | Engine size in liters |
| **Mileage_KM** | Total kilometers driven |
| **Price_USD** | Selling price in USD |
| **Sales_Volume** | Number of units sold |
| **Sales_Classification** | Target column (sales performance category) |

---

## ⚙️ Data Preprocessing
- Verified dataset completeness (no missing values found)
- Detected and removed outliers
- Applied **Label Encoding** for categorical columns
- Applied **StandardScaler** for numerical features
- Performed **train-test split** for model validation

---

## 🤖 Model Building
- **Algorithm Used:** Logistic Regression  
- **Cross Validation:** 5-Fold Cross Validation  
- **Evaluation Metrics:** Accuracy and Confusion Matrix  

---

## 📊 Model Evaluation

| Metric | Result |
|---------|---------|
| **Accuracy** | **100%** |
| **Validation** | 5-Fold Cross Validation |
| **Confusion Matrix** | Perfect classification (no errors) |

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
- **Environment:** Jupyter Notebook / VS Code

## 🧩 Key Steps in Code

# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
data = pd.read_csv("bmw_sales_data.csv")

# Split data
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Encode categorical variables
x = x.apply(LabelEncoder().fit_transform)

# Scale numerical features
scale = StandardScaler()
X = scale.fit_transform(x)

# Train model
model = LogisticRegression()
model.fit(X, y)

# Cross validation
cv = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("Cross Validation Accuracy:", cv.mean())

# Evaluation
y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", cm)
