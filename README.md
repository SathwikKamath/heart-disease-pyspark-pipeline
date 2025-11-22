# ❤️ Heart Disease Prediction – PySpark + SQL ML Pipeline

End-to-end **PySpark + SQL machine learning pipeline** for predicting the presence of heart disease using the UCI Heart Disease dataset.


This project demonstrates a complete workflow:
- Data stored in **MySQL**
- Ingestion & preprocessing using **PySpark**
- **Feature engineering** and **ML pipeline** with Spark MLlib
- Evaluation using standard classification metrics

---

## 🧾 Dataset Source

This project uses the **Heart Disease** dataset from the UCI Machine Learning Repository.

- Dataset link: https://archive.ics.uci.edu/dataset/45/heart+disease  

> ⚠️ The **raw dataset is NOT included** in this repository due to licensing.  
> Please download it from the UCI link and load it into MySQL as described below.

---

## 🏗️ Architecture

High-level data flow:

```text
UCI Dataset (CSV) 
      ↓
MySQL (heart_db.heart_data table)
      ↓ JDBC
PySpark (SparkSession)
      ↓
Preprocessing + Feature Engineering
      ↓
Spark MLlib Models (Logistic Regression, etc.)
      ↓
Metrics, Predictions, HTML / Notebook Analysis
