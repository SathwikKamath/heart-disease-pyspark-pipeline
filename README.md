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
```
### 🛠 Tech Stack
- PySpark (SparkSession, MLlib)
- MySQL (JDBC ingestion)
- Python (pandas, numpy)
- Jupyter Notebook
- HTML EDA Reports
- Machine Learning Models (Logistic Regression etc.)

### 📁 Project Structure

heart-disease-pyspark-pipeline/
│
├── notebooks/
│   ├── heart_analysis.html
│   ├── heart_analysis.ipynb
│   └── heart_analysis.py
│
├── pyspark_pipeline/
│   ├── pyspark_pipeline.html
│   ├── pyspark_pipeline.ipynb
│   └── pyspark_pipeline.py
│
├── README.md

### ▶️ How to Run

1. Install Java 11
2. Install Apache Spark + Hadoop
3. Install MySQL and create database `heart_db`
4. Load CSV into MySQL table
5. Install dependencies:
   pip install pyspark pandas mysql-connector-python
6. Run PySpark pipeline:
   python pyspark_pipeline/pyspark_pipeline.py

### 🤖 ML Pipeline Components
- Data ingestion from MySQL using JDBC
- Missing value handling
- Categorical encoding (StringIndexer + OneHotEncoder)
- VectorAssembler for feature combination
- Model training (Logistic Regression)
- Train/test split
- ROC-AUC and accuracy evaluation

⚠️ Key Challenges & How I Solved Them

SQL and Excel stored categorical values differently (1 vs '1.0')
✔ Solved by creating a custom preprocessing step that standardizes all categorical inputs into a single consistent format.

StringIndexer & OneHotEncoder produced mismatched category mappings
✔ Fixed by re-training the entire ML pipeline so that indexer labels and encoder outputs are fully aligned.

Unseen or missing category values caused indexing problems
✔ Added handleInvalid='keep' and custom fallback rules to safely map unexpected values.

### 📊 Results
- Logistic Regression Accuracy: 67.39%
- ROC-AUC Score: 65.75%
- Confusion matrix included in report
