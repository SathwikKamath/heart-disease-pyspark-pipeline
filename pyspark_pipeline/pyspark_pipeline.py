#!/usr/bin/env python
# coding: utf-8

# # Sathwik Kamath 
# ### E-mail:kamathsathwik18@gmail.com
# ### LinkedIn: www.linkedin.com/in/sathwik-kamath
# ### Github: https://github.com/SathwikKamath

# In[2]:


import os
from pyspark.sql import SparkSession

# Set Java and Hadoop paths (needed for Spark to run locally on Windows)
os.environ["JAVA_HOME"] = "D:/openjdk-11.0.0.2_windows-x64/jdk-11.0.0.2"
os.environ["HADOOP_HOME"] = "D:/hadoop"

# Create Spark Session with MySQL JDBC Connector
spark = (
    SparkSession.builder
    .appName("HeartDiseasePipeline")
    .master("local[*]")  # Use all cores of your CPU
    .config("spark.jars", "file:///D:/mysql-connector-j-9.3.0/mysql-connector-j-9.3.0.jar")  # MySQL driver
    .config("spark.driver.extraClassPath", "D:/mysql-connector-j-9.3.0/mysql-connector-j-9.3.0.jar")
    .getOrCreate()
)


# In[3]:


# MySQL database details
jdbc_url = "jdbc:mysql://localhost:3306/heart_db"
props = {"user": "root", "password": "Kamath@2001"}

# Read data from MySQL table 'heart_data'
df = spark.read.jdbc(url=jdbc_url, table="cleveland_heart_disease", properties=props)

# Display schema and first few rows
print("\nSchema of the data:")
df.printSchema()
print("\nFirst 5 rows:")
df.show(5)


# In[6]:


# Removing the missing value
df_dropna= df.dropna()
df_dropna.show()


# # Apply StandardScaler to scale numeric features
# 

# In[7]:


from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

# 1️⃣ Categorical columns to encode
categorical_cols = ["cp", "thal", "slope"]

# Index categorical columns (convert categories → numeric indices)
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index") for col in categorical_cols]

# One-hot encode the indexed columns
encoders = [OneHotEncoder(inputCol=col+"_index", outputCol=col+"_ohe") for col in categorical_cols]

# 2️⃣ Numeric columns to scale
numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Assemble numeric features before scaling
numeric_assembler = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")

# Apply StandardScaler to scale numeric features
scaler = StandardScaler(inputCol="numeric_features", outputCol="scaled_numeric_features")

# 3️⃣ Combine everything into final features
assembler_inputs = [col+"_ohe" for col in categorical_cols] + ["scaled_numeric_features"]

final_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")


# # Feature Engineering (PySpark)
# ## Sections:
# ## 0) Imports & quick checks
# ## 1) StringIndex categorical columns
# ## 2) OneHotEncode indexed columns
# ## 3) Assemble numeric columns (pre-scaling)
# ## 4) Standard scale numeric features
# ## 5) Final VectorAssembler -> "features"
# ## 6) Build Pipeline, fit & transform, inspect results
# 

# In[8]:


# 0) Imports & quick checks
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col


# In[9]:


# 1) StringIndex categorical columns
# - create indexer objects for each categorical column.
# - handleInvalid="keep" avoids pipeline failure for unseen categories.
categorical_cols = ["cp", "thal", "slope"]
indexers = [
    StringIndexer(inputCol=cat, outputCol=f"{cat}_index", handleInvalid="keep")
    for cat in categorical_cols
]


# In[10]:


# 2) OneHotEncode indexed columns
# - convert each index column into an OHE vector column.
# - output columns will be like "cp_ohe", "thal_ohe", "slope_ohe".
encoders = [
    OneHotEncoder(inputCol=f"{cat}_index", outputCol=f"{cat}_ohe", handleInvalid="keep")
    for cat in categorical_cols
]


# In[11]:


# 3) Assemble numeric columns (pre-scaling)
# - combine raw numeric columns into single vector "numeric_features".
numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
numeric_assembler = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")


# In[12]:


# 4) Standard scale numeric features
# - StandardScaler standardizes (mean=0, std=1). withMean=True centers the data.
scaler = StandardScaler(
    inputCol="numeric_features",
    outputCol="scaled_numeric_features",
    withMean=True,
    withStd=True)


# In[13]:


# 5) Final VectorAssembler -> "features"
# - combine categorical OHE vectors + scaled numeric vector into one "features" vector.
assembler_inputs = [f"{cat}_ohe" for cat in categorical_cols] + ["scaled_numeric_features"]
final_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")


# In[14]:


# 6) Build Pipeline, fit & transform, inspect results
stages = indexers + encoders + [numeric_assembler, scaler, final_assembler]
pipeline = Pipeline(stages=stages)


# In[15]:


# Fit the pipeline (learns index mappings and scaler stats) then transform the DataFrame
pipeline_model = pipeline.fit(df_dropna)
processed_df = pipeline_model.transform(df_dropna)


# In[16]:


# Show important intermediate and final columns for verification
processed_df.select(
    "cp", "cp_index", "cp_ohe",
    "thal", "thal_index", "thal_ohe",
    "slope", "slope_index", "slope_ohe",
    "numeric_features", "scaled_numeric_features",
    "features", "target"
).show(5, truncate=False)


# # convert a few "features" vectors to lists for easy reading (Pandas)
# 

# In[17]:


# convert a few "features" vectors to lists for easy reading (Pandas)
sample_pdf = processed_df.select("features", "target").limit(5).toPandas()
sample_pdf["features_list"] = sample_pdf["features"].apply(lambda v: list(v))

print(sample_pdf[["features_list", "target"]].to_string(index=False))


# # Modeling

# In[18]:


# 0) Imports & setup
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel
import pyspark.sql.functions as F

# Ensure the label column exists and is double
# If you already have a 'label' column from StringIndexer, skip creating it.
if "label" not in processed_df.columns:
    processed_df = processed_df.withColumn("label", col("target").cast("double"))

# Quick check
processed_df.select("features", "target", "label").show(3, truncate=False)

# Train/test split
train_df, test_df = processed_df.randomSplit([0.8, 0.2], seed=42)
print(f"Train rows: {train_df.count()}, Test rows: {test_df.count()}")


# # Define evaluators
# 

# In[19]:


# 1) Evaluators (multiclass)
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")


# # Logistic Regression (multinomial)

# In[20]:


# 2) Logistic Regression (multinomial) - baseline
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100, family="multinomial")

lr_model = lr.fit(train_df)
pred_lr = lr_model.transform(test_df)

acc_lr = evaluator_acc.evaluate(pred_lr)
f1_lr  = evaluator_f1.evaluate(pred_lr)
prec_lr = evaluator_precision.evaluate(pred_lr)
recall_lr = evaluator_recall.evaluate(pred_lr)

print("=== Logistic Regression (multinomial) ===")
print(f"Accuracy: {acc_lr:.4f}  F1: {f1_lr:.4f}  Precision: {prec_lr:.4f}  Recall: {recall_lr:.4f}")


# ### Logistic Regression performed the best among all models.
# ### With an accuracy of 67.39% and precision of 78.53%, 
# ### it predicts heart disease cases quite accurately and minimizes false positives.
# ### Recall of 67.39% means it correctly identified around two-thirds of actual patients.
# ### Overall, it’s a reliable baseline model with balanced performance.
# 

# # Decision Tree — interpretable tree

# In[21]:


# 3) Decision Tree
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=6)  # tune maxDepth as needed
dt_model = dt.fit(train_df)
pred_dt = dt_model.transform(test_df)

acc_dt = evaluator_acc.evaluate(pred_dt)
f1_dt  = evaluator_f1.evaluate(pred_dt)
prec_dt = evaluator_precision.evaluate(pred_dt)
recall_dt = evaluator_recall.evaluate(pred_dt)

print("=== Decision Tree ===")
print(f"Accuracy: {acc_dt:.4f}  F1: {f1_dt:.4f}  Precision: {prec_dt:.4f}  Recall: {recall_dt:.4f}")


# ### Decision Tree achieved 52.17% accuracy and relatively lower precision (47.89%).
# ### This indicates the model made more incorrect positive predictions.
# ### Recall of 52.17% shows it detected only about half of the real patients.
# ### It may be overfitting or not generalizing well to unseen data.
# 

# # Random Forest — stronger general-purpose model
# 

# In[23]:


# 4) Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100, maxDepth=8, seed=42)
rf_model = rf.fit(train_df)
pred_rf = rf_model.transform(test_df)

acc_rf = evaluator_acc.evaluate(pred_rf)
f1_rf  = evaluator_f1.evaluate(pred_rf)
prec_rf = evaluator_precision.evaluate(pred_rf)
recall_rf = evaluator_recall.evaluate(pred_rf)

print("=== Random Forest ===")
print(f"Accuracy: {acc_rf:.4f}  F1: {f1_rf:.4f}  Precision: {prec_rf:.4f}  Recall: {recall_rf:.4f}")


# ### Random Forest performed slightly better than Decision Tree with 56.52% accuracy.
# ### Its precision (47.63%) and recall (56.52%) are moderate,
# ### meaning it balanced false positives and false negatives but wasn’t very strong in either.
# ### It provided more stable predictions than a single tree, but still below Logistic Regression.
# 

# # Compare all models side-by-side
# 

# In[25]:


# 5) Summary table
results = [
    ("LogisticRegression", acc_lr, f1_lr, prec_lr, recall_lr),
    ("DecisionTree", acc_dt, f1_dt, prec_dt, recall_dt),
    ("RandomForest", acc_rf, f1_rf, prec_rf, recall_rf)
]

print("\nModel\t\tAccuracy\tF1\tPrecision\tRecall")
for name, acc, f1, pr, rc in results:
    print(f"{name:17s}\t{acc:.4f}\t{f1:.4f}\t{pr:.4f}\t{rc:.4f}")


# In[26]:


# 7) Feature importances (vector indices -> importance)
importances = rf_model.featureImportances  # this is a SparseVector-like structure
print("Random Forest feature importances (index:value):")
for idx, imp in enumerate(importances):
    if imp > 0:
        print(f"Index {idx}: {imp:.6f}")

# NOTE: to interpret index -> original variable, use the mapping we built earlier
# (i.e., lengths of cp_ohe, thal_ohe, slope_ohe and numeric_cols order).


# The Random Forest model shows that features like oldpeak, thalach, and cholesterol have the highest importance scores, meaning they strongly influence the model’s prediction of heart disease.
# Categorical features such as chest pain type (cp) and thal also contribute but to a lesser extent.
# Overall, the model relies more on numeric medical indicators than on categorical ones to identify heart disease.

# In[27]:


from pyspark.mllib.evaluation import MulticlassMetrics

def confusion_matrix(model, test_df, model_name):
    # Make predictions on the test data
    predictions = model.transform(test_df)
    
    # Convert to (prediction, label) RDD
    pred_and_labels = predictions.select("prediction", "label").rdd.map(lambda x: (float(x[0]), float(x[1])))
    
    # Create metrics object
    metrics = MulticlassMetrics(pred_and_labels)
    
    # Confusion matrix
    cm = metrics.confusionMatrix().toArray()
    print(f"\n=== Confusion Matrix for {model_name} ===")
    print(cm)
    return predictions

# Generate confusion matrices for all three trained models
lr_predictions = confusion_matrix(lr_model, test_df, "Logistic Regression")
dt_predictions = confusion_matrix(dt_model, test_df, "Decision Tree")
rf_predictions = confusion_matrix(rf_model, test_df, "Random Forest")


# ## Logistic Regression
# 
# Most predictions are correct for class 0 (normal) → 25 correct.
# Some confusion between classes 1–3, but overall performs better than others.
# Interpretation: Logistic Regression predicts “no disease” and mild disease well — best overall among three models.
# 
# ## Decision Tree
# 
# Many wrong predictions — noticeable confusion between classes (off-diagonal values).
# Only a few classes correctly identified.
# Interpretation: Decision Tree is overfitting or not generalizing well — performance weaker than Logistic Regression.
# 
# ## Random Forest
# 
# Slightly better than Decision Tree — more correct predictions for class 0.
# Still confused between neighboring disease levels.
# Interpretation: Random Forest improves stability a bit, but still less accurate than Logistic Regression for this dataset.
# 
# ## Summary:
# Logistic Regression → best balanced model.
# Random Forest → moderate, some confusion.
# Decision Tree → weakest, confused across classes

# In[28]:


processed_df =df_dropna.select("*")
processed_df.show(5)


# In[29]:


from pyspark.sql.functions import col

# 0) Ensure label exists
if "label" not in processed_df.columns:
    processed_df = processed_df.withColumn("label", col("target").cast("double"))
processed_df.select("target", "label").show(3)


# In[30]:


from pyspark.ml.classification import LogisticRegression

# 7) Model (use best model; logistic regression here)
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=100)


# Defines a Logistic Regression Classifier, an ensemble of decision trees that vote on the final prediction. It’s robust, handles both numeric and categorical data, and reduces overfitting.

# In[31]:


from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 6) Build Pipeline, fit & transform, inspect results
stages = indexers + encoders + [numeric_assembler, scaler, final_assembler,lr]
pipeline = Pipeline(stages=stages)


# Creates a sequence of operations:
# 
# Assemble features
# 
# Scale them
# 
# Train model
# 
# pipeline.fit() will execute these steps in order
# 
# 

# # Train/test split (deterministic)

# In[33]:


# 9) Train/test split (deterministic)
train_df, test_df = processed_df.randomSplit([0.8, 0.2], seed=42)
print("Train rows:", train_df.count(), "Test rows:", test_df.count())



# Splits your dataset into training (80%) and testing (20%) sets for fair evaluation.

# In[34]:


# 10) Fit pipeline (this learns index mappings and scaler params)
pipeline_model = pipeline.fit(train_df)

save_path = r"D:\Py_Spark\Model_2\heart_pipeline_v1"  # you can pick another folder if you prefer
pipeline_model.write().overwrite().save(save_path)
print("✅ Pipeline model saved successfully at:", save_path)


# Learns scaling parameters (mean, std)
# 
# Trains the Logistic Regression model using training data
# 
# Returns a PipelineModel that contains all fitted components.

# #  Loding the ML Pipeline

# In[35]:


from pyspark.ml import PipelineModel
pm = PipelineModel.load(r"D:\Py_Spark\Model_2\heart_pipeline_v1")
print("✅ Pipeline loaded successfully!")


# In[36]:


# 11) Predict & evaluate on test set
predictions = pipeline_model.transform(test_df)
predictions.select("label", "prediction").show(5, truncate=False)


# Applies all preprocessing + model automatically to the test data, producing:
# 
# label: actual value
# 
# prediction: predicted class
# 
# probability: model confidence

# # Model Evealuvation

# In[38]:


eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
eval_f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
eval_prec= MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
eval_rec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

print("Accuracy :", eval_acc.evaluate(predictions))
print("F1       :", eval_f1.evaluate(predictions))
print("Precision:", eval_prec.evaluate(predictions))
print("Recall   :", eval_rec.evaluate(predictions))



# Calculates and prints the performance metrics — accuracy, precision, recall, and F1 score — to evaluate the model.
# 
# Saves the entire trained pipeline (assembler, scaler, model) to disk for later use — no retraining needed.
# 
# Reloads your trained pipeline. We can use it directly on new datasets that have the same columns — this ensures consistent preprocessing and predictions.
# 
# 

# In[ ]:




