# Task 2: Data Pre-processing, Data Analysis and Visualization
# The manual demonstrates this workflow using Orange.
# This Python version follows the same workflow for GitHub.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Replace with your dataset path
DATASET = "dataset.csv"

df = pd.read_csv(DATASET)
print("\nOriginal data:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# Fill numeric missing values with median
numeric_cols = df.select_dtypes(include="number").columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(exclude="number").columns
for col in categorical_cols:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))

# Standardize numeric features
numeric_cols = df.select_dtypes(include="number").columns
if len(numeric_cols) > 0:
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nProcessed data:")
print(df.head())

# Basic analysis
print("\nDescriptive statistics:")
print(df.describe())

# Distribution plots
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# Correlation visualization
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
