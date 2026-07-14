# ============================================================
# PHASE 1: Setup & First Look
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('HR_Employee_Attrition.csv')

# First look
print(df.shape)          # (rows, columns) — sanity check on size
print(df.info())         # column names, types, non-null counts
print(df.describe())     # stats summary for numeric columns

# Check class imbalance in target variable — important for Phase 4 model evaluation
print(df['Attrition'].value_counts(normalize=True))


# ============================================================
# PHASE 2: Cleaning
# ============================================================

# Check for missing values
print(df.isnull().sum())

# Drop columns where every row has the same value — zero predictive power
df.drop(['EmployeeCount', 'StandardHours', 'Over18'], axis=1, inplace=True)

# Check for duplicate rows
print(df.duplicated().sum())

# Convert target variable to numeric (models need numbers, not text)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})


# ============================================================
# PHASE 3: EDA
# ============================================================

# Who leaves? Attrition by job role (raw counts, then actual rate)
sns.countplot(data=df, x='JobRole', hue='Attrition')
plt.xticks(rotation=45)
plt.title('Attrition Count by Job Role')
plt.tight_layout()
plt.show()

# Rate matters more than count — a small team losing half its people is worse than a big team losing 10%
role_attrition = df.groupby('JobRole')['Attrition'].mean().sort_values(ascending=False)
print(role_attrition)

# Does pay level relate to leaving?
sns.boxplot(data=df, x='Attrition', y='MonthlyIncome')
plt.title('Monthly Income vs Attrition')
plt.show()

# Does overtime relate to leaving?
sns.countplot(data=df, x='OverTime', hue='Attrition')
plt.title('Attrition by OverTime')
plt.show()

# Does satisfaction level relate to leaving?
sns.countplot(data=df, x='JobSatisfaction', hue='Attrition')
plt.title('Attrition by Job Satisfaction Level')
plt.show()

# Are leavers mostly newer employees?
sns.boxplot(data=df, x='Attrition', y='YearsAtCompany')
plt.title('Years at Company vs Attrition')
plt.show()

# Does commute distance relate to leaving?
sns.boxplot(data=df, x='Attrition', y='DistanceFromHome')
plt.title('Distance From Home vs Attrition')
plt.show()

# Correlation heatmap — quick scan for strongest linear relationships with Attrition
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()


# ============================================================
# PHASE 4: Modeling
# ============================================================

# ---- Step 1: Encode categorical columns ----
df_encoded = pd.get_dummies(df, drop_first=True)

# ---- Step 2: Split features and target ----
X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']

# ---- Step 3: Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Step 3.5: Scale features (Logistic Regression is scale-sensitive) ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit only on train
X_test_scaled = scaler.transform(X_test)        # apply same scale to test, don't refit

# ---- Step 4: Logistic Regression ----
# class_weight='balanced' forces the model to pay attention to the minority class (leavers)
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)
print("Logistic Regression Results:")
print(classification_report(y_test, y_pred_log))

# ---- Step 5: Random Forest ----
# Tree-based models aren't scale-sensitive, so use unscaled data here
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Results:")
print(classification_report(y_test, y_pred_rf))

# ---- Step 6: Feature importance — headline chart ----
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=importances.head(10).values, y=importances.head(10).index)
plt.title('Top 10 Attrition Drivers (Random Forest)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

print("\nTop 10 Feature Importances:")
print(importances.head(10))