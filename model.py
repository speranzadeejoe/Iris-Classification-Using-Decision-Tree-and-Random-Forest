import pandas as pd  
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read the CSV file
df = pd.read_csv(r"C:\Users\Admin\Desktop\mini\Iris.csv")  # Use 'r' before path to avoid errors

# Drop duplicates if found
df = df.drop_duplicates()

# Apply Min-Max Scaling on numerical features
scaler = MinMaxScaler()
num_features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
df[num_features] = scaler.fit_transform(df[num_features])

# Detect and remove outliers using IQR method (for SepalWidthCm as an example)
Q1 = df['SepalWidthCm'].quantile(0.25)
Q3 = df['SepalWidthCm'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_cleaned = df[(df['SepalWidthCm'] >= lower_bound) & (df['SepalWidthCm'] <= upper_bound)]

# OneHotEncode the target variable ('Species')
column_transformer = ColumnTransformer(
    transformers=[('onehot', OneHotEncoder(), ['Species'])],
    remainder='passthrough'
)

df_transformed = pd.DataFrame(column_transformer.fit_transform(df_cleaned), 
                              columns=column_transformer.get_feature_names_out())

# Define X (features) and y (target)
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df_cleaned[feature_columns]  # Features
y = pd.DataFrame(column_transformer.fit_transform(df_cleaned[['Species']]), 
                 columns=column_transformer.get_feature_names_out())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Decision Tree Classifier with Pruning
decision_tree_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
decision_tree_classifier.fit(X_train, y_train)

# Predictions on training & testing data
y_train_pred = decision_tree_classifier.predict(X_train)
y_test_pred = decision_tree_classifier.predict(X_test)

# Performance Metrics - Training Data
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average='macro')
train_recall = recall_score(y_train, y_train_pred, average='macro')
train_f1 = f1_score(y_train, y_train_pred, average='macro')

# Performance Metrics - Testing Data
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='macro')
test_recall = recall_score(y_test, y_test_pred, average='macro')
test_f1 = f1_score(y_test, y_test_pred, average='macro')

# Print Results
print("\n📊 Decision Tree - Training Performance:")
print(f"Accuracy: {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall: {train_recall:.4f}")
print(f"F1 Score: {train_f1:.4f}")

print("\n📊 Decision Tree - Testing Performance:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")

# Cross-Validation Score
cv_scores = cross_val_score(decision_tree_classifier, X, y, cv=5, scoring='accuracy')
print(f"\n📊 Cross-Validation Accuracy: {cv_scores.mean():.4f}")

# Random Forest Classifier for Better Generalization
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)
y_test_pred_rf = random_forest.predict(X_test)
test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
print(f"\n🌲 Random Forest Test Accuracy: {test_accuracy_rf:.4f}")

# Feature Importance Analysis
feature_importance = decision_tree_classifier.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_columns, 'Importance': feature_importance})
print("\n📊 Feature Importance:")
print(feature_importance_df.sort_values(by="Importance", ascending=False))
