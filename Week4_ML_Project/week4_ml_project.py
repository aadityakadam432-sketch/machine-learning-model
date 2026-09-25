import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)
# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset loaded successfully!")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

print("\nFeature Names:")
print(data.feature_names)

print("\nTarget Names:")
print(data.target_names)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


StandardScaler()
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000))
])

model.fit(X_train, y_train)

print("\nModel training completed!")

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

print("\nPredictions generated successfully!")

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

precision = precision_score(y_test, y_pred)

print("Precision:", precision)

recall = recall_score(y_test, y_pred)

print("Recall:", recall)

f1 = f1_score(y_test, y_pred)

print("F1 Score:", f1)

roc_auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC:", roc_auc)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))



cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)

disp.plot()

plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()


fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Logistic Regression")

plt.legend()

plt.tight_layout()

plt.savefig("roc_curve.png")

plt.show()