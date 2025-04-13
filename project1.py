import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample synthetic dataset (you can replace this with a real CSV)
data = {
    'StudyHours': [2, 4, 6, 1, 5, 7, 3, 8, 2, 6],
    'Attendance': [70, 80, 90, 60, 85, 95, 75, 98, 65, 88],
    'PreviousScore': [50, 60, 75, 40, 68, 80, 55, 90, 45, 78],
    'Pass': [0, 1, 1, 0, 1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Features and target
X = df[['StudyHours', 'Attendance', 'PreviousScore']]
y = df['Pass']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

# Evaluation
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_preds))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_preds))
print("\nClassification Report for Logistic Regression:\n", classification_report(y_test, lr_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_preds))
