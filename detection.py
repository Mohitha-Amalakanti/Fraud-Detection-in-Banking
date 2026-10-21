"""
Fraud Detection in Banking using Machine Learning
CodTech Internship - Mohitha Amalakanti
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("--- Fraud Detection in Banking ---")

# Synthetic banking data (real dataset lekunte)
np.random.seed(42)
n = 5000
data = {
    'transaction_amount': np.random.normal(5000, 2000, n),
    'transaction_time': np.random.randint(0, 24, n),
    'account_balance': np.random.normal(50000, 15000, n),
    'num_transactions': np.random.randint(1, 10, n),
    'location_risk': np.random.randint(0, 3, n),
    'is_international': np.random.randint(0, 2, n)
}
df = pd.DataFrame(data)
# Fraud logic
df['is_fraud'] = ((df['transaction_amount'] > 8000) & (df['location_risk']==2)).astype(int)

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))
print("Project Success!")