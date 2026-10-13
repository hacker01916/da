#Q1
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('sales.csv')

print(df.head())

plt.boxplot(df.select_dtypes(include='number').values)

plt.title('Box Plot of Daily Sales')
plt.xlabel('Sales Data')
plt.ylabel('Sales Amount')

plt.show()

for col in df.select_dtypes(include='number').columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print("\nOutliers in", col)
    print(outliers)
   
    
#Q2
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(
    y_test, y_pred, average='weighted'
))
print("Recall:", recall_score(
    y_test, y_pred, average='weighted'
))
print("F1 Score:", f1_score(
    y_test, y_pred, average='weighted'
))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

