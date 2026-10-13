#Q1
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('titanic.csv')
print(df.head())
plt.hist(df['Pclass'], bins=[0.5, 1.5, 2.5, 3.5],
         edgecolor='black', color='skyblue')
plt.xticks([1, 2, 3])
plt.xlabel('Passenger Class')
plt.ylabel('Number of Passengers')
plt.title('Histogram of Passengers by Class')
plt.show()

survival = df['Survived'].value_counts().sort_index()

plt.pie(
    survival,
    labels=['Not Survived', 'Survived'],
    autopct='%1.1f%%',
    startangle=90
)

plt.title('Titanic Survival Distribution')
plt.show()


#Q2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv('smartphones.csv')

print(df.head())
print(df.isnull().sum())

df = df.dropna()

X = df.drop('price_range', axis=1)
y = df['price_range']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))