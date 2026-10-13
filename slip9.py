#Q1
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
data = {
    'Product_ID': [101, 102, 103, 104, 105],
    'Price': [100, 200, 300, 400, 500],
    'Quantity_Sold': [10, 20, 15, 30, 25],
    'Discount': [5, 10, 15, 20, 25],
    'Revenue': [950, 3600, 3825, 9600, 9375]
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
scaler = MinMaxScaler()
columns = ['Price', 'Quantity_Sold', 'Discount', 'Revenue']
df[columns] = scaler.fit_transform(df[columns])
print("\nTransformed DataFrame:")
print(df)


#Q2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

data = {
    'RAM': [2, 3, 4, 4, 6, 6, 8, 8, 12, 12, 16, 16],
    'Storage': [32, 64, 64, 128, 128, 256, 128, 256, 256, 512, 512, 1024],
    'Battery': [3000, 3500, 4000, 4200, 4500, 5000,
                4500, 5000, 5000, 5500, 6000, 6000],
    'Camera': [8, 12, 12, 16, 24, 32, 48, 50, 64, 108, 108, 200],
    'Price_Category': [
        'Low', 'Low', 'Low', 'Low',
        'Medium', 'Medium', 'Medium', 'Medium',
        'High', 'High', 'High', 'High'
    ]
}
df = pd.DataFrame(data)
print(df)
X = df[['RAM', 'Storage', 'Battery', 'Camera']]
y = df['Price_Category']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred, average='weighted'))
print("Confusion Matrix:")
print(confusion_matrix(
    y_test, y_pred, labels=['Low', 'Medium', 'High']
))
new_phone = pd.DataFrame({
    'RAM': [8],
    'Storage': [256],
    'Battery': [5000],
    'Camera': [50]
})
new_phone_scaled = scaler.transform(new_phone)
print("Predicted Price Category:",
      model.predict(new_phone_scaled)[0])

