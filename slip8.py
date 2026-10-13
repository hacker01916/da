#Q1

import pandas as pd
import matplotlib.pyplot as plt
data = {
    'Latitude': [37.77, 35.68, 28.61, -6.20, 19.43, 34.05],
    'Longitude': [-122.42, 139.69, 77.21, 106.85, -99.13, -118.24],
    'Depth': [10, 25, 15, 30, 12, 20],
    'Magnitude': [4.5, 5.2, 3.8, 6.1, 4.9, 5.7]
}

df = pd.DataFrame(data)
print(df)

plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    df['Longitude'],
    df['Latitude'],
    c=df['Magnitude'],
    cmap='coolwarm',
    s=df['Magnitude'] * 50,
    edgecolors='black'
)

plt.colorbar(scatter, label='Earthquake Magnitude')
plt.title('Earthquake Locations and Magnitudes')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)

plt.show()


#Q2(B)


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    'YearsExperience': [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    ],
    'Salary': [
        25000, 30000, 35000, 40000, 45000,
        50000, 55000, 60000, 65000, 70000
    ]
}

df = pd.DataFrame(data)
print(df)

X = df[['YearsExperience']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

new_salary = model.predict(
    pd.DataFrame({'YearsExperience': [11]})
)
print("Predicted Salary for 11 Years:", new_salary[0])

plt.scatter(X, y, color='blue', label='Actual Salary')
plt.plot(
    X,
    model.predict(X),
    color='red',
    label='Regression Line'
)

plt.title('Salary Prediction using Linear Regression')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()
