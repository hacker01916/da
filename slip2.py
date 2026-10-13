#Q1
import pandas as pd
import numpy as np

data = {
    'Person_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [18, 22, 25, 28, 32, 35, 40, 45, 50, 60]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)


df['Age_Bin'] = pd.cut(
    df['Age'],
    bins=3,
    labels=['Young', 'Middle', 'Old'],
    include_lowest=True
)

print("\nDiscretized Data:")
print(df)


#Q2

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    'Likes': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'Comments': [10, 20, 25, 40, 45, 60, 70, 75, 90, 100],
    'Views': [1000, 2000, 3000, 4000, 5000,
              6000, 7000, 8000, 9000, 10000]
}

df = pd.DataFrame(data)
print("YouTube Dataset:")
print(df)


X = df[['Likes', 'Comments']]
y = df['Views']


model = LinearRegression()
model.fit(X, y)


df['Predicted_Views'] = model.predict(X)

print("\nPredicted Views:")
print(df[['Views', 'Predicted_Views']].round(2))

plt.scatter(df['Likes'], y, color='blue', label='Actual Views')
plt.plot(df['Likes'], df['Predicted_Views'],
         color='red', label='Regression Line')

plt.xlabel('Likes')
plt.ylabel('Views')
plt.title('YouTube Views Prediction')
plt.legend()
plt.grid(True)
plt.show()

