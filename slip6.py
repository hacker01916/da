##Q1
import pandas as pd
import matplotlib.pyplot as plt
data = {
    'Car Model': ['Swift', 'Honda City', 'Creta',
                  'Baleno', 'Nexon', 'i20'],
    'Engine Size': [1.2, 1.5, 1.6, 1.2, 1.5, 1.4],
    'Fuel Efficiency': [22, 18, 16, 23, 17, 20]
}
df = pd.DataFrame(data)
print(df)
colors = ['red', 'blue', 'green',
          'orange', 'purple', 'brown']
plt.scatter(df['Engine Size'],
            df['Fuel Efficiency'],
            c=colors, s=100)
plt.title('Engine Size vs Fuel Efficiency')
plt.xlabel('Engine Size (Litres)')
plt.ylabel('Fuel Efficiency (km/L)')
plt.grid(True)
plt.show()


##Q2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = {
    'Training Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Performance Score': [35, 40, 50, 55, 65, 70, 80, 85]
}
df = pd.DataFrame(data)
print(df)

X = df[['Training Hours']]
y = df['Performance Score']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("R2 Score:", r2_score(y, y_pred))

print("Predicted Score for 9 Hours:",
      model.predict(pd.DataFrame({'Training Hours': [9]}))[0])

plt.scatter(X, y, color='blue', label='Actual Scores')
plt.plot(X, y_pred, color='red', label='Regression Line')

plt.title('Training Hours vs Performance Score')
plt.xlabel('Training Hours Per Week')
plt.ylabel('Performance Score')
plt.legend()
plt.grid(True)
plt.show()
