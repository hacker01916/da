#Q1

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Fuel_Efficiency': [15, 18, 20, 22, 50],
    'Engine_Power': [100, 120, 150, 180, 500],
    'Vehicle_Weight': [1000, 1200, 1400, 1600, 1800],
    'Engine_Size': [1.0, 1.2, 1.5, 1.8, 2.0],
    'Acceleration': [12, 11, 10, 9, 8]
}

df = pd.DataFrame(data)

print("Vehicle DataFrame:")
print(df)

plt.figure(figsize=(6, 4))
sns.boxplot(y=df['Fuel_Efficiency'])
plt.title("Box Plot - Fuel Efficiency")
plt.show()

# Box plot for Engine Power
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['Engine_Power'])
plt.title("Box Plot - Engine Power")
plt.show()

def find_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return column[(column < lower) | (column > upper)]

print("Fuel Efficiency Outliers:")
print(find_outliers(df['Fuel_Efficiency']))

print("Engine Power Outliers:")
print(find_outliers(df['Engine_Power']))


##Q2

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

data = {
    'Temperature': [25, 30, 35, 20, 28],
    'Humidity': [60, 70, 80, 50, 65],
    'Wind_Speed': [10, 15, 20, 5, 12],
    'Pressure': [1010, 1015, 1020, 1005, 1012],
    'Rainfall': [5, 10, 15, 0, 8]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

minmax = MinMaxScaler()
minmax_data = minmax.fit_transform(df)

print("\nMin-Max Scaling:")
print(pd.DataFrame(minmax_data, columns=df.columns).round(2))

standard = StandardScaler()
standard_data = standard.fit_transform(df)

print("\nStandardization:")
print(pd.DataFrame(standard_data, columns=df.columns).round(2))

normalizer = Normalizer()
normalized_data = normalizer.fit_transform(df)

print("\nNormalization:")
print(pd.DataFrame(normalized_data, columns=df.columns).round(4))

