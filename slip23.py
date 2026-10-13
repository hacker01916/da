#Q1

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

data = {
    'Product_name': [
        'Laptop', 'Mobile', 'Tablet',
        'TV', 'Headphones', 'Camera'
    ],
    'Price': [60000, 20000, 30000, 45000, 3000, 40000],
    'Rating': [4.5, 4.2, 4.3, 4.6, 4.0, 4.4],
    'Number_of_Sales': [150, 300, 200, 100, 500, 120]
}

df = pd.DataFrame(data)

print("Product Dataset:")
print(df)

X = df[['Price', 'Rating', 'Number_of_Sales']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(
    Z,
    labels=df['Product_name'].tolist(),
    leaf_rotation=45
)

plt.title('Dendrogram of Similar Products')
plt.xlabel('Product Name')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()



#Q2
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

