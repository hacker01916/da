#Q1
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

data = {
    'Product': ['Mobile', 'Laptop', 'Tablet', 'TV', 'Camera'],
    'Price': [15000, 55000, 20000, 40000, 30000],
    'Rating': [4.2, 4.5, 4.0, 4.3, 4.1],
    'Sales': [120, 80, 100, 60, 90]
}

df = pd.DataFrame(data)
print("Product DataFrame:")
print(df)

X = df[['Price', 'Rating', 'Sales']]

Z = linkage(X, method='ward')

plt.figure(figsize=(8, 5))
dendrogram(Z, labels=df['Product'].values)

plt.title('Dendrogram of Similar Products')
plt.xlabel('Products')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()


#Q2(B)

import pandas as pd
from sklearn.preprocessing import LabelEncoder

df1 = pd.DataFrame({
    'Brand': ['Samsung', 'Apple', 'OnePlus', 'Samsung'],
    'Model': ['S24', 'iPhone15', '12R', 'A55']
})

df2 = pd.DataFrame({
    'Model': ['S24', 'iPhone15', '12R', 'A55'],
    'Storage': [256, 128, 256, 128],
    'Battery_Capacity': [4000, 3300, 5500, 5000],
    'Operating_System': ['Android', 'iOS', 'Android', 'Android']
})

print("DataFrame 1:")
print(df1)

print("\nDataFrame 2:")
print(df2)

merged_df = pd.merge(df1, df2, on='Model', how='inner')

print("\nMerged DataFrame:")
print(merged_df)

le = LabelEncoder()
merged_df['Brand_Encoded'] = le.fit_transform(merged_df['Brand'])

print("\nAfter Label Encoding:")
print(merged_df)

final_df = pd.get_dummies(
    merged_df,
    columns=['Operating_System'],
    dtype=int
)

print("\nTransformed DataFrame:")
print(final_df)