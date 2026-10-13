#Q1
import matplotlib.pyplot as plt

departments = ['HR', 'Finance', 'IT', 'Sales', 'Marketing']
employees = [20, 35, 50, 40, 25]

plt.bar(departments, employees, color='skyblue')

plt.title('Distribution of Employees by Department')
plt.xlabel('Departments')
plt.ylabel('Number of Employees')

for i, value in enumerate(employees):
    plt.text(i, value, str(value), ha='center', va='bottom')

plt.tight_layout()
plt.show()



#Q2(B)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('ecommerce_customers.csv')

print(df.head())

print(df.isnull().sum())

data = df.select_dtypes(include=['number'])

data = data.fillna(data.median())

data = data.drop(
    columns=['CustomerID', 'Customer_ID'],
    errors='ignore'
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df.loc[data.index, 'Cluster'] = kmeans.fit_predict(X_scaled)

print("\nCustomer Segments:")
print(df[['Cluster']].value_counts().sort_index())


print("\nCluster Centers:")
print(pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=data.columns
))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=df.loc[data.index, 'Cluster'],
    cmap='viridis'
)

plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])
plt.title('E-commerce Customer Segmentation using K-Means')
plt.tight_layout()
plt.show()