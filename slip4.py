#Q1
import matplotlib.pyplot as plt

payment_methods = [
    'UPI', 'Credit Card', 'Debit Card',
    'Cash', 'Net Banking'
]

customers = [150, 80, 60, 40, 30]

total = sum(customers)

print("Payment Method Percentages:")
for method, count in zip(payment_methods, customers):
    percentage = (count / total) * 100
    print(f"{method}: {percentage:.2f}%")

plt.pie(
    customers,
    labels=payment_methods,
    autopct='%1.2f%%',
    startangle=90
)

plt.title("Customer Payment Method Preferences")
plt.axis('equal')
plt.show()


#Q2 (B)

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("ecommerce_customers.csv")

print("Original Dataset:")
print(df.head())

X = df.select_dtypes(include="number")

X = X.drop(
    columns=["CustomerID", "Customer_ID"],
    errors="ignore"
)

X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCustomer Clusters:")
print(df[["Cluster"]].head(10))

print("\nNumber of Customers in Each Cluster:")
print(df["Cluster"].value_counts().sort_index())

print("\nCluster Characteristics:")
print(df.groupby("Cluster")[X.columns].mean().round(2))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=df["Cluster"],
    cmap="viridis",
    s=50
)

centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    color="red",
    marker="X",
    s=200,
    label="Centroids"
)

plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title("E-Commerce Customer Segmentation")
plt.legend()
plt.show()
