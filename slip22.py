#Q1

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {
    'City': ['Pune', 'Mumbai', 'Nashik', 'Pune', 'Mumbai'],
    'House_Type': ['Flat', 'Bungalow', 'Flat', 'Row House', 'Flat'],
    'House_Area': [800, 1500, 600, 1200, 1000],
    'Number_of_Rooms': [2, 4, 1, 3, 2],
    'Monthly_Rent': [15000, 40000, 8000, 25000, 20000]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

columns = ['House_Area', 'Number_of_Rooms', 'Monthly_Rent']

scaler = MinMaxScaler()
df[columns] = scaler.fit_transform(df[columns])

print("\nNormalized DataFrame:")
print(df.round(2))


#Q2

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

data = {
    'Customer_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Customer_Name': [
        'Amit', 'Rahul', 'Priya', 'Sneha', 'Rohan',
        'Neha', 'Kiran', 'Pooja', 'Vijay', 'Anita'
    ],
    'Netflix': [
        'Yes', 'No', 'Yes', 'Yes', 'No',
        'Yes', 'No', 'Yes', 'No', 'Yes'
    ],
    'Amazon_Prime': [
        'No', 'Yes', 'Yes', 'No', 'Yes',
        'Yes', 'No', 'No', 'Yes', 'Yes'
    ]
}

df = pd.DataFrame(data)
print("Customer Dataset:")
print(df)

netflix = set(df[df['Netflix'] == 'Yes']['Customer_ID'])
amazon = set(df[df['Amazon_Prime'] == 'Yes']['Customer_ID'])

only_netflix = len(netflix - amazon)
only_amazon = len(amazon - netflix)
both = len(netflix & amazon)

plt.figure(figsize=(7, 6))
venn2(
    subsets=(only_netflix, only_amazon, both),
    set_labels=('Netflix', 'Amazon Prime'),
    set_colors=('red', 'blue'),
    alpha=0.6
)
plt.title("Netflix vs Amazon Prime Users")
plt.show()

labels = ['Only Netflix', 'Only Amazon Prime', 'Both Services']
sizes = [only_netflix, only_amazon, both]
colors = ['red', 'blue', 'green']

plt.figure(figsize=(7, 6))
plt.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90
)
plt.title("Distribution of Streaming Service Users")
plt.axis('equal')
plt.show()

print("\nOnly Netflix:", only_netflix)
print("Only Amazon Prime:", only_amazon)
print("Both Services:", both)


#Q2(OR)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = {
    'Student': ['S1', 'S2', 'S3', 'S4', 'S5',
                'S6', 'S7', 'S8', 'S9', 'S10'],
    'Attendance': [90, 85, 30, 35, 80, 25, 95, 40, 75, 20],
    'Marks': [85, 80, 35, 40, 78, 30, 92, 45, 75, 25]
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)

X = df[['Attendance', 'Marks']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nClustered Student Dataset:")
print(df)

centroids = scaler.inverse_transform(kmeans.cluster_centers_)

plt.figure(figsize=(8, 6))

for cluster in sorted(df['Cluster'].unique()):
    subset = df[df['Cluster'] == cluster]
    plt.scatter(
        subset['Attendance'],
        subset['Marks'],
        label=f'Cluster {cluster}'
    )

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    color='red',
    marker='X',
    s=200,
    label='Centroids'
)

plt.xlabel('Attendance (%)')
plt.ylabel('Marks')
plt.title('K-Means Clustering of Students (K=3)')
plt.legend()
plt.grid(True)
plt.show()
