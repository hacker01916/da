#Q1

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("weather.csv")

print("Weather Dataset:")
print(df.head())

data = df[['Temperature', 'Humidity',
           'Rainfall', 'Wind Speed']]

corr = data.corr()

print("\nCorrelation Matrix:")
print(corr)

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Weather Parameters Correlation Heat Map")
plt.show()

"""
Temperature,Humidity,Rainfall,Wind Speed
30,65,10,12
32,60,5,15
28,80,25,8
25,85,40,10
35,55,0,20
27,75,30,9
31,62,8,14
29,70,15,11
33,58,3,18
26,82,35,7
 """

#Q2

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    'Attendance': [85, 90, 78, 60, 65, 55, 40, 45, 35, 95],
    'Marks':      [88, 92, 80, 65, 60, 58, 35, 40, 30, 96]
}

df = pd.DataFrame(data)
print("Student Dataset:")
print(df)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(
    df[['Attendance', 'Marks']]
)

print("\nStudents with Cluster Labels:")
print(df)

centroids = kmeans.cluster_centers_
print("\nCluster Centroids:")
print(centroids)

plt.figure(figsize=(8, 6))

for cluster in range(3):
    points = df[df['Cluster'] == cluster]
    plt.scatter(
        points['Attendance'],
        points['Marks'],
        label=f'Cluster {cluster}'
    )

plt.scatter(
    centroids[:, 0], centroids[:, 1],
    marker='X', s=250, color='red',
    label='Centroids'
)

plt.xlabel('Attendance (%)')
plt.ylabel('Marks')
plt.title('K-Means Clustering of Students (K=3)')
plt.legend()
plt.grid(True)
plt.show()


