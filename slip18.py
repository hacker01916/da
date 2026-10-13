#Q1
import pandas as pd

data = {
    'Emp_ID': [101, 102, 103, 104, 105, 105],
    'Age': [25, 30,  None, 45, 35, 35],
    'Dept': ['IT', 'HR', 'Sales', 'IT', 'HR', 'HR'],
    'Years_Experience': [2, 5, 8, 20, 7, 7],
    'Performance_Rating': [4, 3, 5, 4, 4, 4]
}

employee_data = pd.DataFrame(data)

print("Original DataFrame:")
print(employee_data)

employee_data['Age'] = employee_data['Age'].fillna(
    employee_data['Age'].median()
)

employee_data = employee_data.drop_duplicates()

columns = ['Age', 'Years_Experience', 'Performance_Rating']

for col in columns:
    Q1 = employee_data[col].quantile(0.25)
    Q3 = employee_data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = employee_data[
        (employee_data[col] < lower) |
        (employee_data[col] > upper)
    ]

    print("\nOutliers in", col)
    print(outliers)

for col in columns:
    Q1 = employee_data[col].quantile(0.25)
    Q3 = employee_data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    employee_data = employee_data[
        (employee_data[col] >= lower) &
        (employee_data[col] <= upper)
    ]

print("\nCleaned Employee DataFrame:")
print(employee_data)



#Q2

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import squarify


df = pd.read_csv('world_cities.csv')

df = df.dropna(
    subset=['city', 'country', 'latitude', 'longitude', 'population']
)

print("World Cities Dataset:")
print(df.head())

fig = px.scatter_geo(
    df,
    lat='latitude',
    lon='longitude',
    hover_name='city',
    hover_data=['country', 'population'],
    size='population',
    projection='natural earth',
    title='World Cities Population Map',
    size_max=30
)

fig.show()

plt.figure(figsize=(10, 6))

squarify.plot(
    sizes=df['population'],
    label=df['city'],
    alpha=0.8
)

plt.title('Population Distribution of World Cities')
plt.axis('off')
plt.tight_layout()
plt.show()