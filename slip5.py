##Q1
import pandas as pd
from sklearn.preprocessing import LabelEncoder
data = {
    'Customer_ID': [101, 102, 103, 104, 105],
    'ID': [1, 2, 3, 4, 5],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'City': ['Pune', 'Mumbai', 'Delhi', 'Pune', 'Mumbai'],
    'Membership_Type': [
        'Gold', 'Silver', 'Platinum', 'Gold', 'Silver'
    ]
}
customer_DataFrame = pd.DataFrame(data)
print("DataFrame Before Encoding:")
print(customer_DataFrame)
encoder = LabelEncoder()
for col in ['Gender', 'City', 'Membership_Type']:
    customer_DataFrame[col] = encoder.fit_transform(
        customer_DataFrame[col]
    )
print("\nDataFrame After Encoding:")
print(customer_DataFrame)


#Q2

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

data = {
    'Customer_ID': [1, 2, 3, 4, 5, 6, 7, 8],
    'Netflix': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes'],
    'Amazon_Prime': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No']
}

df = pd.DataFrame(data)
print("Customer DataFrame:")
print(df)

netflix_only = len(df[
    (df['Netflix'] == 'Yes') &
    (df['Amazon_Prime'] == 'No')
])

amazon_only = len(df[
    (df['Netflix'] == 'No') &
    (df['Amazon_Prime'] == 'Yes')
])

both = len(df[
    (df['Netflix'] == 'Yes') &
    (df['Amazon_Prime'] == 'Yes')
])

neither = len(df[
    (df['Netflix'] == 'No') &
    (df['Amazon_Prime'] == 'No')
])

print("\nNetflix Only:", netflix_only)
print("Amazon Prime Only:", amazon_only)
print("Both Services:", both)
print("Neither Service:", neither)

plt.figure(figsize=(7, 5))
venn2(
    subsets=(netflix_only, amazon_only, both),
    set_labels=('Netflix', 'Amazon Prime'),
    set_colors=('red', 'blue'),
    alpha=0.6
)
plt.title("Netflix and Amazon Prime Subscribers")
plt.show()

labels = ['Netflix Only', 'Amazon Prime Only', 'Both Services']
sizes = [netflix_only, amazon_only, both]
colors = ['red', 'blue', 'green']

plt.figure(figsize=(7, 5))
plt.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90
)
plt.title("Streaming Service Customer Distribution")
plt.show()
