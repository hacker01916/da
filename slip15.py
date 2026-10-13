#Q1
import pandas as pd

# Create DataFrame
data = {
    'Person_ID': [1, 2, 3, 4, 5, 6, 7, 8],
    'Age': [18, 22, 25, 29, 35, 40, 45, 50]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

df['Age_Group'] = pd.qcut(
    df['Age'],
    q=4,
    labels=['Group1', 'Group2', 'Group3', 'Group4']
)

print("\nDiscretized DataFrame:")
print(df)



#Q2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

data = {
    'City': ['Pune', 'Mumbai', 'Nashik', 'Nagpur', 'Thane'],
    'Area': [331, 603, 259, 228, 147],
    'Average_Temperature': [28, 30, 27, 32, 29],
    'Rainfall': [700, 2200, 800, 1100, 2500],
    'Population_Density': [9000, 20000, 6000, 11000, 15000]
}

df = pd.DataFrame(data)

print("City DataFrame:")
print(df)

plt.figure(figsize=(8, 5))

sns.heatmap(
    df[['Area', 'Average_Temperature', 'Rainfall',
        'Population_Density']].corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title('Correlation Heatmap of City Data')
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    df['Area'],
    df['Average_Temperature'],
    df['Rainfall'],
    c=df['Population_Density'],
    cmap='viridis',
    s=100
)

ax.set_xlabel('Area (sq. km)')
ax.set_ylabel('Average Temperature (°C)')
ax.set_zlabel('Rainfall (mm)')
ax.set_title('3D Scatter Plot of City Data')

plt.colorbar(scatter, label='Population Density')
plt.tight_layout()
plt.show()



#Q2(OR)

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [
    ['Basmati Rice', 'Toor Dal', 'Turmeric Powder', 'Cooking Oil'],
    ['Atta', 'Sugar', 'Tea', 'Milk'],
    ['Rice', 'Moong Dal', 'Salt', 'Cooking Oil'],
    ['Wheat Flour', 'Chickpeas', 'Turmeric Powder', 'Cumin Seeds'],
    ['Poha', 'Peanuts', 'Onions', 'Cooking Oil'],
    ['Basmati Rice', 'Paneer', 'Garam Masala', 'Tomatoes'],
    ['Atta', 'Potatoes', 'Onions', 'Cooking Oil']
]

te = TransactionEncoder()
encoded = te.fit(transactions).transform(transactions)

df = pd.DataFrame(encoded, columns=te.columns_)

print("Transaction Data:")
print(df)

frequent_03 = apriori(
    df, min_support=0.3, use_colnames=True
)

print("\nFrequent Itemsets (Support = 0.3):")
print(frequent_03)

if not frequent_03.empty:
    rules_03 = association_rules(
        frequent_03, metric='confidence', min_threshold=0.1
    )
    print("\nAssociation Rules (Support = 0.3):")
    print(rules_03)

frequent_04 = apriori(
    df, min_support=0.4, use_colnames=True
)

print("\nFrequent Itemsets (Support = 0.4):")
print(frequent_04)

if not frequent_04.empty:
    rules_04 = association_rules(
        frequent_04, metric='confidence', min_threshold=0.1
    )
    print("\nAssociation Rules (Support = 0.4):")
    print(rules_04)
