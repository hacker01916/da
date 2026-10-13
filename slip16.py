#Q1
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = {
    'Study_Hours': [2, 4, 5, 6, 8],
    'Attendance': [60, 70, 75, 85, 95],
    'Assignment_Score': [50, 60, 70, 80, 90],
    'Exam_Score': [45, 55, 65, 78, 92]
}

df = pd.DataFrame(data)

print("Student Performance Data:")
print(df)
corr = df.corr()

print("\nCorrelation Matrix:")
print(corr)

plt.figure(figsize=(8, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')

plt.title('Correlation Heatmap of Student Performance')
plt.tight_layout()
plt.show()



#Q2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

data = {
    'Temperature': [20, 25, 30, 35, 40],
    'Humidity': [40, 50, 60, 70, 80],
    'Wind_Speed': [5, 10, 15, 20, 25],
    'Pressure': [1000, 1005, 1010, 1015, 1020]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

minmax = MinMaxScaler()
minmax_data = pd.DataFrame(
    minmax.fit_transform(df),
    columns=df.columns
)

print("\nMin-Max Scaling:")
print(minmax_data)
n
standard = StandardScaler()
standard_data = pd.DataFrame(
    standard.fit_transform(df),
    columns=df.columns
)

print("\nStandardization:")
print(standard_data)

normalizer = Normalizer()
normalized_data = pd.DataFrame(
    normalizer.fit_transform(df),
    columns=df.columns
)
print("\nNormalization:")
print(normalized_data)


#Q2(OR)
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [
    ['Library', 'Seminar'],
    ['Workshop', 'Project'],
    ['Library', 'Assignment'],
    ['Seminar', 'Workshop'],
    ['Assignment', 'Seminar'],
    ['Workshop', 'Assignment'],
    ['Project', 'Library'],
    ['Library', 'Seminar', 'Project'],
    ['Library', 'Seminar', 'Workshop']
]

te = TransactionEncoder()
encoded = te.fit(transactions).transform(transactions)

df = pd.DataFrame(encoded, columns=te.columns_)

print("Student Activity Dataset:")
print(df)

frequent_02 = apriori(
    df, min_support=0.2, use_colnames=True
)

print("\nFrequent Itemsets (Support = 0.2):")
print(frequent_02)

if not frequent_02.empty:
    rules_02 = association_rules(
        frequent_02, metric='confidence', min_threshold=0.1
    )
    print("\nAssociation Rules (Support = 0.2):")
    print(rules_02)

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