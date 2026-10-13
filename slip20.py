##Q1
import pandas as pd
from sklearn.preprocessing import StandardScaler
data = {
    'Age': [22, 25, 28, 30, 35],
    'Salary': [20000, 25000, 30000, 35000, 40000],
    'Experience': [1, 2, 4, 5, 8],
    'Working_Hours': [6, 7, 8, 8, 9],
    'Performance_Score': [60, 70, 75, 80, 90]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

print("\nStandardized DataFrame:")
print(df_scaled.round(2))




##Q2(OR)

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

print("Transaction Dataset:")
print(df)

for support in [0.2, 0.3]:
    print("\nMinimum Support:", support)

    frequent = apriori(
        df, min_support=support, use_colnames=True
    )

    print("\nFrequent Itemsets:")
    print(frequent)

    if len(frequent) > 0 and frequent['itemsets'].map(len).max() >= 2:
        rules = association_rules(
            frequent, metric="confidence",
            min_threshold=0.0
        )
        print("\nAssociation Rules:")
        print(rules[
            ['antecedents', 'consequents',
             'support', 'confidence']
        ])
    else:
        print("No association rules found.")

