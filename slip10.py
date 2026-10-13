#Q1
import matplotlib.pyplot as plt
transport = ["Train", "Bicycle", "Bike", "Car", "Bus"]
employees = [220, 60, 70, 80, 40]

plt.pie(
    employees,
    labels=transport,
    autopct='%1.1f%%',
    colors=['red', 'blue', 'green', 'orange', 'purple'],
    startangle=90
)
plt.title("Employee Transportation Preferences")
plt.show()


#Q2(OR)


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [
    ['Video Lecture', 'Quiz'],
    ['Video Lecture', 'Assignment'],
    ['Quiz', 'Discussion Forum'],
    ['Video Lecture', 'Quiz', 'Assignment'],
    ['Assignment', 'Discussion Forum'],
    ['Video Lecture', 'Certificate'],
    ['Quiz', 'Certificate'],
    ['Video Lecture', 'Quiz'],
    ['Assignment', 'Certificate'],
    ['Video Lecture', 'Quiz', 'Certificate']
]

te = TransactionEncoder()
encoded = te.fit(transactions).transform(transactions)
df = pd.DataFrame(encoded, columns=te.columns_)

print("Transaction Dataset:")
print(df)

for support in [0.3, 0.4]:
    print("\nMinimum Support:", support)

    frequent_itemsets = apriori(
        df, min_support=support, use_colnames=True
    )
    print("\nFrequent Itemsets:")
    print(frequent_itemsets)

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.5
    )

    print("\nAssociation Rules:")
    print(rules[
        ['antecedents', 'consequents', 'support',
         'confidence', 'lift']
    ])
