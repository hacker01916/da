#Q1
import matplotlib.pyplot as plt

courses = ['BCA', 'BBA', 'B.Com', 'B.Sc', 'BA']
students = [60, 45, 75, 50, 35]

colors = ['red', 'blue', 'green', 'orange', 'purple']

plt.bar(courses, students, color=colors)

plt.title('Number of Students Enrolled in Each Course')
plt.xlabel('Courses')
plt.ylabel('Number of Students')

for i, value in enumerate(students):
    plt.text(i, value + 1, str(value), ha='center')

plt.show()



#Q2
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df1 = pd.DataFrame({
    'Brand': ['Samsung', 'Apple', 'Redmi', 'OnePlus', 'Vivo'],
    'Model': ['A15', 'iPhone15', 'Note13', 'Nord3', 'V29'],
    'Price': [15000, 70000, 18000, 30000, 25000],
    'RAM': [4, 8, 6, 8, 8]
})

df2 = pd.DataFrame({
    'Model': ['A15', 'iPhone15', 'Note13', 'Nord3', 'V29'],
    'Storage': [128, 256, 128, 256, 128],
    'Battery_Capacity': [5000, 3349, 5000, 5000, 4600],
    'Operating_System': [
        'Android', 'iOS', 'Android', 'Android', 'Android'
    ]
})

print("DataFrame 1:")
print(df1)

print("\nDataFrame 2:")
print(df2)

merged_df = pd.merge(df1, df2, on='Model', how='inner')

print("\nMerged DataFrame:")
print(merged_df)

transformed_df = merged_df.copy()

le = LabelEncoder()
transformed_df['Brand'] = le.fit_transform(
    transformed_df['Brand']
)

print("\nAfter Label Encoding:")
print(transformed_df)

transformed_df = pd.get_dummies(
    transformed_df,
    columns=['Operating_System'],
    dtype=int
)

print("\nFinal Transformed DataFrame:")
print(transformed_df)



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

print("Original Transactions:")
for i, items in enumerate(transactions, start=1):
    print(i, items)

te = TransactionEncoder()
encoded = te.fit(transactions).transform(transactions)

df = pd.DataFrame(encoded, columns=te.columns_)

print("\nEncoded Transaction Dataset:")
print(df)

for support in [0.3, 0.4]:
    print("\nMinimum Support =", support)

    frequent_itemsets = apriori(
        df,
        min_support=support,
        use_colnames=True
    )

    print("\nFrequent Itemsets:")
    print(frequent_itemsets)

    pairs = frequent_itemsets[
        frequent_itemsets['itemsets'].apply(len) >= 2
    ]

    if not pairs.empty:
        rules = association_rules(
            frequent_itemsets,
            metric='confidence',
            min_threshold=0.0
        )

        print("\nAssociation Rules:")
        print(rules[
            ['antecedents', 'consequents',
             'support', 'confidence', 'lift']
        ])
    else:
        print("No association rules found at this support.")
