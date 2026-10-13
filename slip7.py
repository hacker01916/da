##Q1

import matplotlib.pyplot as plt

courses = ["BCA", "BBA", "B.Com", "B.Sc", "BA"]
students = [60, 45, 75, 50, 35]

colors = ["red", "blue", "green", "orange", "purple"]

plt.bar(courses, students, color=colors)
plt.title("Number of Students Enrolled in Different Courses")
plt.xlabel("Courses")
plt.ylabel("Number of Students")

plt.show()



#Q2(B)

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Grocery transaction dataset
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
encoded_data = te.fit(transactions).transform(transactions)

df = pd.DataFrame(encoded_data, columns=te.columns_)
print("Transaction Data:")
print(df)

frequent_03 = apriori(
    df, min_support=0.3, use_colnames=True
)

print("\nFrequent Itemsets (Support = 0.3):")
print(frequent_03)

if not frequent_03.empty:
    rules_03 = association_rules(
        frequent_03, metric="confidence", min_threshold=0
    )
    print("\nAssociation Rules (Support = 0.3):")
    print(rules_03[
        ['antecedents', 'consequents', 'support',
         'confidence', 'lift']
    ])
else:
    print("No frequent itemsets found.")

frequent_04 = apriori(
    df, min_support=0.4, use_colnames=True
)

print("\nFrequent Itemsets (Support = 0.4):")
print(frequent_04)

if not frequent_04.empty:
    rules_04 = association_rules(
        frequent_04, metric="confidence", min_threshold=0
    )
    print("\nAssociation Rules (Support = 0.4):")
    print(rules_04[
        ['antecedents', 'consequents', 'support',
         'confidence', 'lift']
    ])
else:
    print("No frequent itemsets found.")
