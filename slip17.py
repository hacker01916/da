#Q1

import pandas as pd
student_details = pd.DataFrame({
    'Student_ID': [1, 2, 3, 4],
    'Name': ['Amit', 'Priya', 'Rahul', 'Sneha'],
    'Age': [20, 21, 19, 22]
})

academic_records = pd.DataFrame({
    'Student_ID': [1, 2, 3, 4],
    'Course': ['BSc', 'BSc', 'BSc', 'BSc'],
    'Marks': [85, 90, 78, 88]
})

print("Student Details:")
print(student_details)

print("\nAcademic Records:")
print(academic_records)

integrated_df = pd.merge(
    student_details,
    academic_records,
    on='Student_ID',
    how='inner'
)

print("\nIntegrated DataFrame:")
print(integrated_df)


#Q2

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

data = {
    'City': [
        'Pune', 'Mumbai', 'Delhi', 'Chennai',
        'Bengaluru', 'Hyderabad'
    ],
    'Population': [
        7000000, 12500000, 11000000,
        7000000, 8500000, 6800000
    ],
    'GDP': [80, 310, 293, 78, 110, 75],
    'Literacy_Rate': [91, 90, 89, 90, 88, 83]
}

df = pd.DataFrame(data)

print("City Dataset:")
print(df)

fig1 = px.treemap(
    df,
    path=['City'],
    values='Population',
    title='City-wise Population Distribution'
)

fig1.show()

fig2 = px.scatter_3d(
    df,
    x='Population',
    y='GDP',
    z='Literacy_Rate',
    color='City',
    text='City',
    title='3D Scatter Plot of City Data'
)

fig2.update_layout(
    scene=dict(
        xaxis_title='Population',
        yaxis_title='GDP',
        zaxis_title='Literacy Rate'
    )
)

fig2.show()



