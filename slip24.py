#Q1

import pandas as pd
data = {
    'Book_ID': [101, 102, 103, 104, 105,
                106, 107, 108, 109, 110],
    'Book_Price': [100, 150, 200, 250, 300,
                   350, 400, 450, 500, 550]
}

book_records = pd.DataFrame(data)
print("Original Data:")
print(book_records)

book_records['Price_Bin'] = pd.cut(
    book_records['Book_Price'],
    bins=3,
    labels=['Low', 'Medium', 'High']
)

print("\nDiscretized Data:")
print(book_records)


#Q2


import pandas as pd
import plotly.express as px

data = {
    'city': [
        'Mumbai', 'Delhi', 'Tokyo', 'New York',
        'London', 'Paris', 'Beijing', 'Sydney'
    ],
    'country': [
        'India', 'India', 'Japan', 'USA',
        'UK', 'France', 'China', 'Australia'
    ],
    'latitude': [
        19.0760, 28.6139, 35.6762, 40.7128,
        51.5072, 48.8566, 39.9042, -33.8688
    ],
    'longitude': [
        72.8777, 77.2090, 139.6503, -74.0060,
        -0.1276, 2.3522, 116.4074, 151.2093
    ],
    'population': [
        12400000, 11000000, 14000000, 8800000,
        9000000, 2100000, 21500000, 5300000
    ]
}

df = pd.DataFrame(data)

print("World Cities Dataset:")
print(df)

fig1 = px.scatter_geo(
    df,
    lat='latitude',
    lon='longitude',
    locations=None,
    hover_name='city',
    hover_data=['country', 'population'],
    size='population',
    projection='natural earth',
    title='World Cities Population Map',
    size_max=35
)

fig1.show()

fig2 = px.treemap(
    df,
    path=['country', 'city'],
    values='population',
    title='Population Distribution of World Cities',
    color='population',
    color_continuous_scale='Blues'
)

fig2.show()

