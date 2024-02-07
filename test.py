import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Example DataFrame
data = {
    'Category': ['A', 'B', 'C', 'D'],
    'Group1': [10, 15, 20, 25],  # Values for Group 1
    'Group2': [12, 17, 22, 27],  # Values for Group 2
}

df = pd.DataFrame(data)

# Melt the DataFrame to long format
df_melted = df.melt(id_vars='Category', var_name='Group', value_name='Value')
# Create the grouped barplot
sns.barplot(data=df_melted, x='Category', y='Value', hue='Group')
plt.show()
