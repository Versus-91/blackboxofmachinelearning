import pandas as pd

# Create a sample DataFrame
data = {'Category': ['A', 'B', 'A', 'C', 'B']}
df = pd.DataFrame(data)

# Specify the column to one-hot encode
column_to_encode = 'Category'

# One-hot encode the specified column
one_hot = pd.get_dummies(df[column_to_encode], prefix=column_to_encode)

# Drop the original column from the DataFrame
df = df.drop(column_to_encode, axis=1)

# Concatenate the one-hot encoded DataFrame with the original DataFrame
df_encoded = pd.concat([df, one_hot], axis=1)

# Display the resulting DataFrame with one-hot encoding
print(df_encoded)
