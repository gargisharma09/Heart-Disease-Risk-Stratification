import pandas as pd
from ucimlrepo import fetch_ucirepo

# fetch dataset (Heart Disease dataset id is 45)
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

# Combine into a single dataframe
df = pd.concat([X, y], axis=1)

# The UCI dataset typically has some missing values represented as NaN here.
# Let's save it directly to CSV. We will handle missing values in the preprocessing step.
# Also, the target column might be called 'num', let's rename it to 'target'
df.rename(columns={'num': 'target'}, inplace=True)

# Save to data/heart.csv
df.to_csv('data/heart.csv', index=False)
print("Successfully downloaded and saved the UCI Heart Disease dataset to data/heart.csv")
print(f"Dataset shape: {df.shape}")
