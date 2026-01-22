import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

print("Starting feature encoding...")

# FORCE working directory to project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

print("Current working directory:")
print(os.getcwd())

# Load cleaned dataset
input_path = os.path.join(BASE_DIR, "cleaned_churn_data.csv")
df = pd.read_csv(input_path)

print("\nDataset loaded successfully")
print(df.head())

# Encode categorical columns
encoder = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

print("\nCategorical features encoded")

# SAVE FILE (FORCED PATH)
output_path = os.path.join(BASE_DIR, "encoded_churn_data.csv")
df.to_csv(output_path, index=False)

print("\nEncoded dataset saved successfully at:")
print(output_path)
print("Feature encoding completed successfully!")
