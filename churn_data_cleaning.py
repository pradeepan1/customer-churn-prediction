import pandas as pd

print("Starting data cleaning...")

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

print("\nDataset loaded successfully")
print(df.head())

# Drop customerID
df.drop("customerID", axis=1, inplace=True)
print("\ncustomerID column removed")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nMissing values before handling:")
print(df.isnull().sum())

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("cleaned_churn_data.csv", index=False)

print("\nCleaned dataset saved as cleaned_churn_data.csv")
print("Data cleaning completed successfully!")
