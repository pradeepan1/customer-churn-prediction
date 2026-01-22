import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

print(df.head())
print(df["Churn"].value_counts())

# Churn vs Contract
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn vs Contract Type")
plt.show()
