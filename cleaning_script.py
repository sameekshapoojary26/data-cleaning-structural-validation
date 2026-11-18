import pandas as pd
import numpy as np
import re

# Load the original dataset
df = pd.read_csv("customer_churn_raw_v2.csv")

print("Original Dataset Shape:", df.shape)

# 1. Standardize column names
df.columns = [
    re.sub(r'[^a-zA-Z0-9]+', '', col.strip()).strip('').lower()
    for col in df.columns
]

# 2. Remove duplicate records
before = len(df)
df = df.drop_duplicates()
print("Duplicates removed:", before - len(df))

# 3. Handle missing values
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

# 4. Clean text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )

# 5. Convert numeric-looking columns
for col in df.columns:
    if col != "customer_id":
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() == df[col].notna().sum():
            df[col] = converted

# 6. Validate data types
print("\nData Types:")
print(df.dtypes)

# 7. Detect possible numeric outliers using IQR
print("\nOutlier counts:")
for col in df.select_dtypes(include=np.number).columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    count = ((df[col] < q1 - 1.5 * iqr) |
             (df[col] > q3 + 1.5 * iqr)).sum()
    print(col, ":", count)

# 8. Export cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)

# 9. Save cleaning report
with open("cleaning_report.txt", "w") as report:
    report.write("DATA CLEANING REPORT\n")
    report.write("===================\n")
    report.write(f"Original rows: {before}\n")
    report.write(f"Final rows: {len(df)}\n")
    report.write(f"Duplicates removed: {before - len(df)}\n")
    report.write(f"Missing values remaining: {df.isnull().sum().sum()}\n")
    report.write(f"Columns: {len(df.columns)}\n")

print("\nCleaning completed successfully!")
print("Cleaned dataset saved as cleaned_dataset.csv")
