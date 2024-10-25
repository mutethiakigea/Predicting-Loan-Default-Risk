# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Data ingestion
df = pd.read_csv("notebooks/Loan_default.csv")
print(df.head())

# Data Transformation
# mapping education categorical data to numerics
education_mapping = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}

df["Education"] = df["Education"].map(education_mapping)

# mapping employmenType categorical data to numerics
employmentType_mapping = {
    "Unemployed": 1,
    "Self-employed": 2,
    "Part-time": 3,
    "Full-time": 4,
}
df["EmploymentType"] = df["EmploymentType"].map(employmentType_mapping)

maritalstatus_mapping = {"Single": 1, "Married": 2, "Divorced": 3}

df["MaritalStatus"] = df["MaritalStatus"].map(maritalstatus_mapping)

hasmortgage_mapping = {"Yes": 0, "No": 1}

df["HasMortgage"] = df["HasMortgage"].map(hasmortgage_mapping)

loanpurpose_mapping = {"Auto": 1, "Business": 2, "Education": 3, "Home": 4, "Other": 5}

df["LoanPurpose"] = df["LoanPurpose"].map(loanpurpose_mapping)

hasdependent_mapping = {"Yes": 1, "No": 0}
df["HasDependents"] = df["HasDependents"].map(hasdependent_mapping)
hascosigner_mapping = {"Yes": 1, "No": 0}

df["HasCoSigner"] = df["HasCoSigner"].map(hascosigner_mapping)

# Convert numeric columns to the appropriate data type if necessary
numeric_columns = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
    "Default",
]

df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

desc_stats = df.describe()
desc_stats

df = df.drop("LoanID", axis=1)
