import pandas as pd

def clean(df):
    df = df.copy()
    # remove impossible values
    df = df[(df["person_age"] < 100) & (df["person_emp_length"].fillna(0) < 60)]
    # fill missing values with median
    for col in ["person_emp_length", "loan_int_rate"]:
        df[col] = df[col].fillna(df[col].median())
    return df

def add_features(df):
    df = df.copy()
    df["loan_to_income"] = df["loan_amnt"] / df["person_income"]
    df["income_per_year_employed"] = df["person_income"] / (df["person_emp_length"] + 1)
    df["age_to_credit_hist"] = df["person_age"] / (df["cb_person_cred_hist_length"] + 1)
    return df