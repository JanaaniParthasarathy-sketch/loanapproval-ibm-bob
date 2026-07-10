"""
Script to generate a realistic Loan Approval dataset for training/demo purposes.
Run this once: python generate_dataset.py
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 614

loan_ids = [f"LP{str(i).zfill(6)}" for i in range(1, n + 1)]
gender = np.random.choice(["Male", "Female"], n, p=[0.81, 0.19])
married = np.random.choice(["Yes", "No"], n, p=[0.65, 0.35])
dependents = np.random.choice(["0", "1", "2", "3+"], n, p=[0.57, 0.17, 0.16, 0.10])
education = np.random.choice(["Graduate", "Not Graduate"], n, p=[0.78, 0.22])
self_employed = np.random.choice(["Yes", "No"], n, p=[0.14, 0.86])
credit_history = np.random.choice([1.0, 0.0], n, p=[0.84, 0.16])

# Generate income based on education
applicant_income = np.where(
    education == "Graduate",
    np.random.lognormal(mean=8.3, sigma=0.7, size=n).astype(int),
    np.random.lognormal(mean=7.8, sigma=0.6, size=n).astype(int)
)
applicant_income = np.clip(applicant_income, 1800, 81000)

coapplicant_income = np.random.choice(
    [0, np.random.lognormal(mean=7.5, sigma=0.8, size=n)],
    n,
    p=[0.4, 0.6]
)
coapplicant_income = np.where(
    np.random.random(n) < 0.4,
    0,
    np.random.lognormal(mean=7.5, sigma=0.8, size=n)
).astype(int)

loan_amount = (applicant_income * np.random.uniform(0.8, 3.0, n) / 1000).astype(int)
loan_amount = np.clip(loan_amount, 28, 700)

loan_amount_term = np.random.choice([120, 180, 240, 300, 360, 480], n, p=[0.05, 0.07, 0.04, 0.06, 0.68, 0.10])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], n, p=[0.38, 0.38, 0.24])

# Loan approval logic based on real-world factors
def determine_approval(row):
    score = 0
    if row['Credit_History'] == 1.0:
        score += 40
    if row['Education'] == 'Graduate':
        score += 15
    total_income = row['ApplicantIncome'] + row['CoapplicantIncome']
    if total_income > 5000:
        score += 15
    elif total_income > 3000:
        score += 8
    emi = (row['LoanAmount'] * 1000) / max(row['Loan_Amount_Term'], 1)
    dti = emi / max(total_income, 1)
    if dti < 0.3:
        score += 15
    elif dti < 0.5:
        score += 5
    if row['Property_Area'] == 'Semiurban':
        score += 8
    elif row['Property_Area'] == 'Urban':
        score += 5
    if row['Married'] == 'Yes':
        score += 5
    noise = np.random.normal(0, 8)
    return 'Y' if (score + noise) >= 50 else 'N'

df = pd.DataFrame({
    'Loan_ID': loan_ids,
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_amount_term,
    'Credit_History': credit_history,
    'Property_Area': property_area
})

df['Loan_Status'] = df.apply(determine_approval, axis=1)

# Introduce some realistic missing values (5%)
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
    mask = np.random.random(n) < 0.04
    df.loc[mask, col] = np.nan

os.makedirs('data', exist_ok=True)
df.to_csv('data/loan_approval.csv', index=False)
print(f"Dataset created: data/loan_approval.csv ({len(df)} rows)")
print(f"Approval Rate: {(df['Loan_Status']=='Y').mean()*100:.1f}%")
print(df['Loan_Status'].value_counts())
