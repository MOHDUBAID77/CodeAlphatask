import numpy as np
import pandas as pd

np.random.seed(42)
n = 3000

income = np.random.randint(18000, 200000, n)
debt = np.random.randint(1000, 120000, n)
credit_age = np.random.randint(1, 25, n)
late_payments = np.random.poisson(1.5, n)
total_payments = np.random.randint(12, 180, n)
credit_utilization = np.clip(np.random.normal(0.45, 0.22, n), 0.02, 0.98)
loan_count = np.random.randint(0, 8, n)
savings = np.random.randint(2000, 250000, n)

debt_to_income = debt / income
payment_ratio = 1 - (late_payments / np.maximum(total_payments, 1))

score = (
    0.9 * (income / income.max())
    - 0.8 * debt_to_income
    + 0.7 * payment_ratio
    + 0.5 * (credit_age / credit_age.max())
    - 0.7 * credit_utilization
    + 0.25 * (savings / savings.max())
    - 0.1 * (loan_count / 8)
    + np.random.normal(0, 0.15, n)
)

creditworthy = (score > np.median(score)).astype(int)

df = pd.DataFrame({
    "income": income,
    "debt": debt,
    "credit_age_years": credit_age,
    "late_payments": late_payments,
    "total_payments": total_payments,
    "credit_utilization": credit_utilization,
    "loan_count": loan_count,
    "savings": savings,
    "creditworthy": creditworthy
})

df.to_csv("credit_data.csv", index=False)
print("Dataset created: credit_data.csv")
print(df.head())
