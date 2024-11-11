from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Assume `loan_data` is a DataFrame loaded from the provided CSV file
loan_data = pd.read_csv('Loan_Data.csv') 
# Step 1: Data Preprocessing (Example)

X = loan_data.drop(columns=['default'])  # Features
y = loan_data['default']  # Target


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Training - Using Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# (Optional) Use Random Forest for comparison
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Evaluate the models
log_reg_acc = accuracy_score(y_test, model.predict(X_test))
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"Logistic Regression Accuracy: {log_reg_acc}")
print(f"Random Forest Accuracy: {rf_acc}")


# Step 6: Define function to estimate expected loss
def calculate_expected_loss(borrower_details, loan_amount, model, recovery_rate=0.1):
    """
    Calculate the expected loss on a loan.

    Parameters:
    - borrower_details: dict with borrower's information (income, outstanding loans, etc.)
    - loan_amount: float, total amount of the loan
    - model: trained model to predict PD
    - recovery_rate: assumed recovery rate on the loan

    Returns:
    - expected_loss: calculated expected loss on the loan
    """
    borrower_df = pd.DataFrame([borrower_details])
    pd_estimate = model.predict_proba(borrower_df)[:, 1][0]  # PD is the probability of default
    expected_loss = pd_estimate * (1 - recovery_rate) * loan_amount
    return expected_loss



# Example usage:
borrower_info = {
    'customer_id': 12345,
    'credit_lines_outstanding': 0,
    'loan_amt_outstanding': 20000,
    'total_debt_outstanding': 15000,
    'income': 50000,
    'years_employed': 5,
    'fico_score': 700,

    # Add other fields to match the model's expected features exactly
}

loan_amount = 10000  # Example loan amount
expected_loss = calculate_expected_loss(borrower_info, loan_amount, model)
print(f"Expected Loss: {expected_loss}")
