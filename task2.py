import pandas as pd
from datetime import timedelta

def estimate_price(date):
    """Returns the estimated price for a given date using a trend model."""
    days_since_start = (pd.to_datetime(date) - data.index[0]).days
    return model.predict(np.array([[days_since_start]]))[0]

def calculate_contract_value(injection_dates, withdrawal_dates, injection_price, withdrawal_price, 
                             injection_rate, withdrawal_rate, max_volume, storage_cost):
    """
    Calculates the value of a natural gas storage contract.
    
    Parameters:
    - injection_dates: List of dates when gas is injected.
    - withdrawal_dates: List of dates when gas is withdrawn.
    - injection_price: Price at which gas can be purchased.
    - withdrawal_price: Price at which gas can be sold.
    - injection_rate: Rate of gas injection per day.
    - withdrawal_rate: Rate of gas withdrawal per day.
    - max_volume: Maximum storage capacity.
    - storage_cost: Daily storage cost per unit of gas.
    
    Returns:
    - contract_value: The net value of the contract.
    """
    
    # Initialize storage and cash flow tracking variables
    storage_volume = 0
    cash_flows = 0
    
    # Calculate total injection cost and update storage volume
    for date in injection_dates:
        daily_injection = min(injection_rate, max_volume - storage_volume)  # Ensure we do not exceed capacity
        price_on_date = injection_price if isinstance(injection_price, (int, float)) else estimate_price(date)
        cash_flows -= daily_injection * price_on_date
        storage_volume += daily_injection

    # Calculate total withdrawal revenue and update storage volume
    for date in withdrawal_dates:
        daily_withdrawal = min(withdrawal_rate, storage_volume)  # Ensure we do not withdraw more than available
        price_on_date = withdrawal_price if isinstance(withdrawal_price, (int, float)) else estimate_price(date)
        cash_flows += daily_withdrawal * price_on_date
        storage_volume -= daily_withdrawal

    # Calculate total storage cost based on the days gas is stored
    total_days = len(set(injection_dates + withdrawal_dates))
    total_storage_cost = total_days * storage_cost * min(storage_volume, max_volume)
    cash_flows -= total_storage_cost

    return cash_flows

# Example usage with sample inputs
injection_dates = ['2024-01-01', '2024-02-01', '2024-03-01']
withdrawal_dates = ['2024-09-01', '2024-10-01', '2024-11-01']
injection_rate = 1000  # units per day
withdrawal_rate = 1000  # units per day
max_volume = 5000  # max storage capacity
storage_cost = 0.1  # daily storage cost per unit

# Test with sample prices or estimated prices
injection_price = 10  # fixed price or replace with None for dynamic pricing
withdrawal_price = 12  # fixed price or replace with None for dynamic pricing

# Calculate contract value
contract_value = calculate_contract_value(injection_dates, withdrawal_dates, injection_price, 
                                          withdrawal_price, injection_rate, withdrawal_rate, 
                                          max_volume, storage_cost)

print(f"The calculated value of the contract is: {contract_value}")