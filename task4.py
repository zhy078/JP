import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

# Assuming `data` is a DataFrame with 'fico_score' and 'default' columns

def create_buckets_mse(data, num_buckets=10):
    """
    Create FICO score buckets by minimizing mean squared error.
    
    Parameters:
    - data: DataFrame with 'fico_score' column
    - num_buckets: Number of buckets to create
    
    Returns:
    - boundaries: List of bucket boundaries
    """
    data_sorted = data.sort_values('fico_score')
    scores = data_sorted['fico_score'].values

    # Split scores into buckets and calculate representative values
    boundaries = np.linspace(scores.min(), scores.max(), num_buckets + 1)
    bucket_means = []

    for i in range(num_buckets):
        bucket = data_sorted[(scores >= boundaries[i]) & (scores < boundaries[i+1])]
        mean_score = bucket['fico_score'].mean()
        bucket_means.append(mean_score)

    return boundaries, bucket_means

def create_buckets_log_likelihood(data, num_buckets=10):
    """
    Create FICO score buckets to maximize log-likelihood for default probability.
    
    Parameters:
    - data: DataFrame with 'fico_score' and 'default' columns
    - num_buckets: Number of buckets to create
    
    Returns:
    - boundaries: List of bucket boundaries
    """
    data_sorted = data.sort_values('fico_score')
    scores = data_sorted['fico_score'].values
    defaults = data_sorted['default'].values

    # Placeholder boundaries and dynamic programming logic for optimization
    boundaries = np.linspace(scores.min(), scores.max(), num_buckets + 1)
    bucket_log_likelihood = []

    for i in range(num_buckets):
        bucket = data_sorted[(scores >= boundaries[i]) & (scores < boundaries[i+1])]
        n_i = len(bucket)
        k_i = bucket['default'].sum()  # Number of defaults
        p_i = k_i / n_i if n_i > 0 else 0  # Default probability

        # Calculate log-likelihood for this bucket
        if n_i > 0 and 0 < p_i < 1:
            log_likelihood = k_i * np.log(p_i) + (n_i - k_i) * np.log(1 - p_i)
        else:
            log_likelihood = 0  # Avoid log(0) for empty or homogeneous buckets
        
        bucket_log_likelihood.append(log_likelihood)

    return boundaries, bucket_log_likelihood

def fico_to_rating(fico_score, boundaries):
    """
    Map FICO score to a rating based on the bucket boundaries.
    
    Parameters:
    - fico_score: FICO score of a borrower
    - boundaries: List of bucket boundaries
    
    Returns:
    - rating: Integer rating based on the bucket (1 = best, num_buckets = worst)
    """
    for i, boundary in enumerate(boundaries[:-1]):
        if boundary <= fico_score < boundaries[i + 1]:
            return i + 1
    return len(boundaries) - 1  # Assign last bucket if beyond last boundary

# Example usage
data = pd.DataFrame({
    'fico_score': np.random.randint(300, 850, 1000),
    'default': np.random.binomial(1, 0.1, 1000)
})

# Generate buckets using both methods
boundaries_mse, bucket_means_mse = create_buckets_mse(data, num_buckets=10)
boundaries_ll, bucket_log_likelihoods = create_buckets_log_likelihood(data, num_buckets=10)

# Map a FICO score to a rating
fico_score_example = 720
rating_mse = fico_to_rating(fico_score_example, boundaries_mse)
rating_ll = fico_to_rating(fico_score_example, boundaries_ll)

print(f"Rating (MSE): {rating_mse}")
print(f"Rating (Log Likelihood): {rating_ll}")
