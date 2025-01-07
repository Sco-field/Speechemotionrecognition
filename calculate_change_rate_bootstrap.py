import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Function to calculate the rate of change for a single CSV file
def calculate_rate_of_change(file_path):
    df = pd.read_csv(file_path, header=0, delimiter=",")  # Read the CSV file
    emotion_labels = df.iloc[:, 1]  # Extract the 5th column (emotion labels, 0-based indexing)

    # Binarize the emotion labels: values > 0.5 -> 1, values <= 0.5 -> 0
    emotion_labels = emotion_labels.apply(lambda x: 1 if x > 0.5 else 0)  # Binarize: >0.5 becomes 1, <=0.5 becomes 0

    # Calculate the rate of change based on binarized labels
    change_rate = (emotion_labels.diff().abs()).sum() / (len(emotion_labels) - 1)  # Calculate the change rate
    return change_rate

# Function to calculate the average rate of change and bootstrap confidence interval
def bootstrap_confidence_interval(directory_path, num_samples=5000, alpha=0.05):
    change_rates = []

    # Walk through all files in the directory and subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Check if the file is a CSV and contains 'sorted' in its name
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)  # Get the full file path
                try:
                    rate_of_change = calculate_rate_of_change(file_path)
                    change_rates.append(rate_of_change)  # Append the rate of change
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    if not change_rates:
        return None, None, None  # In case no valid CSV files are found

    # Calculate the mean rate of change
    mean_rate_of_change = np.mean(change_rates)

    # Bootstrap sampling to calculate the 95% confidence interval
    bootstrap_means = []
    for _ in range(num_samples):
        sampled_data = np.random.choice(change_rates, size=len(change_rates), replace=True)
        bootstrap_means.append(np.mean(sampled_data))

    # Calculate the 95% confidence interval (2.5% and 97.5% percentiles)
    lower_bound = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper_bound = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return mean_rate_of_change, lower_bound, upper_bound, change_rates

# Function to calculate p-value using t-test
def calculate_p_value(group1_rates, group2_rates):
    # Perform an independent t-test between two groups
    t_stat, p_value = ttest_ind(group1_rates, group2_rates, equal_var=False)  # Welch's t-test
    return p_value

# Specify the directory to analyze for suicide and non-suicide groups (replace with your actual directories)
suicide_directory = r'suicide'
non_suicide_directory = r'none_suicide'

# Calculate and print the mean and bootstrap confidence interval for both groups
mean_suicide, lower_suicide, upper_suicide, suicide_rates = bootstrap_confidence_interval(suicide_directory)
mean_non_suicide, lower_non_suicide, upper_non_suicide, non_suicide_rates = bootstrap_confidence_interval(non_suicide_directory)

if mean_suicide is not None and mean_non_suicide is not None:
    print(f"Suicide Group: Mean Rate of Change: {mean_suicide}, 95% CI: ({lower_suicide}, {upper_suicide})")
    print(f"Non-Suicide Group: Mean Rate of Change: {mean_non_suicide}, 95% CI: ({lower_non_suicide}, {upper_non_suicide})")

    # Calculate the p-value between the two groups
    p_value = calculate_p_value(suicide_rates, non_suicide_rates)
    print(f"P-value for the difference between suicide and non-suicide groups: {p_value}")
else:
    print("No valid CSV files found or there was an error in processing the files.")
