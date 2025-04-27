import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =====================
# Preprocessing
# =====================

# Load data
data_file_name = '../../datasets/mall_customers.csv'
df = pd.read_csv(data_file_name)

# Column headers
columns = df.columns

# Check for null values
null_vals = dict(df.isnull().sum())
for item in null_vals:
    if (null_vals[item] != 0):
        print('Null values present in column: ', item)

# Plot histograms

plt.figure(1, figsize=(15, 6))
n = 0
for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.histplot(df[x], bins=15)
    plt.title('Histogram of {}'.format(x))
plt.show()


# =========================
# Perform live clustering
# =========================
print('Implement k means')
