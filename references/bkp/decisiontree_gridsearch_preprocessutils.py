import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import multiprocessing as mp

# Function for parallelized one-hot encoding of categorical variables
def parallel_one_hot_encode(df, num_partitions=None):
    if num_partitions is None:
        num_partitions = mp.cpu_count()  # Use all available CPU cores

    # Split the dataframe into chunks
    df_split = np.array_split(df, num_partitions)

    # Create a multiprocessing Pool
    with mp.Pool(num_partitions) as pool:
        # Apply pd.get_dummies in parallel
        result = pool.map(pd.get_dummies, df_split)

    # Concatenate the results back into a single dataframe
    return pd.concat(result)

# Function for preprocessing: one-hot encoding and imputing missing values
def preprocess_data(df, selected_features):
    y = df['is_fraud']
    X = df[selected_features]

    # Parallelize one-hot encoding of categorical variables
    X_encoded = parallel_one_hot_encode(X)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_encoded)

    return X_imputed, y
