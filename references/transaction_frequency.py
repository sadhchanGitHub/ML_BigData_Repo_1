import pandas as pd
import numpy as np
import multiprocessing as mp
from pandas import Timedelta

# Function to count transactions within the last hour
def count_transactions_within_last_hour(group):
    frequencies = []
    for time in group.index:
        # Ensure the time is a datetime object
        if not isinstance(time, pd.Timestamp):
            time = pd.to_datetime(time)

        # Count the number of transactions within the last hour
        count = group[(group.index >= (time - pd.Timedelta(hours=1))) & (group.index <= time)].shape[0]
        frequencies.append(count)
    return frequencies

# Function to process each chunk of the dataframe
def process_chunk(df_chunk):
    # Ensure the index is datetime for the chunk
    df_chunk.index = pd.to_datetime(df_chunk.index)
    return df_chunk.groupby('CreditCardNumber').apply(count_transactions_within_last_hour)
