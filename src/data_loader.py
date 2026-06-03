import pandas as pd


def load_data(file_path):
    """Load the NYC bus breakdown and delays dataset."""
    return pd.read_csv(file_path, low_memory=False)
