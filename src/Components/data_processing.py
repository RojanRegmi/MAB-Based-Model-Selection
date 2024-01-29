import sys
sys.path.append('..') # This is the path setting on my computer, modify this according to your need

import numpy as np
import pandas as pd

def data_process(file_path, down_rate=5, window_size=50, step=1):

    """
        This function is used to downsample and create a sliding window.

        Parameters:
        file_path: Path to the CSV File
        down_rate: The downsampling rate, which determines hwo many data points are averaged to create a single data point.
        window_size: The size of the sliding windows
        step: the step size betweeen consecutive windows

        Returns:

        windowed_data: np.ndarray of the windowed data of the time-series. Dimension: down_df.index * window_size * df.columns
    """

    df = pd.read_csv(file_path)

    #Downsampling
    down_df=df.groupby(np.arange(len(df.index)) // down_rate).mean()

    #Create Sliding Windows
    windowed_data = down_df.values[np.arange(window_size)[None, :] + np.arange(down_df.shape[0] - window_size, step=step)[:, None]]

    return windowed_data

