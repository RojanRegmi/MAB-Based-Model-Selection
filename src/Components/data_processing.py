import sys
sys.path.append('..') # This is the path setting on my computer, modify this according to your need

import numpy as np
import pandas as pd
from sklearn import preprocessing

def data_process(file_path, down_rate=5, window_size=50, step=1):

    df = pd.read_csv(file_path)

    #Downsampling
    down_df=df.groupby(np.arange(len(df.index)) // down_rate).mean()

    #Create Sliding Windows
    windowed_data = down_df.values[np.arange(window_size)[None, :] + np.arange(down_df.shape[0] - window_size, step=step)[:, None]]

    return windowed_data

