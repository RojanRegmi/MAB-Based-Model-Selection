from sklearn.ensemble import IsolationForest

import numpy as np
import pandas as pd
import os
import pickle

from glob import glob

file_name = '../datasets/MGAB/*.out'

file_list = glob(file_name)

dfs = []

for file in file_list:

    df = pd.read_csv(file)
    dfs.append(dfs)

combined_df = pd.concat(dfs)
combined_np = combined_df.to_numpy(dtype=np.float32)

np.random.seed('42')

IF = IsolationForest(n_estimators=100, contamination=0.03)