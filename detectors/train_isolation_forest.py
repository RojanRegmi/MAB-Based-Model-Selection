
from sklearn.ensemble import IsolationForest

import numpy as np
import pandas as pd
import pickle
from glob import glob

from src.Components.data_processing import data_process


file_name = '../datasets/Dodgers/*.out'

file_list = glob(file_name)

dfs = []

for file in file_list:

    df = pd.read_csv(file)
    dfs.append(dfs)

"""combined_df = pd.concat(dfs)
combined_np = combined_df.to_numpy(dtype=np.float32)"""

window_data = data_process(file_list[0])

np.random.seed(42)

if_model = IsolationForest(n_estimators=100, contamination=float(11.13), max_samples=window_data.shape[0]/5)

if_model.fit(window_data)

filename = './saved_detectors/iforest_dodgers.sav'

pickle.dump(if_model, open(filename, 'wb'))
