
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import pickle
from glob import glob



file_name = '../datasets/Dodgers/*.out'

file_list = glob(file_name)

dfs = []

for file in file_list:

    df = pd.read_csv(file)
    dfs.append(dfs)

"""combined_df = pd.concat(dfs)
combined_np = combined_df.to_numpy(dtype=np.float32)"""

window_data = train_test_split(dfs[0], test_size=0.4, train_size=0.6)

np.random.seed(42)

if_model = IsolationForest(n_estimators=100, contamination=float(11.13), max_samples=window_data.shape[0]/5)

if_model.fit(window_data)

filename = './saved_detectors/iforest_dodgers.sav'

pickle.dump(if_model, open(filename, 'wb'))
