import os
import pandas as pd
import numpy as np
from glob import glob
from statsmodels.tsa.stattools import acf 
from scipy.signal import argrelextrema

def find_subdirectory(target_subdir, parent_dir):

    """
       target_subdir: The file that you are looking for
       parent_dir: The Directory where you want to search

       returns
       dataset_dir: This is all the directories inside parent_dir that have the file target_subdir
    """

    target_dir = []
    target_dir.append(target_subdir)
    dataset_dir = []
        
    for root, dirs, files in os.walk(parent_dir):
            
        if target_dir == files:
                
            dataset_dir.append(root)
        
    return dataset_dir

def train_test_anomaly(data: pd.DataFrame, contamination= 0.111, test_size = 0.3, shuffle=True):

    """
      This function is for creating a train test split for anomaly algorithm training. Make sure to label the anomaly column as 'anomaly'.

      In unsupervised model training, most of the time you'll need to train your model in normal data and test it in data mixed with anomaly.

      This function creates a train test split where the training data is just normal data and the test data is contaminated with anomalies


    """

    train_size = 1 - test_size
    total_length = len(data)
    anomaly_length = int(total_length * contamination)
    normal_length = total_length - anomaly_length

    train_length = int(train_size * total_length)
    test_length_normal = normal_length - train_length
    normal_data_index = data[data['anomaly'] == 0].index.values
    anomaly_index = data.drop(index=normal_data_index).index.values

    normal_shuffle = normal_data_index
    anomaly_shuffle = anomaly_index

    np.random.seed(42)

    if shuffle:
        np.random.shuffle(normal_shuffle)
        np.random.shuffle(anomaly_shuffle)


    normal_train = normal_shuffle[0:train_length]
    normal_test = normal_shuffle[-test_length_normal:]

    
    train_data = data.loc[normal_train]
    test_normal = data.loc[normal_test]
    test_anomaly = data.loc[anomaly_shuffle]

    test_data = pd.concat((test_normal, test_anomaly))

    test_shuffled = test_data.sample(frac=1).reset_index(drop=True)

    return train_data, test_shuffled

def raw_thresholds(raw_scores, contamination=0.1):
    # Adapted from RLMSAD 
    '''raw_scores: each 1D numpy array, the raw anomaly scores'''
    return np.sort(raw_scores)[int(len(raw_scores)*(1-contamination))]

def find_length(data):

    # Adapted from TSB-UAD
    if len(data.shape)>1:
        return 0
    data = data[:min(20000, len(data))]
    
    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    
    
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 125
        return local_max[max_local_max]+base
    except:
        return 125


def detect_anomalies(data, window_size, min_threshold, max_threshold, threshold_factor, overlap):
    anomalies = []
    for i in range(0, len(data) - window_size + 1, overlap):
        window = data[i:i+window_size]
        mean = np.mean(window)
        std_dev = np.std(window)
        threshold = mean + threshold_factor * std_dev
        
        if min_threshold <= threshold <= max_threshold and data[i] > threshold:
            anomalies.append(1)
        else:
            anomalies.append(0)
    
    return anomalies














    