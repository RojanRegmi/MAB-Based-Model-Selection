import os
import pandas as pd
import numpy as np
from glob import glob

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

def train_test_anomaly(data: pd.DataFrame, contamination= 0.11, test_size = 0.3, random_state=42):

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

    np.random.seed(random_state)


    np.random.shuffle(normal_shuffle)
    normal_shuffle_train = normal_shuffle[0:train_length]
    normal_shuffle_test = normal_shuffle[-test_length_normal:]

    np.random.shuffle(anomaly_shuffle)

    train_data = data.loc[normal_shuffle_train]
    test_normal = data.loc[normal_shuffle_test]
    test_anomaly = data.loc[anomaly_shuffle]

    test_data = pd.concat((test_normal, test_anomaly))

    test_shuffled = test_data.sample(frac=1).reset_index(drop=True)

    return train_data, test_shuffled

def raw_thresholds(raw_scores, contamination=0.1):
    
    '''raw_scores: each 1D numpy array, the raw anomaly scores'''
    return np.sort(raw_scores)[int(len(raw_scores)*(1-contamination))]

def concatenate_csv_files(directory):
    csv_files = glob(directory)
    
    dataframes = []
    
    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)
    
    # Concatenate all DataFrames into a single DataFrame
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    
    return concatenated_df










    