import os
import pandas as pd
import numpy as np

def find_subdirectory(target_subdir):

    parent_dir = os.pardir
        
    for root, dirs, files in os.walk(parent_dir):
            
        if target_subdir in dirs:
                
            dataset_dir = os.path.join(root, target_subdir)
            all_files = files
        
    return dataset_dir, all_files

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









    