import numpy as np
import tsfel 
import pandas as pd
import os
from glob import glob
from collections import defaultdict
from src.logger import logging
from src.exception import CustomException

class FeatureExtractor:
    
    def __init__(self, fs=100):
        
        self.dataset_indexer = self.get_dataset_dictionary()
        self.parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.fs = fs
        self.feature_list = None
        
    
    def get_dataset_dictionary(self):
        
        
        logging.info('Dataset Indexer Initiated')
        dataset_indexer = defaultdict()
        walker = list(os.walk('../datasets'))
        
        index = walker[0][1]

        for i in range(1,len(walker)):
            dataset_indexer[index[i-1]] = walker[i][-1]
            
        logging.info('Dataset Indexer returned')
        return dataset_indexer
    
    def get_data_list(self, dataset: str):
        
        filenames = self.dataset_indexer[dataset]
        dataset_dir = os.path.join(self.parent_dir, 'datasets', dataset)

        dataset_list = []   

        for file in filenames:
    
            filepath = os.path.join(dataset_dir, file)
            df = pd.read_csv(filepath)
            dataset_list.append(df)
            
        return dataset_list
    
    def find_subdirectory(self, target_subdir):
        
        for root, dirs, files in os.walk(self.parent_dir):
            
            if target_subdir in dirs:
                
                dataset_dir = os.path.join(root, target_subdir)
                all_files = files
        
        return dataset_dir, all_files
        
        
    
    def feature_extractor(self, dataset: str, feature_type: str = None):
        
        logging.info(f'Feature Extraction Started of type {feature_type}')
        
        cfg_file = tsfel.get_features_by_domain(feature_type)
        
        dataset_list = self.get_data_list(dataset)
        
        feature_list = []
        
        for data in dataset_list:
            feature_df = tsfel.time_series_features_extractor(cfg_file, data, fs=self.fs)
            feature_list.append(feature_df)
            
        
        self.feature_list = feature_list
        
        return feature_list
    
    def feature_extractor_data(self, data: np.ndarray, feature_type: str = None):
        
        cfg_file = tsfel.get_features_by_domain(feature_type)
        
        feature_df = tsfel.time_series_features_extractor(cfg_file, data, fs=self.fs)

        feature_np = np.round(feature_df.values, decimals=4)

        feature_np = feature_np.reshape((-1,))
        
        return feature_np
        
        