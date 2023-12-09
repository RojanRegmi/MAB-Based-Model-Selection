import numpy as np
import tsfel 
import pandas as pd
import os
from glob import glob

class FeatureExtractor:
    
    def __init__(self):
        
        stat_features = extract_features('statitstical')
        temporal_features = extract_features('temporal')
        spectral_features = extract_features('spectral')
        self.dataset_indexer = get_dataset_dictionary()
        
    def extract_features(feature_type):
        
        return tsfel.get_features_by_domain(feature_type)
    
    def get_dataset_dictionary(self):
        
        dataset_indexer = defaultdict()
        walker = list(os.walk('../../datasets'))
        
        index = walker[0][1]

        for i in range(1,len(walker)):
            dataset_indexer[index[i-1]] = walker[i][-1]
        
        return dataset_indexer
        