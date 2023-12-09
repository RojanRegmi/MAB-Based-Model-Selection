import os
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from glob import glob

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    
    def __init__(self):
        
        self.ingestion_config=DataIngestionConfig()
        self.dataset_indexer = get_dataset_dictionary()
    
    
    def get_dataset_dictionary(self):
        
        dataset_indexer = defaultdict()
        walker = list(os.walk('../../datasets'))
        
        index = walker[0][1]

        for i in range(1,len(walker)):
            dataset_indexer[index[i-1]] = walker[i][-1]
        
        return dataset_indexer
        
                
        
    def initiate_data_ingestion(self, dataset_name: str):
        
        logging.info("Entered the data ingestion method")
        
        try:
            
            data_list = []
            
            for file in self.dataset_indexer[f'{dataset_name}']:
                df = pd.read_csv(f'../../datasets/{dataset_name}/{file}')
                data_list.append(df)
            
            logging.info(f'Read the dataset {dataset_name} as list of dataframe')        
        except:
            
            pass
    