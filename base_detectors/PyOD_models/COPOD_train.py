# Set current working directory to the main branch of RLMSAD

import sys
sys.path.append('/usr/local/data/elisejzh/Projects/RLMSAD') # This is the path setting on my computer, modify this according to your need

from pyod.models.copod import COPOD
from data_process import *
import pickle

path_normal="./raw_data/SWaT/SWaT_Dataset_Normal_v1.csv"
path_attack="./raw_data/SWaT/SWaT_Dataset_Attack_v0.csv"

windows_normal, windows_attack=data_process_SWaT(path_normal,path_attack)

# Flatten the windows
num=windows_normal.shape[0]
windows_normal=windows_normal.reshape(num,-1)

# COPOD Model
model=COPOD(contamination=0.11,n_jobs=-1)
model.fit(windows_normal)

# save the model
filename = './base_detectors/COPOD_SWaT.sav'
pickle.dump(model, open(filename, 'wb'))
