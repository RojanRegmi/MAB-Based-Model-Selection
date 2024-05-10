import abc
import numpy as np
import tensorflow as tf
from typing import List
from glob import glob

from tf_agents.agents import tf_agent
from tf_agents.drivers import driver
from tf_agents.environments import py_environment, tf_environment
from tf_agents.policies import tf_policy
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

import pickle
import random
import math

import sys
sys.path.append('..')
sys.path.append('../..')

from src.logger import logging
from src.exception import CustomException
from src.Components.feature_extractor import FeatureExtractor
from src.Components.data_processing import data_process
from src.utils import *

from ..utils import *

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class BanditPyEnvironment(py_environment.PyEnvironment):

  def __init__(self, observation_spec, action_spec):
    self._observation_spec = observation_spec
    self._action_spec = action_spec
    super(BanditPyEnvironment, self).__init__()

  # Helper functions.
  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _empty_observation(self):
    return tf.nest.map_structure(lambda x: np.zeros(x.shape, x.dtype),
                                 self.observation_spec())

  # These two functions below should not be overridden by subclasses.
  def _reset(self):
    """Returns a time step containing an observation."""
    return ts.restart(self._observe(), batch_size=self.batch_size)

  def _step(self, action):
    """Returns a time step containing the reward for the action taken."""
    reward = self._apply_action(action)
    return ts.termination(self._observe(), reward)

  # These two functions below are to be implemented in subclasses.
  @abc.abstractmethod
  def _observe(self):
    """Returns an observation."""

  @abc.abstractmethod
  def _apply_action(self, action):
    """Applies `action` to the Environment and returns the corresponding reward.
    """


class MyModelSelectionEnv(BanditPyEnvironment):

    """
        TF-Agents Bandit Environment for Model Selection.

        Initialization Parameters:

        time_series: The time series to be analysed.
        list_thresholds: The list of thresholds for the the different models
        list_gtruth: The labeled anomaly ground truth
        list_predicted_score: The list of predictions from all the different models

    """

    def __init__(self, time_series: pd.DataFrame, list_thresholds: List[float], list_gtruth: List[float], subsequence: List[pd.DataFrame], list_predicted_score: List[List[float]], list_predicted_label: List[List[float]], filepath):

        self.time_series = time_series
        self.window_size = 25 # find_length(time_series[['value']].to_numpy())
        self.subsequences = subsequence
        # self._batch_size = batch_size

        self.filepath = filepath
        self.score_list = []
        self.label_list = []
        self.action_list = []


        self.list_thresholds = list_thresholds
        self.gtruth = list_gtruth
        self.pointer = 0
        self.pred_scr_list = list_predicted_score
        self.pointer2 = math.ceil((self.window_size - 1) / 2)
        self.labels = list_predicted_label

        self.len_data = len(time_series)
        self.len_feats = len(subsequence[0])
        self._num_actions = len(list_predicted_label)

        # self.features_obj = FeatureExtractor()
        # self.observation_len = len(self._feature_extractor(self.subsequences[0]))

        # self.models = self._load_models()

        action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=(self._num_actions - 1), name='Models')
        observation_spec = array_spec.ArraySpec(shape=(self.len_feats,), dtype=np.float64, name='observation')

        self._time_step_spec = ts.time_step_spec(observation_spec)

        super(MyModelSelectionEnv, self).__init__(observation_spec, action_spec)

    """def _load_models(self):
       
       model1 = pickle.load(open(f'../saved_models/iforest_dodgers_v3.sav','rb'))
       # model2 = pickle.load(open(f'../saved_models/osvm_dodgers_v2.sav', 'rb'))
       model3 =  pickle.load(open(f'../saved_models/clof_dodgers_v2.sav', 'rb')) 
       model4 = pickle.load(open(f'../saved_models/copod_dodgers_v2.sav', 'rb'))

       return [model1, model3, model4]
    """

    def _reset(self):

        self.pointer = 0
        self.pointer2 = math.ceil((self.window_size - 1) / 2)
        self.pred_list = []
        self.done = False

        starter = self._feature_extractor(self.subsequences[0])

        return ts.restart(starter)
    

    def _feature_extractor(self, subseq):
        
        return subseq.reshape(-1,)
    
    def _step(self, action):
        
        
        reward, lab, scr = self._apply_action(action)
        self.pointer += 1
        self.pointer2 += 1

        self.score_list.append(scr)
        self.label_list.append(lab)
        self.action_list.apeend(action)

        if self.pointer % 10000 == 0:
           filename = self.filepath + f'env_output'
           with open(self.filepath, 'wb') as file:
              pickle.dum
              

        if self.pointer >= self.len_data:
            self.done = True
        else:
            self.done = False

        print(f'Step: {self.pointer}, label: {lab}')

        if not self.done:
            return ts.transition(self._observe(), reward)
            
        else:
           return ts.termination(self._observe(), reward)
           

                
    def _observe(self):

        return self._feature_extractor(self.subsequences[self.pointer])
    
    
    def _apply_action(self, action):

        anomaly_score = self.pred_scr_list[action][self.pointer2]
        label_value = self.labels[action][self.pointer2]
        reward = self._reward_function(label_value)
        
        return reward, label_value, anomaly_score
    
    def _reward_function(self, label_value):

        if self.gtruth[self.pointer2]==1: # If the ground truth is 1 anomaly
            if label_value==1: # If the model predicts 1 anomaly correctly - True Positive (TP)
                reward = 8
            else: # If the model predicts 0 normal incorrectly - False Negative (FN)
                reward = -4.5
        else: # If the ground truth is 0 normal
            if label_value==1: # If the model predicts 1 anomaly incorrectly - False Positive (FP)
                reward = -10
            else: # If the model predicts 0 normal correctly - True Negative (TN)
                reward = 0.1

        return reward
    
    def time_step_spec(self) -> ts.TimeStep:
       return super().time_step_spec()
    
    def action_spec(self):
       return super().action_spec()
    
    def get_info(self):
       return super().get_info()
    
    def compute_optimal_reward(self):
        pass