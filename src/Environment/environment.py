import abc
import numpy as np
import tensorflow as tf
from typing import List, Optional, Tuple

from tf_agents.agents import tf_agent
from tf_agents.drivers import driver
from tf_agents.environments import py_environment, tf_environment
from tf_agents.policies import tf_policy
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

import pickle

import sys
sys.path.append('..')
sys.path.append('../..')

from src.logger import logging
from src.exception import CustomException
from src.Components.feature_extractor import FeatureExtractor
from src.Components.data_processing import data_process

from ..utils import *

nest = tf.nest

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

    def __init__(self, time_series: pd.DataFrame, list_thresholds: List[float], list_gtruth: List[float]):

        super().__init__(list_thresholds, list_gtruth)

        self.time_series = time_series
        self.subsequences = data_process(time_series)

        self.list_thresholds = list_thresholds
        self.gtruth = list_gtruth

        self.len_data = len(time_series)

        self.features_obj = FeatureExtractor()

        self.action_list = []

        action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum = 1, name='Select Models')
        observation_spec = array_spec.BoundedArraySpec(shape=(1, 159), dtype=np.float64, name='observation') # shape changed from (1, 318) to (1, 159)

        super(MyModelSelectionEnv, self).__init__(observation_spec, action_spec)

    def _reset(self):

        self.pointer = 0
        self.done = False

        starter = self._feature_extractor(self.subsequences[0])

        return ts.restart(starter)
    

    def _feature_extractor(self, subseq):

        return self.features_obj.feature_extractor_data(subseq)
    
    def _step(self, action):

        reward = self._apply_action(action)
        
        self.pointer += 1

        if self.pointer >= self.len_data:
            self.done = True
        else:
            self.done = False

        if not self.done:
            return ts.transition(self._observe(), reward)
            #return ts.transition(self.subsequences[self.pointer], reward)
        else:
           return ts.termination(self._observe(), reward)
           #return ts.termination(self.subsequences[self.pointer], reward)

                
    def _observe(self):

        self._observation = self._feature_extractor(self.subsequences[self.pointer])
        return self._observation
    
    
    def _apply_action(self, action):

        if action == 0:
            model = pickle.load(open(f'../saved_models/iforest_dodgers_v2.sav','rb'))
            feats = self.subsequences[self.pointer].reshape(-1,1)
            score = model.decision_function(feats)

            label_list = [1 if i < self.list_thresholds[0] else 0 for i in score]
            label_np = np.array(label_list)

        elif action == 1:
            model = pickle.load(open(f'../saved_models/osvm_dodgers_v2.sav', 'rb'))
            feats = self.subsequences[self.pointer].reshape(-1,1)
            score = model.decision_function(feats)

            label_list = [1 if i > self.list_thresholds[1] else 0 for i in score]
            label_np = np.array(label_list)

        reward = self._reward_function(label_np)

        self.action_list.append(action)
        
        return reward
    
    def _reward_function(self, label_np):

        gtruth_split = self.gtruth[self.pointer:self.pointer+50]
        gtruth_np = np.array(gtruth_split)

        tp = 0
        fn = 0
        fp = 0
        tn = 0

        for i, j in zip(gtruth_split, label_np):
           
           if i==1 and j==1:   # If the model predicts anomaly correctly - True Positive (TP)
              tp +=1
              
           elif i==1 and j==0: # If the model predicts 0 normal incorrectly - False Negative (FN)
              fn +=1
        
           elif i==0 and j==1: # If the model predicts 1 anomaly incorrectly - False Positive (FP)
              fp +=1
        
           elif i ==0 and j==0: # If the model predicts 0 normal correctly - True Negative (TN)
              tn +=1

        
        reward = (1 * tp + (-1.5) * fn + (-0.5) * fp + 0.1 * tn) / len(gtruth_split)

        return reward