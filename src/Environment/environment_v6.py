import abc
import numpy as np
import tensorflow as tf
from typing import List
from glob import glob
from numpy.typing import NDArray

from tf_agents.agents import tf_agent
from tf_agents.drivers import driver
from tf_agents.environments import py_environment, tf_environment
from tf_agents.policies import tf_policy
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.bandits.environments import bandit_py_environment

import pickle
import random
import math
import gin

import sys
sys.path.append('..')
sys.path.append('../..')

from src.logger import logging
from src.exception import CustomException
from src.Components.feature_extractor import FeatureExtractor
from src.Components.data_processing import data_process
from src.utils import *
from tf_agents.bandits.environments import bandit_py_environment

from ..utils import *

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


@gin.configurable
class MyModelSelectionEnv(bandit_py_environment.BanditPyEnvironment):

    """
        TF-Agents Bandit Environment for Model Selection.

        Initialization Parameters:

        time_series: The time series to be analysed.
        list_thresholds: The list of thresholds for the the different models
        list_gtruth: The labeled anomaly ground truth
        subsequence: time series features extracted for the different subsequences
        list_predicted_score: The list of predicted raw anomaly scores from all the different models
        list_predicted_label: The list of predicted labels for all the different models
        window_size: size of the sliding window for the subsequences
        batch_size: batch size for training the agent


    """

    def __init__(self, time_series: pd.DataFrame, list_thresholds: List[float], list_gtruth: List[float], subsequence: List[pd.DataFrame], list_predicted_score: List[List[float]], list_predicted_label: List[List[float]], window_size = 20, batch_size = 8):

        self.time_series = time_series
        self.subsequences = subsequence
        self._batch_size = batch_size
        self.window_size = window_size

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

        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=(self._num_actions - 1), name='Models')
        observation_spec = array_spec.ArraySpec(shape=(self.len_feats,), dtype=np.float64, name='observation')

        self._time_step_spec = ts.time_step_spec(observation_spec)
        self._observation = np.zeros((self._batch_size, self.len_feats))


        self._time_step_spec = ts.time_step_spec(observation_spec)

        super(MyModelSelectionEnv, self).__init__(observation_spec, self._action_spec)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batched(self):
        return True

    """ def _reset(self):

        self.pointer = 0
        self.pointer2 = math.ceil((self.window_size - 1) / 2)
        self.pred_list = []
        self.done = False

        starter = self._feature_extractor(self.subsequences, self.pointer)

        return ts.restart(starter)"""
    
    
    """def _step(self, action):
        
        
        reward, lab, scr = self._apply_action(action)
        self.pointer += 1
        self.pointer2 += 1

        if self.pointer >= self.len_data:
            self.done = True
        else:
            self.done = False

        print(f'Step: {self.pointer}, reward: {reward}')

        if not self.done:
            return ts.transition(self._observe(), reward)
            
        else:
           return ts.termination(self._observe(), reward)"""
           

                
    def _observe(self):

        """returns an array containing batch_size number of features"""

        batched_observations = self._feature_extractor(self.subsequences, self.pointer)

        return batched_observations
    
    
    def _apply_action(self, action: NDArray[np.int_]):

        """
        Applies the action received from agent and calls the reward function to give reward based on it.

        The raw anomaly scores are also thresholded based on the base model.

        The scores, labels and actions are saved in object-specific attributes.

        """

        scr = [sublist[self.pointer2:self.pointer2+self.batch_size] for sublist in self.pred_scr_list]
        labl = [sublist[self.pointer2:self.pointer2+self.batch_size] for sublist in self.labels]

        i = 0
        pointer = self.pointer2

        reward_list = []
        label_list = []
        score_list = []

        for act in action:
            anomaly_scr = scr[act][i]
            label_value = labl[act][i]

            reward = self._reward_function(label_value, pointer)
            lbl = self._thresholder(anomaly_scr, act)
            
            score_list.append(anomaly_scr)
            reward_list.append(reward)
            label_list.append(lbl)

            i += 1
            pointer+=1
        
        reward = np.array(reward_list)
        label = np.array(label_list)
        anomaly_score = np.array(score_list)

        self.score_list.extend(anomaly_score)
        self.label_list.extend(label)
        self.action_list.extend(action)

        self.pointer2 = self.pointer2 + self._batch_size
        return reward
    
    def _reward_function(self, label_value, pointer):

        if self.gtruth[pointer]==1: # If the ground truth is 1 anomaly
            if label_value==1: # If the model predicts 1 anomaly correctly - True Positive (TP)
                reward = 4
            else: # If the model predicts 0 normal incorrectly - False Negative (FN)
                reward = -4.5
        else: # If the ground truth is 0 normal
            if label_value==1: # If the model predicts 1 anomaly incorrectly - False Positive (FP)
                reward = -6
            else: # If the model predicts 0 normal correctly - True Negative (TN)
                reward = 0.3

        return reward
    
    def _thresholder(self, score, action):
       
       label = 1 if score >= self.list_thresholds[action] else 0

       return label
    
    def _environment_values(self):

        return self.score_list, self.label_list, self.action_list
    
    def _feature_extractor(self, subseq, pointer):

        batches = [seq.reshape(-1,) for seq in subseq[pointer:pointer+self._batch_size]]
        self.pointer += 1

        return np.array(batches, dtype=np.float64)

    
    def time_step_spec(self) -> ts.TimeStep:
       return super().time_step_spec()
    
    def action_spec(self):
       return super().action_spec()
    
    def get_info(self):
       return super().get_info()
    
    def compute_optimal_reward(self):
        pass