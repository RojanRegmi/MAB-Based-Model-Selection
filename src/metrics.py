# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from typing import Optional, List, Union
from sklearn.metrics import ndcg_score, average_precision_score
from scipy.stats import kendalltau, norm
import random

from loguru import logger

np.random.seed(42)
random.seed(42)

def f1_score(predict, actual):
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN

def f1_soft_score(predict, actual):
    # Predict: 1/0
    # Actual: [0,1]
    actual = actual/np.max(actual)
    
    negatives = 1*(actual==0)

    TP = np.sum(predict * actual)  # weighted by actual
    TN = np.sum((1 - predict) * (negatives) )
    FP = np.sum(predict * (negatives))
    FN = np.sum((1 - predict) * actual) # weighted by actual
    precision = TP / (TP + FP*np.mean(actual[actual>0]) + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    
    return f1, precision, recall, TP, TN, FP, FN

def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score < threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

def best_f1_linspace(scores, labels, n_splits, segment_adjust, f1_type='standard'):
    best_threshold = 0
    best_f1 = 0
    thresholds = np.linspace(scores.min(),scores.max(), n_splits)
    
    if np.sum(labels)>0:
        for threshold in thresholds:
            predict = scores>=threshold
            if segment_adjust:
                predict = adjust_predicts(score=scores, label=(labels>0), threshold=None, pred=predict, calc_latency=False)
            if f1_type=='standard':
                f1, *_ = f1_score(predict, labels)
            elif f1_type=='soft':
                f1, *_ = f1_soft_score(predict, labels)

            if f1 > best_f1:
                best_threshold = threshold
                best_f1 = f1
    else:
        best_threshold = scores.max() + 1
        best_f1 = 1
        
    predict = scores>=best_threshold
    if segment_adjust:
        predict = adjust_predicts(score=scores, label=(labels>0), threshold=None, pred=predict, calc_latency=False)
    
    if f1_type=='standard':
        f1, precision, recall, *_ = f1_score(predict, labels)
    elif f1_type=='soft':
        f1, precision, recall, *_ = f1_soft_score(predict, labels)

    return f1, precision, recall, predict, labels, best_threshold

def normalize_scores(scores, interval_size):
    scores_normalized = []
    for score in scores:
        n_intervals = int(np.ceil(len(score)/interval_size))
        score_normalized = []
        for i in range(n_intervals):
            min_timestamp = i*interval_size
            max_timestamp = (i+1)*interval_size
            std = score[:max_timestamp].std()
            score_interval = score[min_timestamp:max_timestamp]/std
            score_normalized.append(score_interval)
        score_normalized =  np.hstack(score_normalized)
        scores_normalized.append(score_normalized)
    return scores_normalized

def adjusted_precision_recall_f1_auc(y_true:np.ndarray, y_scores:np.ndarray, n_splits=750):
    """Function to compute adjusted precision, recall, PR-AUC (average precision) and predictions.
    """
    from sklearn.metrics import auc
    
    thresholds = np.linspace(y_scores.min(),y_scores.max(), n_splits)
    adjusted_precision = np.zeros(thresholds.shape) 
    adjusted_recall = np.zeros(thresholds.shape) 
    adjusted_f1 = np.zeros(thresholds.shape) 

    for i, threshold in enumerate(thresholds):
        y_pred = y_scores>=threshold
        y_pred = adjust_predicts(score=y_scores, label=(y_true>0), threshold=None, pred=y_pred, calc_latency=False)
        adjusted_f1[i], adjusted_precision[i], adjusted_recall[i], *_ = f1_score(y_pred, y_true)

    best_adjusted_f1 = np.max(adjusted_f1)
    best_threshold = thresholds[np.argmax(adjusted_f1)]
    adjusted_prauc = auc(adjusted_recall, adjusted_precision)

    adjusted_y_pred = y_scores>=best_threshold
    adjusted_y_pred = adjust_predicts(score=y_scores, label=(y_true>0), threshold=None, pred=adjusted_y_pred, calc_latency=False)

    return adjusted_precision, adjusted_recall, best_adjusted_f1, adjusted_prauc, adjusted_y_pred


def range_based_precision_recall_f1_auc(y_true: np.ndarray, y_scores: np.ndarray, n_splits=1000, window_size=1000):
    from sklearn.metrics import auc
    import numpy as np
    """Function to compute range-based precision, recall, range-based F1, PR-AUC, and predictions."""
    thresholds = np.linspace(y_scores.min(), y_scores.max(), n_splits)
    range_precision = np.zeros(thresholds.shape)
    range_recall = np.zeros(thresholds.shape)
    range_f1 = np.zeros(thresholds.shape)
    logger.info(f'= y size is: {y_scores.size}')
    for i, threshold in enumerate(thresholds):

        y_pred = y_scores >= threshold
        y_pred = adjust_predicts(score=y_scores, label=(y_true > 0), threshold=None, pred=y_pred, calc_latency=False)
        # print(f'y_pred {y_pred}')
        # Calculating range-based precision, recall, and F1
        for idx in range(len(y_pred)):
            start_idx = max(0, idx - window_size)
            end_idx = min(len(y_pred), idx + window_size + 1)

            TP = np.sum((y_pred[start_idx:end_idx] == 1) & (y_true[start_idx:end_idx] == 1))
            FP = np.sum((y_pred[start_idx:end_idx] == 1) & (y_true[start_idx:end_idx] == 0))
            FN = np.sum((y_pred[start_idx:end_idx] == 0) & (y_true[start_idx:end_idx] == 1))

            precision = TP / (TP + FP + 0.00001)
            recall = TP / (TP + FN + 0.00001)
            f1 = 2 * precision * recall / (precision + recall + 0.00001)

            range_precision[i] += precision
            range_recall[i] += recall
            range_f1[i] += f1

        range_precision[i] /= len(y_pred)
        range_recall[i] /= len(y_pred)
        range_f1[i] /= len(y_pred)

    best_range_f1 = np.max(range_f1)
    best_threshold = thresholds[np.argmax(range_f1)]
    range_prauc = auc(range_recall, range_precision)

    # Adjusting final predictions
    adjusted_y_pred = y_scores >= best_threshold
    adjusted_y_pred = adjust_predicts(score=y_scores, label=(y_true > 0), threshold=None, pred=adjusted_y_pred,
                                      calc_latency=False)

    return range_precision, range_recall, best_range_f1, range_prauc, adjusted_y_pred
