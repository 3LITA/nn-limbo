import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    num_samples = len(ground_truth)
    if len(prediction) != num_samples:
        raise ValueError

    true_positives = np.sum((prediction == 1) & (ground_truth == 1))

    precision = true_positives / np.sum(prediction == 1) if np.sum(prediction == 1) != 0 else 1
    recall = true_positives / np.sum(ground_truth == 1) if np.sum(ground_truth == 1) != 0 else 1
    accuracy = np.sum(prediction == ground_truth) / num_samples
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    num_samples = len(ground_truth)
    if len(prediction) != num_samples:
        raise ValueError

    return np.sum(prediction == ground_truth) / num_samples  
