from typing import Tuple
import numpy as np
import pytest


@pytest.fixture(scope="module")
def random_data_binary_classifier() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create random data for binary classifier tests.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The simulated data. y_true, y_score
    """
    # create some data
    n_samples = 1000
    y_true = np.random.choice(
        [0, 1], n_samples, p=[0.4, 0.6]
    )  # the true class labels 0 or 1, with class imbalance 40:60

    y_score = np.zeros(y_true.shape)  # a model's probability of class 1 predictions
    y_score[y_true == 1] = np.random.beta(1, 0.6, y_score[y_true == 1].shape)
    y_score[y_true == 0] = np.random.beta(0.5, 1, y_score[y_true == 0].shape)
    return y_true, y_score


def random_data_multiclass_classifier(
    num_classes: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create random data for binary classifier tests.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The simulated data. y_true_one_hot, y_pred
    """
    class_labels = np.arange(num_classes)
    class_probs = np.random.random(num_classes)
    class_probs = class_probs / class_probs.sum()  # normalize
    # True labels
    y_true = np.random.choice(class_labels, p=class_probs, size=1000)
    # one hot encoding
    y_true_one_hot = np.eye(num_classes)[y_true]

    # Predicted labels
    y_pred = np.ones(y_true_one_hot.shape)

    # parameters for Beta distribution for each label (a0,b0 for class 0, a1,b1 for class 1)
    a0, b0 = [0.1, 0.6, 0.3, 0.4, 2] * 10, [0.4, 1.2, 0.8, 1, 5] * 10
    a1, b1 = [0.9, 0.8, 0.9, 1.2, 5] * 10, [0.4, 0.1, 0.5, 0.3, 2] * 10

    # iterate through all the columns/labels and create a beta distribution for each label
    for i in range(y_pred.shape[1]):
        y = y_pred[:, i]
        y_t = y_true_one_hot[:, i]
        y[y_t == 0] = np.random.beta(a0[i], b0[i], size=y[y_t == 0].shape)
        y[y_t == 1] = np.random.beta(a1[i], b1[i], size=y[y_t == 1].shape)

    return y_true_one_hot, y_pred
