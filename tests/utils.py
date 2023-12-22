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