from pathlib import Path
from typing import Tuple
import numpy as np
import pytest

from plotsandgraphs import pipeline


from .utils import random_data_binary_classifier

TEST_RESULTS_PATH = Path("tests/test_results/pipeline")

def test_binary_classification_pipeline(random_data_binary_classifier):
    """
    Test binary classification pipeline.

    Parameters
    ----------
    random_data_binary_classifier : Tuple[np.ndarray, np.ndarray]
        The simulated data.
    """
    y_true, y_score = random_data_binary_classifier
    pipeline.binary_classifier(y_true, y_score, save_fig_path=TEST_RESULTS_PATH / "pipeline.png")