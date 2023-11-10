import pytest
from pathlib import Path
import numpy as np
import pytest
import plotsandgraphs.binary_classifier as binary

TEST_RESULTS_PATH = Path(r"tests\test_results")


@pytest.fixture(scope="module")
def random_data_binary_classifier():
    # create some data
    n_samples = 1000
    y_true = np.random.choice([0,1], n_samples, p=[0.4, 0.6])   # the true class labels 0 or 1, with class imbalance 40:60

    y_prob = np.zeros(y_true.shape)   # a model's probability of class 1 predictions
    y_prob[y_true==1] = np.random.beta(1, 0.6, y_prob[y_true==1].shape)
    y_prob[y_true==0] = np.random.beta(0.5, 1, y_prob[y_true==0].shape)
    return y_true, y_prob

# Test histogram plot
def test_hist_plot(random_data_binary_classifier):
    y_true, y_prob = random_data_binary_classifier
    print(TEST_RESULTS_PATH)
    binary.plot_y_prob_histogram(y_prob, save_fig_path=TEST_RESULTS_PATH/'histogram.png')
    binary.plot_y_prob_histogram(y_prob, y_true, save_fig_path=TEST_RESULTS_PATH/'histogram_2_classes.png')
    return

# test roc curve
def test_roc_curve(random_data_binary_classifier):
    y_true, y_prob = random_data_binary_classifier
    binary.plot_roc_curve(y_true, y_prob, save_fig_path=TEST_RESULTS_PATH/'roc_curve.png')
    return

# test precision recall curve
def test_pr_curve(random_data_binary_classifier):
    y_true, y_prob = random_data_binary_classifier
    binary.plot_pr_curve(y_true, y_prob, save_fig_path=TEST_RESULTS_PATH/'pr_curve.png')
    return

# test confusion matrix
def test_confusion_matrix(random_data_binary_classifier):
    y_true, y_prob = random_data_binary_classifier
    binary.plot_confusion_matrix(y_true, y_prob, save_fig_path=TEST_RESULTS_PATH/'confusion_matrix.png')
    return

# test classification report
def test_classification_report(random_data_binary_classifier):
    y_true, y_prob = random_data_binary_classifier
    binary.plot_classification_report(y_true, y_prob, save_fig_path=TEST_RESULTS_PATH/'classification_report.png')
    return

# test calibration curve
def test_calibration_curve(random_data_binary_classifier):
    y_true, y_prob = random_data_binary_classifier
    binary.plot_calibration_curve(y_prob, y_true, save_fig_path=TEST_RESULTS_PATH/'calibration_curve.png')
    return

    
    
    
