from pathlib import Path
from typing import Tuple
from itertools import product
import numpy as np
import pytest

# import plotsandgraphs.binary_classifier as binary
import plotsandgraphs.multiclass_classifier as multiclass

TEST_RESULTS_PATH = Path(r"tests\test_results\multi_class_classifier")


# @pytest.fixture(scope="module")
def random_data_multiclass_classifier(
    num_classes: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create random data for binary classifier tests.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The simulated data.
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


# Test histogram plot
def test_hist_plot():
    """
    Test histogram plot.

    Parameters
    ----------
    random_data_binary_classifier : Tuple[np.ndarray, np.ndarray]
        The simulated data.
    """
    for num_classes in [1, 2, 3, 4, 5, 10, 16, 25]:
        y_true, y_prob = random_data_multiclass_classifier(num_classes=num_classes)
        print(TEST_RESULTS_PATH)
        multiclass.plot_y_prob_histogram(
            y_true=y_true,
            y_score=y_prob,
            save_fig_path=TEST_RESULTS_PATH / f"histogram_{num_classes}_classes.png",
        )
        # multiclass.plot_y_prob_histogram(y_prob=y_prob, save_fig_path=TEST_RESULTS_PATH / "histogram_classes.png")


def test_roc_curve():
    """
    Test roc curve.

    Parameters
    ----------
    random_data_binary_classifier : Tuple[np.ndarray, np.ndarray]
        The simulated data.
    """

    # helper function for file name, to avoid repeating code
    def get_path_name(
        confidence_interval, highlight_roc_area, n_bootstraps, figsize, split_plots
    ):
        if split_plots is False:
            fig_path = (
                TEST_RESULTS_PATH
                / f"roc_curves_conf_{confidence_interval}_highlight_{highlight_roc_area}_nboot_{n_bootstraps}_figsize_{figsize}.png"
            )
        else:
            fig_path_1 = (
                TEST_RESULTS_PATH
                / f"roc_curves_split_conf_{confidence_interval}_highlight_{highlight_roc_area}_nboot_{n_bootstraps}_figsize_{figsize}.png"
            )
            fig_path_2 = (
                TEST_RESULTS_PATH
                / f"auroc_comparison_conf_{confidence_interval}_highlight_{highlight_roc_area}_nboot_{n_bootstraps}_figsize_{figsize}.png"
            )
            fig_path = [fig_path_1, fig_path_2]
        return fig_path

    y_true, y_prob = random_data_multiclass_classifier(num_classes=3)

    confidence_intervals = [None, 0.99]
    highlight_roc_area = [True, False]
    n_bootstraps = [1, 1000]
    figsizes = [None]
    split_plots = [True, False]

    # From the previous lists I want all possible combinations
    combinations = list(
        product(
            confidence_intervals,
            highlight_roc_area,
            n_bootstraps,
            figsizes,
            split_plots,
        )
    )

    for (
        confidence_interval,
        highlight_roc_area,
        n_bootstraps,
        figsize,
        split_plots,
    ) in combinations:
        # check if one or two figures should be saved (splot_plots=True or False)
        # if split_plots is False:
        #     fig_path = TEST_RESULTS_PATH / f"roc_curves_conf_{confidence_interval}_highlight_{highlight_roc_area}_nboot_{n_bootstraps}_figsize_{figsize}.png"
        # else:
        #     fig_path_1 = TEST_RESULTS_PATH / f"roc_curves_conf_{confidence_interval}_highlight_{highlight_roc_area}_nboot_{n_bootstraps}_figsize_{figsize}.png"
        #     fig_path_2 = TEST_RESULTS_PATH / f"auroc_comparison_conf_{confidence_interval}_highlight_{highlight_roc_area}_nboot_{n_bootstraps}_figsize_{figsize}.png"
        #     fig_path = [fig_path_1, fig_path_2]

        fig_path = get_path_name(
            confidence_interval, highlight_roc_area, n_bootstraps, figsize, split_plots
        )

        # It should raise an error when confidence_interval is None but highlight_roc_area is True
        if confidence_interval is None and highlight_roc_area is True:
            with pytest.raises(ValueError):
                multiclass.plot_roc_curve(
                    y_true=y_true,
                    y_score=y_prob,
                    confidence_interval=confidence_interval,
                    highlight_roc_area=highlight_roc_area,
                    n_bootstraps=n_bootstraps,
                    figsize=figsize,
                    split_plots=split_plots,
                    save_fig_path=fig_path,
                )
        # Otherwise no error
        else:
            multiclass.plot_roc_curve(
                y_true=y_true,
                y_score=y_prob,
                confidence_interval=confidence_interval,
                highlight_roc_area=highlight_roc_area,
                n_bootstraps=n_bootstraps,
                figsize=figsize,
                split_plots=split_plots,
                save_fig_path=fig_path,
            )

    # check for SMALL figure size
    confidence_interval, highlight_roc_area, n_bootstraps, figsize, split_plots = (
        0.95,
        True,
        100,
        (3, 3),
        False,
    )
    fig_path = get_path_name(
        confidence_interval, highlight_roc_area, n_bootstraps, figsize, split_plots
    )
    multiclass.plot_roc_curve(
        y_true=y_true,
        y_score=y_prob,
        confidence_interval=confidence_interval,
        highlight_roc_area=highlight_roc_area,
        n_bootstraps=n_bootstraps,
        figsize=figsize,
        split_plots=split_plots,
        save_fig_path=fig_path,
    )

    # check for BIG figure size
    confidence_interval, highlight_roc_area, n_bootstraps, figsize, split_plots = (
        0.95,
        True,
        100,
        (15, 15),
        False,
    )
    fig_path = get_path_name(
        confidence_interval, highlight_roc_area, n_bootstraps, figsize, split_plots
    )
    multiclass.plot_roc_curve(
        y_true=y_true,
        y_score=y_prob,
        confidence_interval=confidence_interval,
        highlight_roc_area=highlight_roc_area,
        n_bootstraps=n_bootstraps,
        figsize=figsize,
        split_plots=split_plots,
        save_fig_path=fig_path,
    )
