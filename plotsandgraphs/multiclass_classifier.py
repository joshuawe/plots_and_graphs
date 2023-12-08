from pathlib import Path
from typing import Optional, List, Callable, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import BoxStyle
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    accuracy_score,
    precision_recall_curve,
    roc_auc_score
)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from tqdm import tqdm

from plotsandgraphs.utils import bootstrap, set_black_title_box


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    figsize: Optional[Tuple[float, float]]=None,
    save_fig_path=None,
    confidence_interval: float = 0.95,
    highlight_roc_area: bool=True,
    n_bootstraps: int=1,
) -> Tuple[Figure, Figure]:
    """
    Creates a ROC curve for a multiclass classifier. Includes the option for bootstrapping.

    Parameters
    ----------
    y_true : np.ndarray
        The actual labels of the data. Either 0 or 1. One hot encoded.
    y_score : np.ndarray
        The output scores of the classifier. Between 0 and 1.
    figsize : tuple, optional
        The size of the figure. By default (5,5).
    save_fig_path : str, optional
        Path to folder where the figure should be saved. If None then plot is not saved, by default None. E.g. 'figures/roc_curve.png'.
    confidence_interval : float, optional
        The confidence interval to use for the calibration plot. By default 0.95. Between 0 and 1. Has no effect when not using n_bootstraps.
    highlight_roc_area : bool, optional
        Whether to highlight the area under the ROC curve. By default True. Has no effect when using n_bootstraps.
    n_bootstraps : int, optional
        Number of bootstrap samples to use for the calibration plot. Recommended minimum: 1000, moderate: 5000-10000, high: 50000-100000.
        If None, then no bootstrapping is done. By default None.

    Returns
    -------
    figures : Tuple[Figure, Figure] 
        The figures of the calibration plot. First the roc curves, then the AUROC overview.
    """
    
    # Aiming for a square plot
    plot_cols = np.ceil(np.sqrt(y_true.shape[-1])).astype(int) # Number of plots in a row
    plot_rows = np.ceil(y_true.shape[-1] / plot_cols).astype(int) # Number of plots in a column
    figsize = (plot_cols*4+1, plot_rows*4) if figsize is None else figsize
    fig, axes = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=figsize, sharey=True)
    plt.suptitle("Receiver Operating Characteristic (ROC), One vs Rest")
    
    # the roc metric function to pass for bootstrapping
    def roc_metric_function(y_true, y_score):
        fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=False)
        auc_score = auc(fpr, tpr)
        # print lengths
        # print(fpr.shape, tpr.shape, auc_score)
        return fpr, tpr, auc_score
    
    aucs_lower, aucs_median, aucs_upper = [], [], []

    # for each class calculate the ROC
    for i in tqdm(range(y_true.shape[-1]), desc='ROC for Class'):
        # only plot axis that should be printed
        if i >= y_true.shape[-1]:
            # axes.flat[i].axis("off")
            continue
        # bootstrap the ROC curve for class i
        roc_result = bootstrap(
                        metric_function=roc_metric_function,
                        input_resample=[y_true[:, i], y_score[:, i]],
                        n_bootstraps=n_bootstraps, 
                        metric_kwargs={})
        # unpack the results
        fprs, tprs, aucs = zip(*roc_result)
        fprs, tprs, aucs = [x for x in [fprs, tprs, aucs]]
        
        # Determine min and max FPR values across all bootstrapped samples
        min_fpr = min(min(fpr) for fpr in fprs)
        max_fpr = max(max(fpr) for fpr in fprs)
        
        # Define common FPR values for interpolation within the min and max range
        common_fpr = np.linspace(min_fpr, max_fpr, 200)
        
        # Interpolate TPRs for each bootstrap sample
        interp_tprs = [np.interp(common_fpr, np.sort(fpr), tpr[np.argsort(fpr)]) for fpr, tpr in zip(fprs, tprs)]
        
        # calculate median and quantiles
        quantiles = [0.5-confidence_interval/2, 0.5, 0.5+confidence_interval/2]
        tpr_lower, tpr_median, tpr_upper = np.quantile(interp_tprs, q=quantiles, axis=0)
        auc_lower, auc_median, auc_upper = np.quantile(aucs, q=quantiles, axis=0)
        aucs_lower.append(auc_lower)
        aucs_median.append(auc_median)
        aucs_upper.append(auc_upper)
        
        ax = axes.flat[i]
        ax.plot(common_fpr, tpr_median, label='Median ROC')
        ax.fill_between(common_fpr, tpr_lower, tpr_upper, alpha=0.2, label=f'{confidence_interval:.1%} CI')
        ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        # if subplot in first column
        if (i % plot_cols) == 0:
            ax.set_ylabel("True Positive Rate")
        # if subplot in last row
        if (i // plot_rows) + 1 == plot_rows:
            ax.set_xlabel("False Positive Rate")
        
        # plot AUROC at the bottom right of the plot
        auroc_text = f"AUROC: {auc_median:.3f} {confidence_interval:.0%} CI: [{auc_lower:.3f},{auc_upper:.3f}]"
        ax.text(0.99, 0.02, auroc_text, ha="right", va="bottom", transform=ax.transAxes)
        
        
        if i == 0:
            # plot legend above previous text
            ax.legend(loc="center right", frameon=True, bbox_to_anchor=(0.5, 0., 0.5, 0.5))
        ax.spines[:].set_color("grey")
        # ax.grid(True, linestyle="-", linewidth=0.5, color="grey", alpha=0.5)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
    plt.tight_layout()
    
    # make the subplot tiles (and black boxes)
    for i in range(y_true.shape[-1]):
        ax = axes.flat[i]
        set_black_title_box(ax, f"Class {i}")
    plt.tight_layout(h_pad=1.5)

    if save_fig_path:
        path = Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches="tight")
        
    # ---------- AROC overview plot ----------
    # Make an AUROC overview plot comparing the aurocs per class and combined
    
    def auroc_metric_function(y_true, y_score, average, multi_class):
        auc = roc_auc_score(y_true, y_score, average=average, multi_class=multi_class)
        return auc
    
    # get the combined auroc bootstrap results for one vs rest
    roc_result = bootstrap(
                    metric_function=auroc_metric_function,
                    input_resample=[y_true, y_score],
                    n_bootstraps=n_bootstraps, 
                    metric_kwargs={'average': 'macro', 'multi_class': 'ovr'})
    
    # get the lower, median and upper quantiles
    auc_comb_lower, auc_comb_median, auc_comb_upper = np.quantile(roc_result, q=quantiles, axis=0)
    # add the result to the beginning of the numpy array
    aucs_lower = np.insert(aucs_lower, 0, auc_comb_lower)
    aucs_median = np.insert(aucs_median, 0, auc_comb_median)
    aucs_upper = np.insert(aucs_upper, 0, auc_comb_upper)

    
    
    # create the plot
    fig_aurocs = plt.figure(figsize=(5,5))
    plt.title(f"AUROC (One vs Rest, CI={confidence_interval:.0%})")
    labels = ['One vs Rest'] + [f'Class {i}' for i in range(y_true.shape[-1])]
    
    for i, (lower, median, upper) in enumerate(zip(aucs_lower, aucs_median, aucs_upper)):
        lower = median - lower
        upper = upper - median
        ax = plt.errorbar(i, median, yerr=[[lower], [upper]], fmt='o', color='black', ecolor='skyblue', capsize=5, capthick=2)
    
    # get axis
    ax = plt.gca()
    ax.spines[:].set_color("grey")
    plt.xticks(range(len(labels)), labels, rotation=0)
    plt.grid(True, linestyle=":", axis='both')

    return fig, fig_aurocs


def plot_y_prob_histogram(y_true: np.ndarray, y_prob: Optional[np.ndarray] = None, save_fig_path=None) -> Figure:
    
    # Aiming for a square plot
    plot_cols = np.ceil(np.sqrt(y_true.shape[-1])).astype(int) # Number of plots in a row
    plot_rows = np.ceil(y_true.shape[-1] / plot_cols).astype(int) # Number of plots in a column
    fig, axes = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(plot_cols*4+1, plot_rows*4), sharey=True)
    alpha = 0.6
    plt.suptitle("Predicted probability histogram")
    
    # Flatten axes if there is only one class, even though this function is designed for multiclasses
    if y_true.shape[-1] == 1:
        axes = np.array([axes])
    
    for i, ax in enumerate(axes.flat):
        if i >= y_true.shape[-1]:
            ax.axis("off")
            continue
        
        if y_prob is not None:
            y_true_i = y_true[:, i]
            y_prob_i = y_prob[:, i]
            ax.hist(y_prob_i[y_true_i==0], 
                    bins=10, 
                    label="$\\hat{y} = 0$",
                    alpha=alpha,
                    edgecolor="midnightblue",
                    linewidth=2,
                    rwidth=1,)
            ax.hist(y_prob_i[y_true_i==1], 
                    bins=10, 
                    label="$\\hat{y} = 1$",
                    alpha=alpha,
                    edgecolor="midnightblue",
                    linewidth=2,
                    rwidth=1,)
            ax.set_title(f"Class {i}")
            ax.set_xlim((-0.005, 1.0))
            # if subplot in first column
            if (i % plot_cols) == 0:
                ax.set_ylabel("Count [-]")
            # if subplot in last row
            if (i // plot_cols) + 1 == plot_rows:
                ax.set_xlabel("$P\\,(y=1)$")
            # ax.spines[:].set_visible(False)
            ax.grid(True, linestyle="-", linewidth=0.5, color="grey", alpha=0.5)
            ax.set_xticks(np.arange(0, 1.1, 0.2))
            # only first subplot should have legends
            if i == 0:
                ax.legend()
    
    plt.tight_layout()
    
    # save plot
    if save_fig_path is not None:
        path = Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches="tight")
    return fig
