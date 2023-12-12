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

from plotsandgraphs.utils import bootstrap, set_black_title_box, scale_ax_bbox, get_cmap


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    confidence_interval: float = 0.95,
    highlight_roc_area: bool=True,
    n_bootstraps: int=1,
    figsize: Optional[Tuple[float, float]]=None,
    class_labels: Optional[List[str]]=None,
    split_plots: bool=True,
    save_fig_path=None,
) -> Tuple[Figure, Figure]:
    """
    Creates two plots. 
    1) ROC curves for a multiclass classifier. Includes the option for bootstrapping.
    2) An overview of the AUROC for each class and the macro average AUROC for one vs rest.
    Note: That the AUROC overview plot can be included with the ROC curves by setting split_plots=False.
    

    Parameters
    ----------
    y_true : np.ndarray
        The actual labels of the data. Either 0 or 1. One hot encoded.
    y_score : np.ndarray
        The output scores of the classifier. Between 0 and 1.
    figsize : tuple, optional
        The size of the figure. By default (5,5).
    confidence_interval : float, optional
        The confidence interval to use for the calibration plot. By default 0.95. Between 0 and 1. Has no effect when not using n_bootstraps.
    highlight_roc_area : bool, optional
        Whether to highlight the area under the ROC curve. By default True. Has no effect when using n_bootstraps.
    n_bootstraps : int, optional
        Number of bootstrap samples to use for the calibration plot. Recommended minimum: 1000, moderate: 5000-10000, high: 50000-100000.
        If None, then no bootstrapping is done. By default None.
    class_labels : List[str], optional
        The labels of the classes. By default None.
    split_plots : bool, optional
        Whether to split the plots into two separate figures. By default True.
    save_fig_path : str, optional
        Path to folder where the figure should be saved. If None then plot is not saved, by default None. E.g. 'figures/'.

    Returns
    -------
    figures : Tuple[Figure, Figure] 
        The figures of the calibration plot. First the roc curves, then the AUROC overview.
    """
    
    num_classes = y_true.shape[-1]
    class_labels = [f"Class {i}" for i in range(num_classes)] if class_labels is None else class_labels
    cmap, colors = get_cmap('roma', n_colors=num_classes+1)    
    
    # ------ ROC curves ------
    n_subplots = num_classes + (split_plots is False)
    # Aiming for a square plot
    plot_cols = np.ceil(np.sqrt(n_subplots)).astype(int) # Number of plots in a row
    plot_rows = np.ceil(n_subplots / plot_cols).astype(int) # Number of plots in a column
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
    for i in tqdm(range(num_classes), desc='ROC for Class'):
        # only plot axis that should be printed
        if i >= num_classes:
            continue
        
        # --- BOOTSTRAPPING / CALCULATIONS ---
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
        
        # --- PLOTTING ---
        ax = axes.flat[i]
        ax.plot(common_fpr, tpr_median, label='Median ROC', c=colors[i])
        if highlight_roc_area:
            ax.fill_between(common_fpr, tpr_lower, tpr_upper, alpha=0.2, label=f'{confidence_interval:.1%} CI', fc=colors[i])
        ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        # if subplot in first column
        if (i % plot_cols) == 0:
            ax.set_ylabel("True Positive Rate")
        # if subplot in last row
        if (i // plot_cols) + 1 == plot_rows:
            ax.set_xlabel("False Positive Rate")
        
        # plot AUROC at the bottom right of each plot
        auroc_text = f"AUROC: {auc_median:.3f} {confidence_interval:.0%} CI: [{auc_lower:.3f},{auc_upper:.3f}]"
        ax.text(0.99, 0.02, auroc_text, ha="right", va="bottom", transform=ax.transAxes)
        
        # Legend only in first subplot
        if i == 0:
            # plot legend above previous text
            ax.legend(loc="center right", frameon=True, bbox_to_anchor=(0.5, 0., 0.5, 0.5))
        ax.spines[:].set_color("grey")
        # ax.grid(True, linestyle="-", linewidth=0.5, color="grey", alpha=0.5)
        ax.set_yticks(np.arange(0, 1.1, 0.2))

    # disable axis for subplots in last row that are not required
    for i in range(num_classes, len(axes.flat)):
        axes.flat[i].axis("off")
    
    # make the subplot tiles (and black boxes) 
    for i in range(num_classes): set_black_title_box(axes.flat[i], f"Class {i}")
    plt.tight_layout(h_pad=1.5)
    # make the subplot tiles (and black boxes)
    #  First time to get the approx. correct spacing with plt.tight_layout()
    #  Second time to get the correct width of the black box
    #  Thank you matplotlib ...
    for i in range(num_classes):
        set_black_title_box(axes.flat[i], f"Class {i}", set_title_kwargs={'fontdict': {'fontname': 'Arial Black', 'fontweight': 'bold'}})

        
        
    # ---------- AUROC overview plot comparing classes ----------
    # Make an AUROC overview plot comparing the aurocs per class and combined
    
    # Define metric funtion to calculate one vs rest AUROC
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
    if split_plots:
        fig_aurocs = plt.figure(figsize=(5,5))
        ax = fig_aurocs.add_subplot(111)
    else:
        fig_aurocs = None 
        ax = axes.flat[num_classes+1-1] # +1 cause we plot after class roc curve, -1 cause indexing begins at 0
        scale_ax_bbox(ax, 0.9) # needed to avoid plot overlap and calling plt.tight_layout() again
        ax.axis("on")
        # y ticks on
        ax.tick_params(axis='y', which='both', left=True, labelleft=True)
    ax.set_title(f"AUROC (One vs Rest, CI={confidence_interval:.0%})")
    labels = ['Macro\nAverage'] + class_labels
    
    for i, (lower, median, upper) in enumerate(zip(aucs_lower, aucs_median, aucs_upper)):
        lower = median - lower
        upper = upper - median
        ax.errorbar(i, median, yerr=[[lower], [upper]], ecolor='grey', capsize=5, capthick=2)
        ax.scatter(i, median, s=50, color=colors[i-1], zorder=10, alpha=0.8)

    ax.set(xticks=range(len(labels)),xticklabels=labels)
    ax.tick_params(axis='x', rotation=0)
    ax.set_yticks(np.arange(0, 1.1, 0.1), minor=True)
    ax.grid(True, linestyle=":", axis='both')
    
    # save auroc comparison plot
    if save_fig_path and split_plots is True:
        path = Path(save_fig_path) / "aurocs_comparison.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig_aurocs.savefig(path, bbox_inches="tight")
    # save roc curves plot
    if save_fig_path:
        path = Path(save_fig_path) / "roc_curves.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
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
