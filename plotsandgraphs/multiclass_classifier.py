from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
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
)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from tqdm import tqdm


def plot_y_prob_histogram(y_true: np.ndarray, y_prob: Optional[np.ndarray] = None, save_fig_path=None) -> Figure:
    
    plot_len = np.ceil(np.sqrt(y_true.shape[-1])).astype(int) # Number of plots in a row/column
    fig, axes = plt.subplots(nrows=plot_len, ncols=plot_len, figsize=(plot_len*4+1, plot_len*4), sharey=True)
    alpha = 0.6
    plt.suptitle("Predicted probability histogram")
    
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
            if i % plot_len == 0:
                ax.set_ylabel("Count [-]")
            # if subplot in last row
            if i >= plot_len*(plot_len-1):
                ax.set_xlabel("Predicted probability [-]")
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
