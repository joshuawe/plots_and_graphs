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


def plot_accuracy(y_true, y_pred, name="", save_fig_path=None) -> Figure:
    """Really ugly plot, I am not sure if the scalar value for accuracy should receive an entire plot."""
    accuracy = accuracy_score(y_true, y_pred)

    # accuracy = 0
    # for t in range(max_seq_len):
    #     accuracy += accuracy_score( y[:,t,0].round()  , y_pred[:,t] )
    # accuracy = accuracy / max_seq_len
    fig = plt.figure(figsize=(4, 5))
    plt.bar(np.array([0]), np.array([accuracy]))
    # axs[0].set_xticks(ticks=range(2))
    # axs[0].set_xticklabels(["train", "test"])
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    # axs[0].set_xlabel('Features')
    title = "Predictor model: {}".format(name)
    plt.title(title)
    plt.tight_layout()

    if save_fig_path is not None:
        path = Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches="tight")
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_fig_path=None) -> Figure:
    import matplotlib.colors as colors

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred.round())
    # normalize the confusion matrix
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create the ConfusionMatrixDisplay instance and plot it
    cmd = ConfusionMatrixDisplay(cm, display_labels=["class 0\nnegative", "class 1\npositive"])
    fig, ax = plt.subplots(figsize=(4, 4))
    cmd.plot(
        cmap="YlOrRd",
        values_format="",
        colorbar=False,
        ax=ax,
        # text_kw={"visible": False},
    )
    cmd.texts_ = []
    cmd.text_ = []

    text_labels = ["TN", "FP", "FN", "TP"]
    cmap_min, cmap_max = cmd.im_.cmap(0), cmd.im_.cmap(1.0)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{text_labels[i * 2 + j]}\n{cmd.im_.get_array()[i, j]:.2%}",
                ha="center",
                va="center",
                color=cmap_min if cmd.im_.get_array()[i, j] > 0.5 else cmap_max,
            )

    ax.vlines([0.5], *ax.get_ylim(), color="white", linewidth=1)
    ax.hlines([0.49], *ax.get_xlim(), color="white", linewidth=1)
    ax.spines[:].set_visible(False)

    bounds = np.linspace(0, 1, 11)
    cmap = plt.cm.get_cmap("YlOrRd", len(bounds) + 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    cbar = ax.figure.colorbar(
        cmd.im_,
        ax=ax,
        cmap=cmap,
        norm=norm,
        boundaries=bounds,
        ticks=bounds[::2],
        location="right",
        shrink=0.8,
    )
    # cbar.set_ticks(np.arange(0,1.1,0.1))
    cbar.ax.yaxis.set_ticks_position("both")
    cbar.outline.set_visible(False)
    plt.tight_layout()

    if save_fig_path is not None:
        path = Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches="tight")

    return fig


def plot_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title="Classification Report",
    figsize=(8, 4),
    save_fig_path=None,
    **kwargs,
):
    """
    TODO: save all these plots
    Plot the classification report of sklearn

    Parameters
    ----------
    y_test : pandas.Series of shape (n_samples,)
        Targets.
    y_pred : pandas.Series of shape (n_samples,)
        Predictions.
    title : str, default = 'Classification Report'
        Plot title.
    fig_size : tuple, default = (8, 6)
        Size (inches) of the plot.
    dpi : int, default = 70
        Image DPI.
    save_fig_path : str, defaut=None
        Full path where to save the plot. Will generate the folders if they don't exist already.
    **kwargs : attributes of classification_report class of sklearn

    Returns
    -------
        fig : Matplotlib.pyplot.Figure
            Figure from matplotlib
        ax : Matplotlib.pyplot.Axe
            Axe object from matplotlib
    """
    import matplotlib as mpl
    import matplotlib.colors as colors
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)

    cmap = "YlOrRd"

    clf_report = classification_report(y_true, y_pred, output_dict=True, **kwargs)
    keys_to_plot = [key for key in clf_report.keys() if key not in ("accuracy", "macro avg", "weighted avg")]
    df = pd.DataFrame(clf_report, columns=keys_to_plot).T
    # the following line ensures that dataframe are sorted from the majority classes to the minority classes
    df.sort_values(by=["support"], inplace=True)

    # first, let's plot the heatmap by masking the 'support' column
    rows, cols = df.shape
    mask = np.zeros(df.shape)
    mask[:, cols - 1] = True

    bounds = np.linspace(0, 1, 11)
    cmap = plt.cm.get_cmap("YlOrRd", len(bounds) + 1)  # type: ignore[assignment]
    norm = colors.BoundaryNorm(bounds, cmap.N)  # type: ignore[attr-defined]

    ax = sns.heatmap(
        df,
        mask=mask,
        annot=False,
        cmap=cmap,
        fmt=".3g",
        cbar_kws={"ticks": bounds[::2], "norm": norm, "boundaries": bounds},
        vmin=0.0,
        vmax=1.0,
        linewidths=2,
        linecolor="white",
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_ticks_position("both")

    cmap_min, cmap_max = cbar.cmap(0), cbar.cmap(1.0)

    # add text annotation to heatmap
    dx, dy = 0.5, 0.5
    for i in range(rows):
        for j in range(cols - 1):
            text = f"{df.iloc[i, j]:.2%}"  # if (j<cols) else f"{df.iloc[i, j]:.0f}"
            ax.text(
                j + dx,
                i + dy,
                text,
                # ha="center", va="center", color='black')
                ha="center",
                va="center",
                color=cmap_min if df.iloc[i, j] > 0.5 else cmap_max,
            )

    # then, let's add the support column by normalizing the colors in this column
    mask = np.zeros(df.shape)
    mask[:, : cols - 1] = True

    ax = sns.heatmap(
        df,
        mask=mask,
        annot=False,
        cmap=cmap,
        cbar=False,
        linewidths=2,
        linecolor="white",
        fmt=".0f",
        vmin=df["support"].min(),
        vmax=df["support"].sum(),
        norm=mpl.colors.Normalize(vmin=df["support"].min(), vmax=df["support"].sum()),
    )

    cmap_min, cmap_max = cbar.cmap(0), cbar.cmap(1.0)
    for i in range(rows):
        j = cols - 1
        text = f"{df.iloc[i, j]:.0f}"  # if (j<cols) else f"{df.iloc[i, j]:.0f}"
        color = (df.iloc[i, j]) / (df["support"].sum())
        ax.text(
            j + dx,
            i + dy,
            text,
            # ha="center", va="center", color='black')
            ha="center",
            va="center",
            color=cmap_min if color > 0.5 else cmap_max,
        )

    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=360)
    plt.tight_layout()

    if save_fig_path is not None:
        path = Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches="tight")

    return fig, ax


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    figsize=(5, 5),
    save_fig_path=None,
    confidence_interval: float = 0.95,
    highlight_roc_area=True,
    n_bootstraps=None,
) -> Figure:
    """
    Creates a ROC curve for a binary classifier. Includes the option for bootstrapping.

    Parameters
    ----------
    y_true : np.ndarray
        The actual labels of the data. Either 0 or 1.
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
    fig : matplotlib.pyplot figure
        The figure of the calibration plot
    """

    # create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if n_bootstraps is None:
        base_fpr, mean_tprs, thresholds = roc_curve(y_true, y_score)
        mean_auc = auc(base_fpr, mean_tprs)
        if highlight_roc_area is True:
            plt.fill_between(base_fpr, 0, mean_tprs, alpha=0.2, zorder=2)
        if confidence_interval is not None:
            print(
                "Warning: confidence_intervals is not None, but n_bootstraps is None. Confidence intervals will not be plotted."
            )
    else:
        # Bootstrapping for AUROC
        bootstrap_aucs, bootstrap_tprs = [], []
        base_fpr = np.linspace(0, 1, 101)
        for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"):
            indices = resample(np.arange(len(y_true)), replace=True)
            fpr_i, tpr_i, _ = roc_curve(y_true[indices], y_score[indices])
            roc_auc_i = auc(fpr_i, tpr_i)
            bootstrap_aucs.append(roc_auc_i)

            # Interpolate tpr_i to base_fpr, so we have the tpr for the same fpr values for each bootstrap iteration
            tpr_i_interp = np.interp(base_fpr, fpr_i, tpr_i)
            tpr_i_interp[0] = 0.0
            bootstrap_tprs.append(tpr_i_interp)

        mean_auc = np.mean(bootstrap_aucs)
        tprs = np.array(bootstrap_tprs)
        mean_tprs = tprs.mean(axis=0)

        # visualize confidence intervals
        if confidence_interval is not None:
            CI_upper = confidence_interval + (1 - confidence_interval) / 2
            CI_lower = (1 - confidence_interval) / 2
            tprs_upper = np.quantile(tprs, CI_upper, axis=0)
            tprs_lower = np.quantile(tprs, CI_lower, axis=0)
            auc_upper = np.quantile(bootstrap_aucs, CI_upper)
            auc_lower = np.quantile(bootstrap_aucs, CI_lower)
            label = f"{confidence_interval:.0%} CI: [{auc_lower:.2f}, {auc_upper:.2f}]"
            plt.fill_between(base_fpr, tprs_lower, tprs_upper, alpha=0.3, label=label, zorder=2)

        if highlight_roc_area is True:
            print(
                "Warning: highlight_roc_area is True, but n_bootstraps is not None. The area under the ROC curve will not be highlighted."
            )

    plt.plot(base_fpr, mean_tprs, label=f"ROC curve (AUROC = {mean_auc:.2f})", zorder=3)
    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
    plt.xlim([0.0, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    # reverse legend entry order
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    plt.legend(handles, labels, loc="lower right", frameon=False)
    ax.spines[:].set_visible(False)
    ax.grid(True, linestyle="-", linewidth=0.5, color="grey", alpha=0.5)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    plt.tight_layout()

    if save_fig_path:
        path = Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches="tight")

    return fig


def plot_calibration_curve(y_prob: np.ndarray, y_true: np.ndarray, n_bins=10, save_fig_path=None) -> Figure:
    """
    Creates calibration plot for a binary classifier and calculates the ECE.

    Parameters
    ----------
    y_prob : np.ndarray
        The output probabilities of the classifier. Between 0 and 1.
    y_true : np.ndarray
        The actual labels of the data. Either 0 or 1.
    n_bins : int
        The number of bins to use for the calibration curve.
    save_fig_path : str, optional
        Path to folder where the figure should be saved. If None then plot is not saved, by default None

    Returns
    -------
    fig : matplotlib.pyplot figure
        The figure of the calibration plot
    ece : float
        The expected calibration error.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    # Find the number of samples in each bin
    bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]

    # Calculate the weighted absolute difference (ECE)
    ece = np.abs(prob_pred - prob_true) * (bin_counts / len(y_prob))
    ece = ece.sum().round(2)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    # Evenly spaced bar locations on the x-axis and reduced bar width for spacing
    bar_centers = np.linspace(0, 1, n_bins, endpoint=False) + 0.5 / n_bins
    bar_width = 1.0 / n_bins  # * 0.9  # 90% of the bin width to create gaps

    # Plotting
    ax.bar(
        bar_centers,
        prob_true,
        width=bar_width,
        align="center",
        zorder=3,
        facecolor=to_rgba("C0", 0.75),
        edgecolor="midnightblue",
        linewidth=2,
        label=f"True Calibration",
    )
    ax.bar(
        bar_centers,
        prob_pred - prob_true,
        bottom=prob_true,
        width=bar_width,
        align="center",
        zorder=3,
        alpha=0.5,
        edgecolor="red",
        fill=False,
        linewidth=2,
        label=f"Mean ECE = {ece}",
        hatch="//",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="grey",
        zorder=3,
        label="Perfect Calibration",
    )

    # Labels and titles
    ax.set(xlabel="Predicted probability", ylabel="True probability")
    plt.xlim([0.0, 1.005])
    plt.ylim([-0.01, 1.0])
    ax.legend(loc="upper left", frameon=False)

    # show y-grid
    ax.spines[:].set_visible(False)
    ax.grid(True, linestyle="-", linewidth=0.5, color="grey", alpha=0.5)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    plt.tight_layout()

    # save plot
    if save_fig_path is not None:
        path = Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches="tight")

    return fig


def plot_y_prob_histogram(y_prob: np.ndarray, y_true: Optional[np.ndarray] = None, save_fig_path=None) -> Figure:
    """
    Provides a histogram for the predicted probabilities of a binary classifier. If ```y_true``` is provided, it divides the ```y_prob``` values into the two classes and plots them jointly into the same plot with different colors.

    Parameters
    ----------
    y_prob : np.ndarray
        The output probabilities of the classifier. Between 0 and 1.
    y_true : Optional[np.ndarray], optional
        The true class labels, by default None
    save_fig_path : _type_, optional
        Path where to save figure, by default None

    Returns
    -------
    Figure
        The histrogram as a matplotlib figure
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    if y_true is None:
        ax.hist(y_prob, bins=10, alpha=0.9, edgecolor="midnightblue", linewidth=2, rwidth=1)
        # same histogram as above, but with border lines
        # ax.hist(y_prob, bins=10, alpha=0.5, edgecolor='black', linewidth=1.2)
    else:
        alpha = 0.6
        ax.hist(
            y_prob[y_true == 0],
            bins=10,
            alpha=alpha,
            edgecolor="midnightblue",
            linewidth=2,
            rwidth=1,
            label="$\\hat{y} = 0$",
        )
        ax.hist(
            y_prob[y_true == 1],
            bins=10,
            alpha=alpha,
            edgecolor="darkred",
            linewidth=2,
            rwidth=1,
            label="$\\hat{y} = 1$",
        )

        plt.legend()
    ax.set(xlabel="Predicted probability [-]", ylabel="Count [-]", xlim=(-0.01, 1.0))
    ax.set_title("Histogram of predicted probabilities")

    ax.spines[:].set_visible(False)
    ax.grid(True, linestyle="-", linewidth=0.5, color="grey", alpha=0.5)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    plt.tight_layout()

    # save plot
    if save_fig_path is not None:
        path = Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches="tight")

    return fig


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    figsize=(5, 5),
    save_fig_path: Optional[str] = None,
    color: Optional[str] = None,
    label: Optional[str] = None,
    title: Optional[str] = None,
) -> Figure:
    """
    Visualize the Precision-Recall curve for a binary classifier.

    Parameters
    ----------
    y_true : np.ndarray
        The actual labels of the data. Either 0 or 1.
    y_score : np.ndarray
        The output scores of the classifier. Between 0 and 1.
    figsize : tuple, optional
        The size of the figure. By default (5,5).
    save_fig_path : str, optional
        Path to folder where the figure should be saved. If None then plot is not saved, by default None. E.g. 'figures/pr_curve.png'.
    color : str, optional
        Color of the PR curve, by default None.
    label : str, optional
        Custom label for the plot. If None, a default label is used. By default None.

    Returns
    -------
    fig : matplotlib.pyplot figure
        The figure of the PR curve
    """

    # Create a new figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Compute Precision-Recall curve and area for each class
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    pr_auc = auc(recall, precision)

    if label is None:
        # Use a default label if none is provided
        label = "PR curve"

    label += f" (area = {pr_auc:.3f})"

    # Plot Precision-Recall curve
    ax.plot(recall, precision, label=label, color=color)
    ax.set_xlim((0.0, 1.01))
    ax.set_ylim((-0.01, 1.01))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="lower right")
    ax.spines[:].set_visible(False)
    ax.grid(True, linestyle="-", linewidth=0.5, color="grey", alpha=0.5)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    plt.tight_layout()

    # Save the figure if save_fig_path is specified
    if save_fig_path:
        plt.savefig(save_fig_path, bbox_inches="tight")

    return fig
