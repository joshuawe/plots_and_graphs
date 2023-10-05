import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score
from sklearn.calibration import calibration_curve
from pathlib import Path



def plot_accuracy(y_true, y_pred, name='', save_fig_path=None):
    """ Really ugly plot, I am not sure if the scalar value for accuracy should receive an entire plot."""
    accuracy = accuracy_score(y_true, y_pred)
        
    # accuracy = 0
    # for t in range(max_seq_len): 
    #     accuracy += accuracy_score( y[:,t,0].round()  , y_pred[:,t] )
    # accuracy = accuracy / max_seq_len
    fig= plt.figure( figsize=(4,5))
    plt.bar( np.array([0]), np.array([  accuracy  ]))
    # axs[0].set_xticks(ticks=range(2))
    # axs[0].set_xticklabels(["train", "test"])
    plt.ylabel('Accuracy')
    plt.ylim([0,1])
    # axs[0].set_xlabel('Features')
    title = "Predictor model: {}".format(name )
    plt.title(title)
    plt.tight_layout()
    
    if (save_fig_path != None):
        path = Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches='tight')
    return fig, accuracy

def plot_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, save_fig_path=None) -> "matplotlib.figure.Figure":
    import matplotlib.colors as colors
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred.round())
    # normalize the confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create the ConfusionMatrixDisplay instance and plot it
    cmd = ConfusionMatrixDisplay(cm, display_labels=['class 0\nnegative', 'class 1\npositive'])
    fig, ax = plt.subplots(figsize=(4,4))
    cmd.plot(cmap='YlOrRd', values_format='', colorbar=False, ax=ax, text_kw={'visible':False})
    cmd.texts_ = []
    cmd.text_ = []

    text_labels = ['TN', 'FP', 'FN', 'TP']
    cmap_min, cmap_max = cmd.im_.cmap(0), cmd.im_.cmap(1.0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{text_labels[i * 2 + j]}\n{cmd.im_.get_array()[i, j]:.2%}",
                    ha="center", va="center", color=cmap_min if cmd.im_.get_array()[i, j] > 0.5 else cmap_max)
            
    ax.vlines([0.5], *ax.get_ylim(), color='white', linewidth=1)
    ax.hlines([0.49], *ax.get_xlim(), color='white', linewidth=1)
    ax.spines[:].set_visible(False)
    
    
    bounds = np.linspace(0, 1, 11)
    cmap = plt.cm.get_cmap('YlOrRd', len(bounds)+1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    cbar = ax.figure.colorbar(cmd.im_, ax=ax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds[::2], location="right", shrink=0.8)
    # cbar.set_ticks(np.arange(0,1.1,0.1))
    cbar.ax.yaxis.set_ticks_position('both')
    cbar.outline.set_visible(False)
    plt.tight_layout()
    
    if (save_fig_path != None):
        path = Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches='tight')
    
    return fig




def plot_classification_report(y_test: torch.Tensor, 
                               y_pred: torch.Tensor, 
                               title='Classification Report', 
                               figsize=(8, 4), 
                               save_fig_path=None, **kwargs):
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
    import pathlib
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cmap = 'YlOrRd'
        
    clf_report = classification_report(y_test, y_pred, output_dict=True, **kwargs)
    keys_to_plot = [key for key in clf_report.keys() if key not in ('accuracy', 'macro avg', 'weighted avg')]
    df = pd.DataFrame(clf_report, columns=keys_to_plot).T
    #the following line ensures that dataframe are sorted from the majority classes to the minority classes
    df.sort_values(by=['support'], inplace=True) 
    
    #first, let's plot the heatmap by masking the 'support' column
    rows, cols = df.shape
    mask = np.zeros(df.shape)
    mask[:,cols-1] = True
    
    bounds = np.linspace(0, 1, 11)
    cmap = plt.cm.get_cmap('YlOrRd', len(bounds)+1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    ax = sns.heatmap(df, mask=mask, annot=False, cmap=cmap, fmt='.3g',
            cbar_kws={'ticks':bounds[::2], 'norm':norm, 'boundaries':bounds},
            vmin=0.0,
            vmax=1.0,
            linewidths=2, linecolor='white'
                    )
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_ticks_position('both')
    
    cmap_min, cmap_max = cbar.cmap(0), cbar.cmap(1.0)
    
    # add text annotation to heatmap
    dx, dy = 0.5, 0.5
    for i in range(rows):
        for j in range(cols-1):
            text = f"{df.iloc[i, j]:.2%}" #if (j<cols) else f"{df.iloc[i, j]:.0f}"
            ax.text(j + dx , i + dy, text,
                    # ha="center", va="center", color='black')
                    ha="center", va="center", color=cmap_min if df.iloc[i, j] > 0.5 else cmap_max)
    
    #then, let's add the support column by normalizing the colors in this column
    mask = np.zeros(df.shape)
    mask[:,:cols-1] = True    
    
    ax = sns.heatmap(df, mask=mask, annot=False, cmap=cmap, cbar=False,
            linewidths=2, linecolor='white', fmt='.0f',
            vmin=df['support'].min(),
            vmax=df['support'].sum(),         
            norm=mpl.colors.Normalize(vmin=df['support'].min(),
                                      vmax=df['support'].sum())
                ) 
    
    cmap_min, cmap_max = cbar.cmap(0), cbar.cmap(1.0)
    for i in range(rows):
        j = cols-1
        text = f"{df.iloc[i, j]:.0f}" #if (j<cols) else f"{df.iloc[i, j]:.0f}"
        color = (df.iloc[i, j]) / (df['support'].sum())
        ax.text(j + dx , i + dy, text,
                # ha="center", va="center", color='black')
                ha="center", va="center", color=cmap_min if color > 0.5 else cmap_max)
            
    plt.title(title)
    plt.xticks(rotation = 45)
    plt.yticks(rotation = 360)
    plt.tight_layout()
         
    if (save_fig_path != None):
        path = Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches='tight')
    
    return fig, ax





def plot_roc_curve(y_true: torch.Tensor, y_score: torch.Tensor, figsize=(5,5), save_fig_path=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5,5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.scatter(fpr, tpr, marker='o', alpha=0.1, facecolors='None', edgecolors='C0')
    plt.fill_between(fpr, tpr, alpha=0.2, color='C0')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlim([0.0, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right", frameon=False)
    ax.spines[:].set_visible(False)
    ax.grid(True, linestyle='-', linewidth=0.5, color='grey', alpha=0.5)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    plt.tight_layout()
    
    if (save_fig_path != None):
        path = Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches='tight')
    return fig, roc_auc



def plot_calibration_curve(y_prob: np.ndarray, y_true: np.ndarray, save_fig_path=None):
    """
    Creates calibration plot for a binary classifier.

    Parameters
    ----------
    y_prob : np.ndarray
        The output probabilities of the classifier. Between 0 and 1.
    y_true : np.ndarray
        The actual labels of the data. Either 0 or 1.
    save_fig_path : _type_, optional
        Path to folder where the figure should be saved. If None then plot is not saved, by default None

    Returns
    -------
    fig : matplotlib.pyplot figure
        The figure of the calibration plot
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    expected_cal_error = np.abs(prob_pred-prob_true).mean().round(2)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    
    # Calculate bar width
    bar_width = (prob_pred[1:] - prob_pred[:-1]).mean() * 0.75
    
    # Plotting
    ax.bar(prob_pred, prob_true, width=bar_width, zorder=3, facecolor=to_rgba('C0',0.75), edgecolor='midnightblue', linewidth=2, label=f'True Calibration')
    ax.bar(prob_pred, prob_pred - prob_true, bottom=prob_true, width=bar_width, zorder=3, alpha=0.5, edgecolor='red', fill=False, linewidth=2, label=f'Error, mean = {expected_cal_error}', hatch='//')
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', zorder=3, label='Perfect Calibration')
        
    # Labels and titles
    ax.set(xlabel='Predicted probability', ylabel='True probability')
    plt.xlim([0.0, 1.005])
    plt.ylim([-0.01, 1.0])
    ax.legend(loc='upper left', frameon=False)
    
    # show y-grid
    ax.spines[:].set_visible(False)
    ax.grid(True, linestyle='-', linewidth=0.5, color='grey', alpha=0.5)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    plt.tight_layout()
    
    # save plot
    if (save_fig_path != None):
        path = Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches='tight')
    
    return fig


def plot_y_prob_histogram(y_prob: np.ndarray, save_fig_path=None):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.hist(y_prob, bins=10, alpha=0.9, edgecolor='midnightblue', linewidth=2, rwidth=1)
    # same histogram as above, but with border lines
    # ax.hist(y_prob, bins=10, alpha=0.5, edgecolor='black', linewidth=1.2)
    ax.set(xlabel='Predicted probability [-]', ylabel='Count [-]', xlim=(-0.01, 1.0))
    ax.set_title('Histogram of predicted probabilities')
    
    ax.spines[:].set_visible(False)
    ax.grid(True, linestyle='-', linewidth=0.5, color='grey', alpha=0.5)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    plt.tight_layout()
    
    # save plot
    if (save_fig_path != None):
        path = Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches='tight')
    
    return fig
