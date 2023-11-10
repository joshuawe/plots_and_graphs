import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from typing import List, Tuple, Optional


def plot_raincloud(df: pd.DataFrame,
                   x_col: str,
                   y_col: str, 
                   colors: Optional[List[str]] = None, 
                   order: Optional[List[str]] = None, 
                   title: Optional[str] = None, 
                   x_label: Optional[str] = None, 
                   x_range: Optional[Tuple[float, float]] = None, 
                   show_violin = True, 
                   show_scatter = True, 
                   show_boxplot = True):
    
    """
    Generate a raincloud plot using Pandas DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The data frame containing the data.
    - x_col (str): The column name for the x-axis data.
    - y_col (str): The column name for the y-axis categories.
    - colors (List[str], optional): List of colors for each category. Defaults to tab10 cmap.
    - order (List[str], optional): Order of categories on y-axis. Defaults to unique values in y_col.
    - title (str, optional): Title of the plot.
    - x_label (str, optional): Label for the x-axis.
    - x_range (Tuple[float, float], optional): Range for the x-axis.
    - show_violin (bool, optional): Whether to show violin plot. Defaults to True.
    - show_scatter (bool, optional): Whether to show scatter plot. Defaults to True.
    - show_boxplot (bool, optional): Whether to show boxplot. Defaults to True.

    Returns:
    - matplotlib.figure.Figure: The generated plot figure.
    """
        
    fig, ax = plt.subplots(figsize=(16, 8))
    offset = 0.2  # Offset value to move plots

    if order is None:
        order = df[y_col].unique()

    # if colors are none, use distinct colors for each group
    if colors is None:
        cmap = plt.get_cmap('tab10')
        colors = [mpl.colors.to_hex(cmap(i)) for i in np.linspace(0, 1, len(order))]
    else:
        assert len(colors) == len(order), 'colors and order must be the same length'
        colors = colors
        
    # Boxplot
    if show_boxplot:
        bp = ax.boxplot([df[df[y_col] == grp][x_col].values for grp in order],
                        patch_artist=True, vert=False, positions=np.arange(1 + offset, len(order) + 1 + offset), widths=0.2)

        # Customize boxplot colors
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        # Set median line color to black
        for median in bp['medians']:
            median.set_color('black')

    # Violinplot
    if show_violin:
        vp = ax.violinplot([df[df[y_col] == grp][x_col].values for grp in order],
                        positions=np.arange(1 + offset, len(order) + 1 + offset), showmeans=False, showextrema=False, showmedians=False, vert=False)

        # Customize violinplot colors
        for idx, b in enumerate(vp['bodies']):
            b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx + 1 + offset, idx + 2 + offset)
            b.set_color(colors[idx])

    # Scatterplot with jitter
    if show_scatter:
        for idx, grp in enumerate(order):
            features = df[df[y_col] == grp][x_col].values
            y = np.full(len(features), idx + 1 - offset)
            jitter_amount = 0.12
            y += np.random.uniform(low=-jitter_amount, high=jitter_amount, size=len(y))
            plt.scatter(features, y, s=10, c=colors[idx], alpha=0.3, facecolors='none')

    # Labels
    plt.yticks(np.arange(1, len(order) + 1), order)

    if x_label is None:
        x_label = x_col
    plt.xlabel(x_label)
    if title:
        plt.title(title + '\n')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.grid(True)

    if x_range:
        plt.xlim(x_range)
        
    return fig
