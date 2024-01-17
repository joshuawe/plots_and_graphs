"""
    This script holds the figure class that is used to create the figures for the plots and graphs project.
    It helps to keep the code clean and organized.
"""

from typing import Tuple, List, Dict, Union, Any, Optional, Literal
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


Number = Union[int, float]

class SinglePlotFigure():
    
    fig: Figure
    ax: plt.Axes
    
    def __init__(self, 
                figsize: Optional[Tuple[Number, Number]]=None, 
                title: Optional[str]=None,
                labels: Optional[Tuple[str, str]]=None,
                grid: bool=True):
        self.fig = plt.figure(figsize=figsize, tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        
        self.set_title(title)
        self.despine()
        self.set_labels(labels)
        self.set_grid(grid)
        
    def set_labels(self, labels: Optional[Tuple[str, str]]=None):
        if labels is not None:
            self.ax.set_xlabel(labels[0])
            self.ax.set_ylabel(labels[1])
        
    def set_title(self, title: Optional[str]):
        if title is not None:
            self.ax.set_title(title)
        
    def despine(self, spines: Optional[List[str]]=None):
        if spines is None:
            spines = ['top', 'bottom', 'left', 'right']
        self.ax.spines[spines].set_visible(False)
        
    def spine(self, spines: Optional[List[str]]=None):
        if spines is None:
            spines = ['top', 'bottom', 'left', 'right']
        self.ax.spines[spines].set_visible(True)
        
    def set_grid(self, grid: bool=True):
        self.ax.grid(grid, linestyle="-", linewidth=0.5, color="grey", alpha=0.5)
        
    def set_ticks(self, which: Literal['x', 'y'], ticks: Optional[Union[bool, List[Number]]]=None):
        """ Sets the ticks on the given axis. `which` specifies the axis, `ticks` specifies the ticks. For `ticks` choose:
            - None: remove the ticks
            - True: use the default ticks
            - list of numbers: use those as ticks
        """
        # choose which axis to set the ticks on
        if which == 'x':
            set_ticks = self.ax.set_xticks
        elif which == 'y':
            set_ticks = self.ax.set_yticks
        else:
            raise ValueError(f"which must be either 'x' or 'y', received {which}")
        # if x_ticks is None, remove the ticks
        if ticks is None:
            set_ticks([])
        # if x_ticks is True, use the default ticks
        elif ticks is True:
            set_ticks(np.arange(0, 1.1, 0.1))
        # if x_ticks is a list of numbers, use those as ticks
        else:
            set_ticks(ticks)
        
        
    def save_fig(self, path: Optional[Union[str, Path]]=None):
        self.fig.tight_layout()
        if path is not None:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(save_path, bbox_inches="tight")
    


if __name__ == '__main__':
    figsize = (5, 5)
    title = "Random title"
    labels = ("x", "y")
    spfig = SinglePlotFigure(figsize=figsize, title=title, labels=labels)
    spfig.set_ticks('x', True)
    spfig.set_ticks('y', True)
    spfig.fig.show()
    plt.show(block=True)