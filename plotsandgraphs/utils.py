from typing import Optional, List, Callable, Dict, Tuple, Union
from tqdm import tqdm
from sklearn.utils import resample
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import BoxStyle
from matplotlib.colors import LinearSegmentedColormap


def bootstrap(metric_function: Callable, input_resample: List[np.ndarray], n_bootstraps: int, metric_kwargs: Dict={}) -> List:
    """
    A bootstrapping function for a metric function. The metric function should take the same number of arguments as the length of input_resample.

    Parameters
    ----------
    metric_function : Callable
        The metric function to use. Should take the same number of arguments as the length of input_resample.
    input_resample : List[np.ndarray]
        A list of arrays to resample. The arrays should have the same length. The arrays are passed to the metric function.
    n_bootstraps : int
        The number of bootstrap iterations.
    metric_kwargs : Dict
        Keyword arguments to pass to the metric function.

    Returns
    -------
    List
        A list of the metric function results for each bootstrap iteration.
    """
    results = []
    # for each bootstrap iteration
    for _ in tqdm(range(n_bootstraps), desc='Bootsrapping', leave=True):
        # resample indices with replacement
        indices = resample(np.arange(len(input_resample[0])), replace=True)
        input_resampled = [x[indices] for x in input_resample]
        # calculate metric
        result = metric_function(*input_resampled, **metric_kwargs)
        
        results.append(result)
        
    return results



class ExtendedTextBox_v2:
    """
    Black background boxes for titles in maptlolib subplots
    
    From:
    https://stackoverflow.com/questions/40796117/how-do-i-make-the-width-of-the-title-box-span-the-entire-plot
    https://matplotlib.org/stable/gallery/userdemo/custom_boxstyle01.html?highlight=boxstyle+_style_list
    """

    def __init__(self, pad=0.3, width=500.):
        """
        The arguments must be floats and have default values.

        Parameters
        ----------
        pad : float
            amount of padding
        """
        self.width = width
        self.pad = pad
        super().__init__()

    def __call__(self, x0, y0, width, height, mutation_size):
        """
        Given the location and size of the box, return the path of the box
        around it.

        Rotation is automatically taken care of.

        Parameters
        ----------
        x0, y0, width, height : float
            Box location and size.
        mutation_size : float
            Reference scale for the mutation, typically the text font size.
        """
        # padding
        pad = mutation_size * self.pad
        # width and height with padding added
        #width = width + 2.*pad
        height = height +  3 * pad
        # boundary of the padded box
        y0 = y0 - pad  # change this to move the text
        y1 = y0 + height 
        _x0 = x0
        x0 = _x0 +width /2. - self.width/2.
        x1 = _x0 +width /2. + self.width/2.
        # return the new path
        return Path([(x0, y0),
                     (x1, y0), (x1, y1), (x0, y1),
                     (x0, y0)],
                    closed=True)


def set_black_title_box(ax: "maptlotlib.axes.Axes", title=str, backgroundcolor='black', color='white', set_title_kwargs: Dict={}):
    """
    Sets the title of the given axes with a black bounding box.
    Note: When using `plt.tight_layout()` the box might not have the correct width. First call `plt.tight_layout()` and then `set_black_title_box()`.

    Parameters:
    - ax: The matplotlib.axes.Axes object to set the title for.
    - title: The title string to be displayed.
    - backgroundcolor: The background color of the title box (default: 'black').
    - color: The color of the title text (default: 'white').
    - set_title_kwargs: Keyword arguments to pass to `ax.set_title()`.
    """
    BoxStyle._style_list["ext"] = ExtendedTextBox_v2 
    ax_width = ax.get_window_extent().width
    # make title with black bounding box
    title = ax.set_title(title, backgroundcolor=backgroundcolor, color=color, **set_title_kwargs)
    bb = title.get_bbox_patch() # get bbox from title
    bb.set_boxstyle("ext", pad=0.1, width=ax_width) # use custom style
    
    
def scale_ax_bbox(ax: "maptlotlib.axes.Axes", factor: float):
    # Get the current position of the subplot
    box = ax.get_position()

    # Calculate the new width and the adjustment for the x-position
    new_width = box.width * factor
    adjustment = (box.width - new_width) / 2

    # Set the new position
    ax.set_position([box.x0 + adjustment, box.y0, new_width, box.height])
    
    return



def get_cmap(cmap_name: str, n_colors: Optional[int]=None) -> Tuple[LinearSegmentedColormap, Union[None, Tuple]]:
    """
    Loads one of the custom cmaps from the cmaps folder.

    Parameters
    ----------
    cmap_name : str
        The name of the cmap file without the extension.

    Returns
    -------
    Tuple[LinearSegmentedColormap, Union[None, Tuple]]
        A tuple of the cmap and a list of colors if n_colors is not None.
        
    Example
    -------
    >>> cmap_name = 'hawaii'
    >>> cmap, color_list = get_cmap(cmap_name, n_colors=10)
    """
    from pathlib import Path as PathClass
    
    cm_path = PathClass(__file__).parent / ('cmaps/' + cmap_name + '.txt')
    print(cm_path)
    cm_data = np.loadtxt(cm_path)
    cmap_name = cmap_name.split('.')[0]
    cmap = LinearSegmentedColormap.from_list(cmap_name, cm_data)
    if n_colors is not None:
        color_list = cmap(np.linspace(0, 1, n_colors))
    else:
        color_list = None
    return cmap, color_list



    