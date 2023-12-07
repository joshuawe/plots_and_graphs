from typing import Optional, List, Callable, Dict
from tqdm import tqdm
from sklearn.utils import resample
import numpy as np

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