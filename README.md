<p align="center">
 <img width="300px" src="images/logo.png" align="center" alt="Plots and Graphs" />
 <h2 align="center">Plots and graphs</h2>
 <p align="center">Save your time. Create plots and graphs that look nice. </p>
</p>


[![GitHub commits](https://badgen.net/github/commits/joshuawe/plots_and_graphs)](https://GitHub.com/joshuawe/plots_and_graphs/commits)
[![GitHub contributors](https://img.shields.io/github/contributors/joshuawe/plots_and_graphs.svg)](https://GitHub.com/Naereen/badges/graphs/contributors/)
[![GitHub issues](https://badgen.net/github/issues/joshuawe/plots_and_graphs/)](https://GitHub.com/joshuawe/plots_and_graphs/issues/)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[![Open Source Love png2](https://badges.frapsoft.com/os/v2/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)


The repository will contain a collection of useful, useable, and reuseable plots and graphs.

I know, ChatGPT is your friend and helper and with the right prompt can create any graph you would like to have. 
However, it is very helpful to configure everything once and properly and only finetune the detail is necessary.

So there is a collection for different types of plots and graphs. 
Some might be great for visualizing data in general such as raincloud graphs or ridgeline plots while others might be more suitable for specific use cases. 
These could include visualizing the results for a binary classifier, for which plots such as a confusion matrix or a callibration plot can be created.

# Overview of graphs and plots
>  TBD!

- **binary classifier**
    - Accuracy
    - Calibration Curve
    - Classification Report
    - Confusion Matrix
    - ROC curve (AUROC)
    - y_prob histogram


- **comparing distributions**
    - raincloud plot

# Gallery

> TBD! Here, each new plot should be rendered and included as a small reference.

| <img src="/images/calibration_plot.png" width="300" height="300" alt="Your Image"> | <img src="/images/classification_report.png" width="300" height="150" alt="Your Image"> | <img src="/images/confusion_matrix.png" width="300" height="250" alt="Your Image"> |
|:--------------------------------------------------:|:----------------------------------------------------------:|:-------------------------------------------------:|
|                    Calibration Curve               |                  Classification Report                     |                 Confusion Matrix                 |

| <img src="/images/roc_curve.png" width="300" height="300" alt="Your Image">        | <img src="/images/y_prob_histogram.png" width="300" height="300" alt="Your Image">  |                                                   |
|:--------------------------------------------------:|:----------------------------------------------------------:|:-------------------------------------------------:|
|                    ROC Curve (AUROC)               |                  y_prob histogram                          |                                                   |






# Other resources

Why create everything from scratch, if some things exist already? Here are some helpful resources.

+ [Python Graph Gallery](https://python-graph-gallery.com) - great collection of graphs. First look at their gallery, then check out their '[BEST](https://python-graph-gallery.com/best-python-chart-examples/)' collection for inspiration.
+ [Scientific Visualization Book](https://github.com/rougier/scientific-visualization-book) - definitely check out the first part for essential tips for good graphs. And deep dive further to improve your visualization game.


# How to use
> TBD: How to use the functions. What is their standardization? How can a figure be altered?

```python
import plotsandgraphs
import matplotlib.pyplot as plt
import numpy as np

# create some data
n_samples = 1000
y_prob = np.random.uniform(0, 1, n_samples)   # the probability of class 1 predictions
y_true = np.random.choice([0,1], n_samples)   # the actually corresponding class labels

# create figure
fig_auroc = plotsandgraphs.binaryclassifier.plot_calibration_curve(y_prob, y_true, save_fig_path=None)

# customize figure
axes = fig_auroc.get_axes()
ax0 = axes[0]
ax0.set_title('New Title for Calibration Plot')

# save figure
fig.savefig('calibration_plot.png', bbox_inches='tight')
```

# Requirements
> Show all requirements


## Contributors

![Contributors](https://contrib.rocks/image?repo=joshuawe/plots_and_graphs)
