<p align="center">
 <img width="150px" src="https://github.com/joshuawe/plots_and_graphs/blob/main/images/logo_plots_and_graphs.png?raw=true" align="center" alt="Plots and Graphs" />
 <h2 align="center">Plots and graphs</h2>
 <p align="center">Save your time. Create plots and graphs that look nice. </p>
</p>

<p align="center">
  <a href="https://GitHub.com/joshuawe/plots_and_graphs/commits">
    <img src="https://badgen.net/github/commits/joshuawe/plots_and_graphs" alt="GitHub commits">
  </a>
  <a href="https://GitHub.com/Naereen/badges/graphs/contributors/">
    <img src="https://img.shields.io/github/contributors/joshuawe/plots_and_graphs.svg" alt="GitHub contributors">
  </a>
  <a href="https://GitHub.com/joshuawe/plots_and_graphs/issues/">
    <img src="https://badgen.net/github/issues/joshuawe/plots_and_graphs/" alt="GitHub issues">
  </a>
  <a href="https://github.com/joshuawe/plots_and_graphs/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="GPLv3 license">
  </a>
  <a href="https://github.com/ellerbrock/open-source-badges/">
    <img src="https://badges.frapsoft.com/os/v2/open-source.png?v=103" alt="Open Source Love png2">
  </a>
</p>


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

| <img src="https://github.com/joshuawe/plots_and_graphs/blob/main/images/calibration_plot.png?raw=true" width="300" alt="Your Image"> | <img src="https://github.com/joshuawe/plots_and_graphs/blob/main/images/classification_report.png?raw=true" width="300" alt="Your Image"> | <img src="https://github.com/joshuawe/plots_and_graphs/blob/main/images/confusion_matrix.png?raw=true" width="300" alt="Your Image"> |
|:--------------------------------------------------:|:----------------------------------------------------------:|:-------------------------------------------------:|
|                    Calibration Curve               |                  Classification Report                     |                 Confusion Matrix                 |

| <img src="https://github.com/joshuawe/plots_and_graphs/blob/main/images/roc_curve.png?raw=true" width="300" alt="Your Image">        | <img src="https://github.com/joshuawe/plots_and_graphs/blob/main/images/roc_curve_bootstrap.png?raw=true" width="300" alt="Your Image">        | <img src="https://github.com/joshuawe/plots_and_graphs/blob/main/images/y_prob_histogram.png?raw=true" width="300" alt="Your Image">  |
|:--------------------------------------------------:|:----------------------------------------------------------:|:-------------------------------------------------:|
|                    ROC Curve (AUROC)               |                  ROC Curve (AUROC) with bootstrapping                          |                  y_prob histogram                                 |


| <img src="https://github.com/joshuawe/plots_and_graphs/blob/main/images/raincloud.png?raw=true" width="300" alt="Your Image">        |  <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" width="300" height="300" alt=""> | <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" width="300" height="300" alt=""> |
|:--------------------------------------------------:|:-------------------------------------------------:| :-------------------------------------------------:|
|                    Raincloud              |                                                | |



# Other resources

Why create everything from scratch, if some things exist already? Here are some helpful resources.

+ [Python Graph Gallery](https://python-graph-gallery.com) - great collection of graphs. First look at their gallery, then check out their '[BEST](https://python-graph-gallery.com/best-python-chart-examples/)' collection for inspiration.
+ [Scientific Visualization Book](https://github.com/rougier/scientific-visualization-book) - definitely check out the first part for essential tips for good graphs. And deep dive further to improve your visualization game.
+ [CHARTIO](https://chartio.com/learn/charts/how-to-choose-colors-data-visualization/) - must read on how to choose colors and color palettes.


# How to use
> TBD: How to use the functions. What is their standardization? How can a figure be altered?

Install the package (currently).
```bash
pip install -e .
```

Install the package (in the future).
```bash
pip install plotsandgraphs
```

Example usage for a calibration curve.

```python
import plotsandgraphs
import matplotlib.pyplot as plt
import numpy as np

# create some data
n_samples = 1000
y_prob = np.random.uniform(0, 1, n_samples)   # the probability of class 1 predictions
y_true = np.random.choice([0,1], n_samples)   # the actually corresponding class labels

# create figure
fig_auroc = plotsandgraphs.binary_classifier.plot_calibration_curve(y_prob, y_true, save_fig_path=None)

# customize figure
axes = fig_auroc.get_axes()
ax0 = axes[0]
ax0.set_title('New Title for Calibration Plot')

# save figure
fig.savefig('calibration_plot.png', bbox_inches='tight')
```

# Requirements
> Show all requirements


# Contributors

![Contributors](https://contrib.rocks/image?repo=joshuawe/plots_and_graphs)

+ [DALL-E 3](https://openai.com/dall-e-3) created the project logo on 17th October 2023. Prompt used: *Illustration of a stylized graph with colorful lines and bars, representing data visualization, suitable for a project logo named 'plots and graphs'.*


# Reference

Of course we are happy to be mentioned in any way, if our repository has helped you.
You can also share this repository with your friends and collegues to make their lives easier. Cheers!
