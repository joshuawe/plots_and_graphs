<p align="center">
 <img width="150px" src="https://github.com/joshuawe/plots_and_graphs/blob/main/images/logo_plots_and_graphs.png?raw=true" align="center" alt="Plots and Graphs" />
 <h2 align="center">Plots and graphs</h2>
 <p align="center">Visualizations of Machine Learning Metrics quick and simple. Save your time.</p>
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


Every Machine Learning project requires the evaluation of the algorithm's performance. Many metrics are task specific (classification, regression, ...) but are used again and again and have to be plotted again and again. **Plotsandgraphs** makes it easier for you to visualize these metrics by providing a library with tidy and clear graphs for the most common metrics. It serves as a wrapper for the popular *matplotlib* package to create figures. This also allows users to apply custom changes, if necessary. 

**Plotsandgraphs** is model- and framework-agnostic. This means that **plotsandgraphs** only needs the algorithm's results to perform analysis and visualization and not the actual model itself. In the case of a binary classifier only the *true labels* as well as the *predicted probabilities* are required as input. Instead of spending time in visualizing results for each metric, **plotsandgraphs** can calculate and visualize all classification metrics with a single line of code. 

**Plotsandgraphs** provides analysis and visualization for the following problem types:
- **binary classification**
- *multi-class classification (coming soon)*
- *regression (coming soon)*

Furthermore, this library presents other useful visualizations, such as **comparing distributions**.


# Overview of graphs and plots

- **binary classifier**
    - Accuracy
    - Calibration Curve
    - Classification Report
    - Confusion Matrix
    - ROC curve (AUROC)
    - y_prob histogram

- *multi-class classifier*

- *regression*


- **comparing distributions**
    - raincloud plot

# Gallery

| <img src="https://github.com/joshuawe/plots_and_graphs/blob/main/images/calibration_plot.png?raw=true" width="300" alt="Your Image"> | <img src="https://github.com/joshuawe/plots_and_graphs/blob/main/images/classification_report.png?raw=true" width="300" alt="Your Image"> | <img src="https://github.com/joshuawe/plots_and_graphs/blob/main/images/confusion_matrix.png?raw=true" width="300" alt="Your Image"> |
|:--------------------------------------------------:|:----------------------------------------------------------:|:-------------------------------------------------:|
|                    Calibration Curve               |                  Classification Report                     |                 Confusion Matrix                 |

| <img src="https://github.com/joshuawe/plots_and_graphs/blob/main/images/roc_curve_bootstrap.png?raw=true" width="300" alt="Your Image">        | <img src="https://github.com/joshuawe/plots_and_graphs/blob/main/images/pr_curve.png?raw=true" width="300" alt="Your Image">        | <img src="https://github.com/joshuawe/plots_and_graphs/blob/main/images/y_prob_histogram.png?raw=true" width="300" alt="Your Image">  |
|:--------------------------------------------------:|:----------------------------------------------------------:|:-------------------------------------------------:|
|                    ROC Curve (AUROC) with bootstrapping             |                 Precision-Recall Curve                          |                  y_prob histogram                                 |


| <img src="https://github.com/joshuawe/plots_and_graphs/blob/main/images/raincloud.png?raw=true" width="300" alt="Your Image">        |  <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" width="300" height="300" alt=""> | <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" width="300" height="300" alt=""> |
|:--------------------------------------------------:|:-------------------------------------------------:| :-------------------------------------------------:|
|                    Raincloud              |                                                | |



# Other resources

Why create everything from scratch, if some things already exist? Here are some helpful resources that can improve your visualization skills.

+ [Python Graph Gallery](https://python-graph-gallery.com) - A great collection of graphs. First look at their gallery, then check out their '[BEST](https://python-graph-gallery.com/best-python-chart-examples/)' collection for inspiration.
+ [Scientific Visualization Book](https://github.com/rougier/scientific-visualization-book) - Definitely check out the first part for essential tips for good graphs. And then deep dive further to improve your visualization game.
+ [CHARTIO](https://chartio.com/learn/charts/how-to-choose-colors-data-visualization/) - A must read on how to choose colors and color palettes.


# Install

Install the package via pip.
```bash
pip install plotsandgraphs
```

Alternatively install the package from git.
```bash
git clone https://github.com/joshuawe/plots_and_graphs
cd plots_and_graphs
pip install -e .
```

# Usage

Example usage of results from a binary classifier for a calibration curve.

```python
import matplotlib.pyplot as plt
import numpy as np
import plotsandgraphs as pandg

# create some predictions of a hypothetical binary classifier
n_samples = 1000
y_true = np.random.choice([0,1], n_samples, p=[0.4, 0.6])   # the true class labels 0 or 1, with class imbalance 40:60

y_prob = np.zeros(y_true.shape)   # a model's probability of class 1 predictions
y_prob[y_true==1] = np.random.beta(1, 0.6, y_prob[y_true==1].shape)
y_prob[y_true==0] = np.random.beta(0.5, 1, y_prob[y_true==0].shape)

# show prob distribution
fig_hist = pandg.binary_classifier.plot_y_prob_histogram(y_prob, y_true, save_fig_path=None)

# create calibration curve
fig_auroc = pandg.binary_classifier.plot_calibration_curve(y_prob, y_true, save_fig_path=None)


# --- OPTIONAL: Customize figure ---
# get axis of figure and change title
axes = fig_auroc.get_axes()
ax0 = axes[0]
ax0.set_title('New Title for Calibration Plot')
fig_auroc.show()
```

# Requirements
> Show all requirements


# Contributors

![Contributors](https://contrib.rocks/image?repo=joshuawe/plots_and_graphs)

+ [DALL-E 3](https://openai.com/dall-e-3) created the project logo on 17th October 2023. Prompt used: *Illustration of a stylized graph with colorful lines and bars, representing data visualization, suitable for a project logo named 'plots and graphs'.*


# Reference

Of course we are happy to be mentioned in any way, if our repository has helped you.
You can also share this repository with your friends and collegues to make their lives easier. Cheers!
