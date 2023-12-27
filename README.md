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
    - y_score histogram

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
|                    ROC Curve (AUROC) with bootstrapping             |                 Precision-Recall Curve                          |                  y_score histogram                                 |


| <img src="https://github.com/joshuawe/plots_and_graphs/blob/main/images/multiclass/histogram_4_classes.png?raw=true" width="300" alt="Your Image">        |  <img src="https://github.com/joshuawe/plots_and_graphs/blob/main/images/multiclass/roc_curves_multiclass.png?raw=true" width="300" alt=""> | <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" width="300" height="300" alt=""> |
|:--------------------------------------------------:|:-------------------------------------------------:| :-------------------------------------------------:|
|                    Histogram (y_scores)              |    ROC curves (AUROC) with bootstrapping                                            | |



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

Alternatively install the package from git directly.
```bash
git clone https://github.com/joshuawe/plots_and_graphs
cd plots_and_graphs
pip install -e .
```

# Usage

Get all classification metrics with **ONE** line of code. Here, for a binary classifier:

```python
import plotsandgraphs as pandg
# ...
pandg.pipeline.binary_classifier(y_true, y_score)
```

Or with some more configs:
```Python
configs = {
  'roc': {'n_bootstraps': 10000},
  'pr': {'figsize': (8,10)}
}
pandg.pipeline.binary_classifier(y_true, y_score, save_fig_path='results/metrics', file_type='png', plot_kwargs=configs)
```

For multiclass classification:

```Python
# with multiclass data y_true (one-hot encoded) and y_score
pandg.pipeline.multiclass_classifier(y_true, y_score)
```

# Requirements
> Show all requirements


# Contributors

![Contributors](https://contrib.rocks/image?repo=joshuawe/plots_and_graphs)

+ [DALL-E 3](https://openai.com/dall-e-3) created the project logo on 17th October 2023. Prompt used: *Illustration of a stylized graph with colorful lines and bars, representing data visualization, suitable for a project logo named 'plots and graphs'.*

+ The [Scientific colour maps](https://www.fabiocrameri.ch/colourmaps/) in the `plotsandgraphs/cmaps` folder [(Crameri 2018)](https://doi.org/10.5281/zenodo.1243862) are used in this library to prevent visual distortion of the data and exclusion of readers with colour-vision deficiencies [(Crameri et al., 2020)](https://www.nature.com/articles/s41467-020-19160-7).


# Reference

Of course we are happy to be mentioned in any way, if our repository has helped you.
You can also share this repository with your friends and collegues to make their lives easier. Cheers!
