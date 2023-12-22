from . import binary_classifier as bc

from tqdm.auto import tqdm



def binary_classifier(y_true, y_score, save_fig_path=None, plot_kwargs={}):
    
    # Create new tqdm instance
    tqdm_instance = tqdm(total=7, desc='Binary classifier metrics')
    
    # Update tqdm instance
    tqdm_instance.update()
    
    # 1) Plot ROC curve
    roc_kwargs = plot_kwargs.get('roc', {})
    bc.plot_roc_curve(y_true, y_score, save_fig_path=save_fig_path, **roc_kwargs)
    
    # 2) Plot precision-recall curve
    pr_kwargs = plot_kwargs.get('pr', {})
    bc.plot_pr_curve(y_true, y_score, save_fig_path=save_fig_path, **pr_kwargs)
    
    # 3) Plot calibration curve
    cal_kwargs = plot_kwargs.get('cal', {})
    bc.plot_calibration_curve(y_true, y_score, save_fig_path=save_fig_path, **cal_kwargs)
    
    # 3) Plot confusion matrix
    cm_kwargs = plot_kwargs.get('cm', {})
    bc.plot_confusion_matrix(y_true, y_score, save_fig_path=save_fig_path, **cm_kwargs)
    
    # 5) Plot classification report
    cr_kwargs = plot_kwargs.get('cr', {})
    bc.plot_classification_report(y_true, y_score, save_fig_path=save_fig_path, **cr_kwargs)
    
    # 6) Plot y_score histogram
    hist_kwargs = plot_kwargs.get('hist', {})
    bc.plot_y_prob_histogram(y_true, y_score, save_fig_path=save_fig_path, **hist_kwargs)
    
    return