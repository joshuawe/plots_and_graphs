from typing import Literal, Union
from pathlib import Path
from tqdm.auto import tqdm

from . import binary_classifier as bc
from . import multiclass_classifier as mc


FILE_ENDINGS = Literal["pdf", "png", "jpg", "jpeg", "svg"]


def binary_classifier(
    y_true, y_score, save_fig_path=None, plot_kwargs={}, file_type: FILE_ENDINGS = "png"
):
    # Create new tqdm instance
    tqdm_instance = tqdm(total=6, desc="Binary classifier metrics", leave=True)

    # Update tqdm instance
    tqdm_instance.update()

    # 1) Plot ROC curve
    roc_kwargs = plot_kwargs.get("roc", {})
    save_path = get_file_path(save_fig_path, "roc_curve", file_type)
    bc.plot_roc_curve(y_true, y_score, save_fig_path=save_path, **roc_kwargs)
    tqdm_instance.update()

    # 2) Plot precision-recall curve
    pr_kwargs = plot_kwargs.get("pr", {})
    save_path = get_file_path(save_fig_path, "pr_curve", file_type)
    bc.plot_pr_curve(y_true, y_score, save_fig_path=save_path, **pr_kwargs)
    tqdm_instance.update()

    # 3) Plot calibration curve
    cal_kwargs = plot_kwargs.get("cal", {})
    save_path = get_file_path(save_fig_path, "calibration_curve", file_type)
    bc.plot_calibration_curve(y_true, y_score, save_fig_path=save_path, **cal_kwargs)
    tqdm_instance.update()

    # 3) Plot confusion matrix
    cm_kwargs = plot_kwargs.get("cm", {})
    save_path = get_file_path(save_fig_path, "confusion_matrix", file_type)
    bc.plot_confusion_matrix(y_true, y_score, save_fig_path=save_path, **cm_kwargs)
    tqdm_instance.update()

    # 5) Plot classification report
    cr_kwargs = plot_kwargs.get("cr", {})
    save_path = get_file_path(save_fig_path, "classification_report", file_type)
    bc.plot_classification_report(y_true, y_score, save_fig_path=save_path, **cr_kwargs)
    tqdm_instance.update()

    # 6) Plot y_score histogram
    hist_kwargs = plot_kwargs.get("hist", {})
    save_path = get_file_path(save_fig_path, "y_score_histogram", file_type)
    bc.plot_y_score_histogram(y_true, y_score, save_fig_path=save_path, **hist_kwargs)
    tqdm_instance.update()

    return


def multiclass_classifier(
    y_true, y_score, save_fig_path=None, plot_kwargs={}, file_type: FILE_ENDINGS = "png"
):
    # Create new tqdm instance
    tqdm_instance = tqdm(total=6, desc="Binary classifier metrics", leave=True)

    # Update tqdm instance
    tqdm_instance.update()

    # 1) Plot ROC curve
    roc_kwargs = plot_kwargs.get("roc", {})
    save_path = get_file_path(save_fig_path, "roc_curve", "")
    mc.plot_roc_curve(y_true, y_score, save_fig_path=save_path, **roc_kwargs)
    tqdm_instance.update()

    # 2) Plot precision-recall curve
    # pr_kwargs = plot_kwargs.get('pr', {})
    # mc.plot_pr_curve(y_true, y_score, save_fig_path=save_fig_path, **pr_kwargs)
    # tqdm_instance.update()

    # 3) Plot calibration curve
    # cal_kwargs = plot_kwargs.get('cal', {})
    # mc.plot_calibration_curve(y_true, y_score, save_fig_path=save_fig_path, **cal_kwargs)
    # tqdm_instance.update()

    # 3) Plot confusion matrix
    # cm_kwargs = plot_kwargs.get('cm', {})
    # mc.plot_confusion_matrix(y_true, y_score, save_fig_path=save_fig_path, **cm_kwargs)
    # tqdm_instance.update()

    # 5) Plot classification report
    # cr_kwargs = plot_kwargs.get('cr', {})
    # mc.plot_classification_report(y_true, y_score, save_fig_path=save_fig_path, **cr_kwargs)
    # tqdm_instance.update()

    # 6) Plot y_score histogram
    hist_kwargs = plot_kwargs.get("hist", {})
    save_path = get_file_path(save_fig_path, "y_score_histogram", file_type)
    mc.plot_y_score_histogram(y_true, y_score, save_fig_path=save_path, **hist_kwargs)
    tqdm_instance.update()

    return


def get_file_path(save_fig_path: Union[Path, None, str], name: str, ending: str):
    if save_fig_path is None:
        return None
    else:
        result = Path(save_fig_path) / f"{name}.{ending}"
        print(result)
        return str(result)
