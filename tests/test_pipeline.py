from pathlib import Path
import shutil

from plotsandgraphs import pipeline


from .utils import random_data_binary_classifier, random_data_multiclass_classifier

TEST_RESULTS_PATH = Path("tests/test_results/pipeline")


def test_binary_classification_pipeline(random_data_binary_classifier):
    """
    Test binary classification pipeline.

    Parameters
    ----------
    random_data_binary_classifier : Tuple[np.ndarray, np.ndarray]
        The simulated data.
    """
    save_fig_path = TEST_RESULTS_PATH / "binary_classifier"

    # Delete the folder and its previous contents
    if save_fig_path.exists() and save_fig_path.is_dir():
        shutil.rmtree(save_fig_path)

    y_true, y_score = random_data_binary_classifier
    pipeline.binary_classifier(
        y_true, y_score, save_fig_path=save_fig_path, file_type="png"
    )

    # assert that there are files with the names xy in the save_fig_path
    assert (save_fig_path / "roc_curve.png").exists()
    assert (save_fig_path / "y_score_histogram.png").exists()
    assert (save_fig_path / "calibration_curve.png").exists()
    assert (save_fig_path / "confusion_matrix.png").exists()
    assert (save_fig_path / "pr_curve.png").exists()


def test_multiclassification_pipeline():
    """
    Test multiclassification pipeline.
    """
    for num_classes in [3]:
        save_fig_path = TEST_RESULTS_PATH / f"multiclass_{num_classes}_classes"

        # Delete the folder and its previous contents
        if save_fig_path.exists() and save_fig_path.is_dir():
            shutil.rmtree(save_fig_path)

        y_true, y_score = random_data_multiclass_classifier(num_classes=num_classes)
        pipeline.multiclass_classifier(
            y_true, y_score, save_fig_path=save_fig_path, file_type="png"
        )

        # assert that there are files with the names xy in the save_fig_path
        assert (save_fig_path / "roc_curve.png").exists()
        assert (save_fig_path / "y_score_histogram.png").exists()
