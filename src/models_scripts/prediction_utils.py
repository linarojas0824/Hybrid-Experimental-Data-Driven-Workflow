
from utils.paths import EXP_RESULT,MODELS_DIR
from pathlib import Path
import pandas as pd
import yaml
import joblib


def save_experiment_outputs(
    exp_name,
    results,
    model,
    x_train,
    x_test,
    y_test,
    split_dict,
    results_dir=EXP_RESULT,
    space_predictions_df=None,
):
    """
    Save metrics, test predictions, summary yaml, model,
    and optional composition-space predictions.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    # -----------------------------
    # Save metrics
    # -----------------------------
    metrics_path = results_dir / "metrics_all.csv"
    
    metrics_df = pd.DataFrame([{
        "experiment": exp_name,
        "best_score_cv": results["best_score_cv"],
        "train_r2": results["train_metrics"]["r2"],
        "train_mse": results["train_metrics"]["mse"],
        "train_rmse": results["train_metrics"]["rmse"],
        "train_mae": results["train_metrics"]["mae"],
        "test_r2": results["test_metrics"]["r2"],
        "test_mse": results["test_metrics"]["mse"],
        "test_rmse": results["test_metrics"]["rmse"],
        "test_mae": results["test_metrics"]["mae"],

    }])

    # append if exists, otherwise create
    if metrics_path.exists():
        metrics_df.to_csv(metrics_path, mode="a", header=False, index=False)
    else:
        metrics_df.to_csv(metrics_path, mode="w", header=True, index=False)
    # -----------------------------
    # Save test predictions
    # -----------------------------
    y_test_pred, y_test_std = model.predict(x_test, return_std=True)

    predictions_df = split_dict["X_test"][["Cu", "Ni", "Al"]].copy()
    predictions_df["y_true"] = y_test.values
    predictions_df["y_pred"] = y_test_pred
    predictions_df["y_std"] = y_test_std

    predictions_df.to_csv(results_dir / f"{exp_name}_predictions.csv", index=False)

    # -----------------------------
    # Save summary YAML
    # -----------------------------
    summary = {
        "experiment": exp_name,
        "best_score_cv": float(results["best_score_cv"]),
        "best_params": results["best_params"],
        "train_metrics": {k: float(v) for k, v in results["train_metrics"].items()},
        "test_metrics": {k: float(v) for k, v in results["test_metrics"].items()},
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }

    with open(results_dir / f"{exp_name}_summary.yaml", "w") as f:
        yaml.dump(summary, f, sort_keys=False)

    # -----------------------------
    # Save model
    # -----------------------------
    joblib.dump(model, MODELS_DIR / f"{exp_name}_model.pkl")

    # -----------------------------
    # Save whole composition-space predictions
    # -----------------------------
    if space_predictions_df is not None:
        space_predictions_df.to_csv(
            results_dir / f"{exp_name}_space_predictions.csv",
            index=False
        )