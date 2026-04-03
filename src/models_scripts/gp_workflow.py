from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    WhiteKernel,
    ConstantKernel as C,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class BaseGPApproach:

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model_: Optional[RegressorMixin] = None
        self.results_: Dict[str, Any] = {}

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("The model has not been fitted yet.")
        return self.model_.predict(X)

    def evaluate(self, X, y) -> Dict[str, float]:
        """
        Evaluate the fitted model on any dataset.
        """
        y_pred = self.predict(X)

        metrics = {
            "r2": r2_score(y, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "mae": mean_absolute_error(y, y_pred),
        }
        return metrics


@dataclass
class ExperimentalGPApproach(BaseGPApproach):
    """
    Standard Gaussian Process workflow for experimental data only.

    Features:
    - sklearn Pipeline
    - scaler selection with GridSearchCV
    - kernel selection
    - alpha tuning
    - CV score tracking
    """

    random_state: int = 42
    n_splits: int = 5
    shuffle: bool = True
    n_jobs: int = -1
    verbose: int = 2
    scoring: str = "r2"
    normalize_y: bool = True
    n_restarts_optimizer: int = 5
    alpha_grid: List[float] = field(default_factory=lambda: [1e-8, 1e-6, 1e-4])

    def __post_init__(self):
        super().__init__(random_state=self.random_state)
        self.grid_: Optional[GridSearchCV] = None

    @staticmethod
    def default_kernels() -> List:
        """
        Default kernel list for GP model selection.
        """
        return [
            C(1.0, (1e-3, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1)),

            C(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=1.0,
                length_scale_bounds=(1e-2, 1e2),
                nu=1.5,
            )
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1)),

            C(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=1.0,
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5,
            )
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1)),

            C(1.0, (1e-3, 1e3))
            * RationalQuadratic(length_scale=1.0, alpha=1.0)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1)),
        ]

    def build_pipeline(self) -> Pipeline:
        """
        Build the basic sklearn pipeline.
        """
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "gp",
                GaussianProcessRegressor(
                    normalize_y=self.normalize_y,
                    n_restarts_optimizer=self.n_restarts_optimizer,
                    random_state=self.random_state,
                ),
            ),
        ])
        return pipeline

    def build_param_grid(self, kernels: Optional[List] = None) -> Dict[str, List[Any]]:
        """
        Build the parameter grid for GridSearchCV.
        """
        if kernels is None:
            kernels = self.default_kernels()

        param_grid = {
            "scaler": [StandardScaler(), MinMaxScaler(), "passthrough"],
            "gp__kernel": kernels,
            "gp__alpha": self.alpha_grid,
        }
        return param_grid

    def build_cv(self) -> KFold:
        """
        Build the cross-validation splitter.
        """
        return KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

    def fit(
        self,
        X_train,
        y_train,
        kernels: Optional[List] = None,
    ) -> "ExperimentalGPApproach":
        """
        Fit the GridSearchCV workflow and store the best estimator.
        """
        pipeline = self.build_pipeline()
        param_grid = self.build_param_grid(kernels=kernels)
        cv = self.build_cv()

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=self.scoring,
            cv=cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            return_train_score=True,
        )

        grid.fit(X_train, y_train)

        self.grid_ = grid
        self.model_ = grid.best_estimator_
        self.results_ = {
            "best_score_cv": grid.best_score_,
            "best_params": grid.best_params_,
            "best_estimator": grid.best_estimator_,
        }
        return self

    def fit_and_evaluate(
        self,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        kernels: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Fit on training data and optionally evaluate on train/test.
        """
        self.fit(X_train, y_train, kernels=kernels)

        output = {
            "best_score_cv": self.results_["best_score_cv"],
            "best_params": self.results_["best_params"],
            "train_metrics": self.evaluate(X_train, y_train),
        }

        if X_test is not None and y_test is not None:
            output["test_metrics"] = self.evaluate(X_test, y_test)

        return output

    def get_cv_results(self):
        """
        Return GridSearchCV results as a dataframe.
        """
        if self.grid_ is None:
            raise ValueError("You need to fit the model before accessing CV results.")

        import pandas as pd

        return pd.DataFrame(self.grid_.cv_results_)


# ---------------------------------------------------------------------
# Future classes can be added below
# ---------------------------------------------------------------------


class ResidualGPApproach(BaseGPApproach):
    """
    Placeholder for future residual-based GP workflow.
    """

    def fit(self, X, y):
        raise NotImplementedError("ResidualGPApproach is not implemented yet.")


class CustomMeanGPyApproach(BaseGPApproach):
    """
    Placeholder for future GPy workflow with custom mean function.
    """

    def fit(self, X, y):
        raise NotImplementedError("CustomMeanGPyApproach is not implemented yet.")


if __name__ == "__main__":
    # Minimal usage example
    # Replace X_train, y_train, X_test, y_test with your own data.
    #
    # gp_exp = ExperimentalGPApproach(
    #     random_state=42,
    #     n_splits=5,
    #     scoring="r2",
    # )
    #
    # results = gp_exp.fit_and_evaluate(
    #     X_train=X_train,
    #     y_train=y_train,
    #     X_test=X_test,
    #     y_test=y_test,
    # )
    #
    # print("Best CV score:", results["best_score_cv"])
    # print("Best params:")
    # print(results["best_params"])
    # print("Train metrics:", results["train_metrics"])
    # print("Test metrics:", results.get("test_metrics"))
    pass
