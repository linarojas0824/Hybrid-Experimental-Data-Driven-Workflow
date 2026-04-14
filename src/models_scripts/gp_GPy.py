import numpy as np
import GPy

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


class GPyMeanModel:
    def __init__(
        self,
        kernel=None,
        mean_function="linear",
        optimizer="bfgs",
        max_iters=1000,
        ard=True,
        noise_var=1e-3,
        noise_bounds=(1e-8, 1.0),
    ):
        self.kernel = kernel
        self.mean_function = mean_function
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.ard = ard
        self.noise_var = noise_var
        self.noise_bounds = noise_bounds

        self.x_scaler = StandardScaler()
        self.model = None

        self.y_mean = None
        self.y_std = None
        self.input_dim = None

    def _prepare_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _prepare_y(self, y):
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

    def _build_mean_function(self):
        if self.mean_function is None:
            return None

        if self.mean_function == "linear":
            return GPy.mappings.Linear(
                input_dim=self.input_dim,
                output_dim=1
            )

        raise ValueError(f"Unsupported mean_function: {self.mean_function}")

    def _build_kernel(self):
        if self.kernel is not None:
            return self.kernel

        return GPy.kern.Matern52(
            input_dim=self.input_dim,
            ARD=self.ard
        )

    def fit(self, X_train, y_train, messages=True):
        X_train = self._prepare_X(X_train)
        y_train = self._prepare_y(y_train)

        self.input_dim = X_train.shape[1]

        # Scale X
        X_train_s = self.x_scaler.fit_transform(X_train)

        # Scale y
        self.y_mean = y_train.mean()
        self.y_std = y_train.std()
        if self.y_std == 0:
            self.y_std = 1.0

        y_train_s = (y_train - self.y_mean) / self.y_std

        mean_func = self._build_mean_function()
        kernel = self._build_kernel()
        likelihood = GPy.likelihoods.Gaussian()

        self.model = GPy.core.GP(
            X=X_train_s,
            Y=y_train_s,
            kernel=kernel,
            likelihood=likelihood,
            mean_function=mean_func
        )

        self.model.likelihood.variance = self.noise_var
        self.model.likelihood.variance.constrain_bounded(
            self.noise_bounds[0],
            self.noise_bounds[1]
        )

        self.model.optimize(
            optimizer=self.optimizer,
            max_iters=self.max_iters,
            messages=messages
        )

        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("You need to fit the model first.")

        X = self._prepare_X(X)
        X_s = self.x_scaler.transform(X)

        y_pred_s, y_var_s = self.model.predict(X_s)

        y_pred = y_pred_s * self.y_std + self.y_mean
        y_std_pred = np.sqrt(y_var_s) * self.y_std

        return y_pred.ravel(), y_std_pred.ravel()

    def evaluate(self, X, y_true):
        y_true = self._prepare_y(y_true)
        y_pred, _ = self.predict(X)

        y_true_1d = y_true.ravel()

        mse = mean_squared_error(y_true_1d, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_1d, y_pred)

        return {
            "r2": r2,
            "rmse": rmse,
            "mse": mse
        }

    def fit_and_evaluate(self, X_train, y_train, X_test, y_test, messages=True):
        self.fit(X_train, y_train, messages=messages)
        return self.evaluate(X_test, y_test)

    def predict_dataframe(self, X, y_true=None):
        y_pred, y_std = self.predict(X)

        data = {
            "y_pred": y_pred,
            "y_std": y_std
        }

        if y_true is not None:
            y_true = self._prepare_y(y_true).ravel()
            data["y_true"] = y_true

        import pandas as pd
        return pd.DataFrame(data)