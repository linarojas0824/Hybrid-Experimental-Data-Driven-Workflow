from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


@dataclass
class DataSplitter:
    test_size: float = 0.15
    random_state: int = 42
    split_method: str = "random"   # "random" or "cluster"
    n_clusters: int = 4
    cluster_feature_start: int = 3
    force_pure_train: bool = False

    def split(self, X, y):
        if self.force_pure_train:
            return self._split_with_pure_train(X, y)

        if self.split_method == "random":
            return self._random_split(X, y)
        elif self.split_method == "cluster":
            return self._cluster_split(X, y)
        else:
            raise ValueError("split_method must be 'random' or 'cluster'")

    def _split_with_pure_train(self, X, y):
        pure_mask = self._get_pure_mask(X)

        if hasattr(X, "iloc"):
            X_pure = X.loc[pure_mask].copy()
            X_main = X.loc[~pure_mask].copy()
        else:
            X_pure = X[pure_mask]
            X_main = X[~pure_mask]

        if hasattr(y, "iloc"):
            y_pure = y.loc[pure_mask].copy()
            y_main = y.loc[~pure_mask].copy()
        else:
            y_pure = y[pure_mask]
            y_main = y[~pure_mask]

        if self.split_method == "random":
            split_dict = self._random_split(X_main, y_main)
        elif self.split_method == "cluster":
            split_dict = self._cluster_split(X_main, y_main)
        else:
            raise ValueError("split_method must be 'random' or 'cluster'")

        if hasattr(split_dict["X_train"], "iloc"):
            split_dict["X_train"] = pd.concat([split_dict["X_train"], X_pure], ignore_index=True)
            split_dict["y_train"] = pd.concat([split_dict["y_train"], y_pure], ignore_index=True)
        else:
            split_dict["X_train"] = np.concatenate([split_dict["X_train"], X_pure], axis=0)
            split_dict["y_train"] = np.concatenate([split_dict["y_train"], y_pure], axis=0)

        return split_dict
    

    
    def _get_pure_mask(self, X):
        if not hasattr(X, "columns"):
            raise ValueError("force_pure_train=True requires X to be a pandas DataFrame")

        required_cols = ["Cu", "Ni", "Al"]
        for col in required_cols:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' is required to detect pure compositions")

        comp = X[required_cols].astype(float).copy()
        tol = 1e-8

        total = comp.sum(axis=1)

        pure_mask = (
            (np.isclose(comp["Cu"], total, atol=tol) & np.isclose(comp["Ni"], 0, atol=tol) & np.isclose(comp["Al"], 0, atol=tol)) |
            (np.isclose(comp["Ni"], total, atol=tol) & np.isclose(comp["Cu"], 0, atol=tol) & np.isclose(comp["Al"], 0, atol=tol)) |
            (np.isclose(comp["Al"], total, atol=tol) & np.isclose(comp["Cu"], 0, atol=tol) & np.isclose(comp["Ni"], 0, atol=tol))
        )

        return pure_mask

    def _random_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def _cluster_split(self, X, y):
        X_numpy = np.asarray(X, dtype=float)
        X_cluster = X_numpy[:, self.cluster_feature_start:]

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        clusters = kmeans.fit_predict(X_cluster)

        train_idx = []
        test_idx = []

        rng = np.random.RandomState(self.random_state)

        for c in np.unique(clusters):
            cluster_indices = np.where(clusters == c)[0]

            if len(cluster_indices) == 1:
                train_idx.extend(cluster_indices)
                continue

            test_point = rng.choice(cluster_indices, size=1, replace=False)
            train_points = np.setdiff1d(cluster_indices, test_point)

            test_idx.extend(test_point)
            train_idx.extend(train_points)

        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)

        if hasattr(X, "iloc"):
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
        else:
            X_train = X[train_idx]
            X_test = X[test_idx]

        if hasattr(y, "iloc"):
            y_train = y.iloc[train_idx].copy()
            y_test = y.iloc[test_idx].copy()
        else:
            y_train = y[train_idx]
            y_test = y[test_idx]

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }