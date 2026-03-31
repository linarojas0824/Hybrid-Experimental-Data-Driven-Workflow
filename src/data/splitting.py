from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


@dataclass
class DataSplitter:
    test_size: float = 0.15
    random_state: int = 42
    split_method: str = "random"   # "random" or "cluster"
    n_clusters: int = 4
    cluster_feature_start: int = 3

    def split(self, X, y):
        if self.split_method == "random":
            return self._random_split(X, y)
        elif self.split_method == "cluster":
            return self._cluster_split(X, y)
        else:
            raise ValueError("split_method must be 'random' or 'cluster'")

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