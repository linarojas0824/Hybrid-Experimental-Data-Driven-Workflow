from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataclasses import dataclass
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

@dataclass

@dataclass
class DataSplitter:
    test_size: float = 0.15
    random_state: int = 42
    split_method: str = "random"   # "random" or "cluster"
    n_clusters: int = 4
    cluster_feature_start: int = 3  # use X[:, 3:] for clustering

    def split_training(self, X, y):

        if self.split_method == "random":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=True
            )

            split_dict = {}
            split_dict["X_train_split"] = X_train
            split_dict["X_test_split"] = X_test
            split_dict["y_train"] = y_train
            split_dict["y_test"] = y_test

            return split_dict

        elif self.split_method == "cluster":

            X_numpy = np.asarray(X, dtype=float)
            X_n2 = X_numpy[:, self.cluster_feature_start:]

            kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            clusters = kmeans.fit_predict(X_n2)

            train_idx = []
            test_idx = []

            rng = np.random.RandomState(self.random_state)

            for c in np.unique(clusters):
                cluster_indices = np.where(clusters == c)[0]

                # If cluster has only 1 point → keep in train
                if len(cluster_indices) == 1:
                    train_idx.extend(cluster_indices)
                    continue

                # Select ONE test point per cluster
                test_point = rng.choice(cluster_indices, size=1, replace=False)

                # Remaining points go to train
                train_points = np.setdiff1d(cluster_indices, test_point)

                test_idx.extend(test_point)
                train_idx.extend(train_points)

            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)

            # ===== SPLIT DATA =====
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()

            y_train = y.iloc[train_idx].copy()
            y_test = y.iloc[test_idx].copy()

            # ===== OUTPUT =====
            cluster_split = {}
            cluster_split["X_train_split"] = X_train
            cluster_split["X_test_split"] = X_test
            cluster_split["y_train"] = y_train
            cluster_split["y_test"] = y_test

            # Extra useful info
            cluster_split["train_idx"] = train_idx
            cluster_split["test_idx"] = test_idx
            cluster_split["clusters"] = clusters

            return cluster_split

        else:
            raise ValueError("split_method must be 'random' or 'cluster'")

class DataPreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = None

    def fit_transform(self,X_train):
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        return X_train_scaled
    
    #---------------- Multiply the DR electronic data by the composition ----------------------- #
    @staticmethod
    def composition_DR_ele_P(elect_data_DR, X_data): 
        df_red = elect_data_DR.set_index("element") 
        dims = df_red.columns.tolist()
    
        elemnt = [el for el in X_data.columns if el in df_red.index]
    
        X = X_data[elemnt].to_numpy(dtype=float)
        Z = df_red.loc[elemnt, dims].to_numpy(dtype=float)
    
        W = X[:, :, None] * Z[None, :, :]
        W2 = W.reshape(X.shape[0], -1) 
    
        colnames = [f"{el}_{d}" for el in elemnt for d in dims] 
        df_final = pd.DataFrame(W2, index=X_data.index, columns=colnames)
        
        return df_final
    
     #---------------- Create expanded composition Matrix ----------------------- #
    @staticmethod
    def expan_comp_df(df, df_database, prope_columns=None):

        df = df.copy()
        df_database = df_database.copy()

        elem_list = list(df_database.columns)

        df_comp = df.reindex(columns=elem_list, fill_value=0)

        if prope_columns is not None:
    
            if isinstance(prope_columns, str):
                prope_columns = [prope_columns]

            missing = set(prope_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Missing columns in df: {missing}")

    
            df_expand = pd.concat(
                [df[prope_columns].reset_index(drop=True),
                df_comp.reset_index(drop=True)],
                axis=1
            )
        else:
            df_expand = df_comp.reset_index(drop=True)

        return df_expand
