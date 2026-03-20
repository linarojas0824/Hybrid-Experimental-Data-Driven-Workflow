from typing import Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

            return cluster_split

        else:
            raise ValueError("split_method must be 'random' or 'cluster'")

class DataPreprocessor:
    @staticmethod
    def train_test(
        split_dict,
        calculator,
        feature_cols,
        scaler: Optional[object] = None):
          
        # Transform data
        df_train = calculator.transform(split_dict['X_train_split'])
        df_test = calculator.transform(split_dict['X_test_split'])

        # Extract features
        X_train = df_train[feature_cols].to_numpy()
        X_test = df_test[feature_cols].to_numpy()
    
        # Extract targets
        y_train = np.asarray(split_dict['y_train']).ravel()
        y_test = np.asarray(split_dict['y_test']).ravel()

        # Apply scaling if provided
        if scaler is not None:
            x_train = scaler.fit_transform(X_train)
            x_test = scaler.transform(X_test)
        else:
            x_train = X_train
            x_test = X_test

        return x_train, x_test, y_train, y_test,scaler
    
    @staticmethod
    def gp_predict_and_errors(
        calculator,
        df_inse,
        gp_model,
        scaler=None,
        X_test=None,
        y_test=None,
        feature_cols = ['r','del_r','del_EN','S','VEC'],
        pred_col="e2",
        std_col="std"
    ):


        comp_space = calculator.transform(df_inse)
        df_pred = comp_space.copy()

        X_space = comp_space[feature_cols].to_numpy() if isinstance(comp_space, pd.DataFrame) else np.asarray(comp_space)

        if scaler is not None:
            X_space_proc = scaler.transform(X_space)
        else:
            X_space_proc = X_space

        y_space_pred, y_space_std = gp_model.predict(X_space_proc, return_std=True)

        df_pred[pred_col] = y_space_pred
        df_pred[std_col] = y_space_std

        # ---- Initialize error outputs ----
        error_dict = {
            "mae": None,
            "rmse": None,
            "r2": None,
            "df_residuals": pd.DataFrame()
        }

        # ---- Evaluate on test set if provided ----
        if X_test is not None and y_test is not None:
            X_test_arr = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else np.asarray(X_test)
            y_test_arr = np.asarray(y_test).ravel()

            if scaler is not None:
                X_test_proc = scaler.transform(X_test_arr)
            else:
                X_test_proc = X_test_arr

            y_test_pred = gp_model.predict(X_test_proc)

            mae = mean_absolute_error(y_test_arr, y_test_pred)
            rmse = np.sqrt(mean_squared_error(y_test_arr, y_test_pred))
            r2 = r2_score(y_test_arr, y_test_pred)

            residuals = y_test_arr - y_test_pred

            df_residuals = pd.DataFrame({
            "y_true": y_test_arr,
            "y_pred": y_test_pred,
            "error": residuals,
            "error_abs": np.abs(residuals),
            "error_sq": residuals**2
            })
            
            error_dict = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "df_residuals": df_residuals
            }

        return df_pred, error_dict
    
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
