from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

class DataPreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = None
        
    def split_training(self,X,y):
        X_train, X_test, y_train,y_test = train_test_split(X,y,
                                                           test_size=self.test_size,
                                                           random_state=self.random_state,
                                                           shuffle=True)
        return (X_train, X_test, y_train,y_test)

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
