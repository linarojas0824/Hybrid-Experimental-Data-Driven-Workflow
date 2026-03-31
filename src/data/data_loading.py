import pandas as pd
import numpy as np


# ========= DATABASE LOADING =========
def load_table(db_path, table_name):
    from data.database_manager import DatabaseManager

    db = DatabaseManager(db_path)
    df = db.table_dataframe(table_name)
    db.close()

    return df

# ========= FILE LOADING =========
def load_csv(path):
    return pd.read_csv(path)


def load_pickle(path):
    return pd.read_pickle(path)