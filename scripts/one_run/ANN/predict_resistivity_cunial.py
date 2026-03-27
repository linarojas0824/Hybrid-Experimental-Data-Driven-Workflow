# Paths
from src.utils.paths import MODELS_DIR, RAW_DIR, PREDICTIONS_DIR, MODELS_DIR

# Libraries
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import pandas as pd
import pickle
import numpy as np

PRETRAINED_PATH = MODELS_DIR/"ANN_customized.h5"  


#========= LOAD DATASETS ====================#
columns_names = ["r", "del_r", "del_EN", "S", "VEC"]
comp_colum = ["Cu","Ni","Al"]

X_exp_data = pd.read_pickle(RAW_DIR/"descriptors_exp.pkl")
df_comp = X_exp_data[comp_colum]

x_all = np.asarray(X_exp_data[columns_names])

#========= SCALE ====================#
scaler = MinMaxScaler()
X_all_scale = scaler.fit_transform(x_all)


#========= PREDICTION ====================#
model = tf.keras.models.load_model(str(PRETRAINED_PATH))

exp_data = model.predict(X_all_scale, verbose=0).reshape(-1)

#========= SAVE DOCUMENT ====================#

exp_resis_predic = pd.DataFrame(x_all, columns=columns_names)
exp_resis_predic[comp_colum] = df_comp.values
exp_resis_predic['resistivity'] = exp_data


with open(PREDICTIONS_DIR / "experimental_resis_prediction.pkl", "wb") as f:
    pickle.dump(exp_resis_predic, f)
