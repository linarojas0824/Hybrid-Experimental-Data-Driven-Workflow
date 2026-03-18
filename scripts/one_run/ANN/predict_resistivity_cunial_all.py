# Paths
from src.utils.paths import MODELS_DIR, PREDICTIONS_DIR, RAW_DIR

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

comp_space = pd.read_csv(RAW_DIR/"CuNiAl_descriptors.csv")

x_all = comp_space[columns_names]

#========= SCALE ====================#
scaler = MinMaxScaler()
X_all_scale = scaler.fit_transform(x_all)


#========= PREDICTION ====================#
PRETRAINED_PATH = MODELS_DIR / "ANN_customized.h5"
model = tf.keras.models.load_model(str(PRETRAINED_PATH))

exp_data = model.predict(X_all_scale, verbose=0).ravel()

#========= SAVE DOCUMENT ====================#

exp_resis_predic = pd.DataFrame()
exp_resis_predic[comp_colum] = comp_space[comp_colum]
exp_resis_predic[columns_names] = x_all
exp_resis_predic['resistivity'] = exp_data


with open(PREDICTIONS_DIR / "comp_space_resis_prediction.pkl", "wb") as f:
    pickle.dump(exp_resis_predic, f)
