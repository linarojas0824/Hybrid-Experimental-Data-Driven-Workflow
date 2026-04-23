# Paths
from src.utils.paths import MODELS_DIR, RAW_DIR, PREDICTIONS_DIR, MODELS_DIR

# Libraries
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle
import numpy as np

PRETRAINED_PATH = MODELS_DIR/"ANN_customized.h5"  

#========= LOAD SCALER ====================#
with open(MODELS_DIR / "scaler_ANN_customized.pkl", "rb") as f:
    scaler = pickle.load(f)  

#========= LOAD DATASETS ====================#


columns_names = ["r", "del_r", "del_EN", "S", "VEC"]
comp_colum = ["Cu","Ni","Al"]

exp_space = pd.read_csv(RAW_DIR/"exp_CuNiAl_descriptors.csv")

x_all = exp_space[columns_names]

#========= SCALE ====================#
X_all_scale = scaler.transform(x_all)

#========= PREDICTION ====================#
model = tf.keras.models.load_model(str(PRETRAINED_PATH))

exp_data = model.predict(X_all_scale, verbose=0).reshape(-1)

#========= SAVE DOCUMENT ====================#

# Add prediction directly to original dataframe
exp_space['resistivity'] = exp_data

# Save as CSV
exp_space.to_csv(PREDICTIONS_DIR / "exp_CuNiAl_with_resistivity.csv", index=False)
