# Paths
from src.utils.paths import MODELS_DIR, PREDICTIONS_DIR, RAW_DIR

# Libraries
import tensorflow as tf
from sklearn.metrics import r2_score
import pandas as pd
import pickle
import numpy as np


#========= LOAD SCALER ====================#
with open(MODELS_DIR / "scaler_ANN_customized.pkl", "rb") as f:
    scaler = pickle.load(f)  

#========= LOAD DATASETS ====================#

columns_names = ["r", "del_r", "del_EN", "S", "VEC"]
comp_colum = ["Cu","Ni","Al"]

comp_space = pd.read_csv(RAW_DIR/"CuNiAl_descriptors.csv")

x_all = comp_space[columns_names]

#========= SCALE ====================#
X_all_scale = scaler.transform(x_all)


#========= PREDICTION ====================#
PRETRAINED_PATH = MODELS_DIR / "ANN_customized.h5"
model = tf.keras.models.load_model(str(PRETRAINED_PATH))

exp_data = model.predict(X_all_scale, verbose=0).ravel()

#========= SAVE DOCUMENT ====================#

# Add prediction directly to original dataframe
comp_space['resistivity'] = exp_data

# Save as CSV
comp_space.to_csv(PREDICTIONS_DIR / "CuNiAl_with_resistivity.csv", index=False)

