# Paths
from src.utils.paths import MODELS_DIR, SAMPLING_DIR, PREDICTIONS_DIR, MODELS_DIR

# Libraries
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import pandas as pd
import pickle
import numpy as np

PRETRAINED_PATH = MODELS_DIR/"ANN_customized.h5"  


#========= LOAD DATASETS ====================#

with open(SAMPLING_DIR/"random_split", "rb") as f:
    random_split = pickle.load(f)

X_train_random = random_split['X_train_split'].copy()
X_test_random = random_split['X_test_split'].copy()

columns_names = ["r", "del_r", "del_EN", "S", "VEC"]
comp_colum = ["Cu","Ni","Al"]

df_comp = pd.concat(
    [X_train_random[comp_colum], X_test_random[comp_colum]],
    axis=0,
    ignore_index=True
)

X_train_random = X_train_random[columns_names]
X_test_random = X_test_random[columns_names]

X = np.asarray(X_train_random, dtype=float)
X_test = np.asarray(X_test_random, dtype=float)
x_all = np.concatenate((X, X_test), axis=0)

#========= SCALE ====================#
scaler = MinMaxScaler()
X_all_scale = scaler.fit_transform(x_all)


#========= PREDICTION ====================#
PRETRAINED_PATH = MODELS_DIR / "ANN_customized.h5"
model = tf.keras.models.load_model(str(PRETRAINED_PATH))

exp_data = model(tf.convert_to_tensor(X_all_scale, dtype=tf.float32)).numpy()
exp_data = model.predict(X_all_scale, verbose=0).reshape(-1)

#========= SAVE DOCUMENT ====================#

exp_resis_predic = pd.DataFrame()
exp_resis_predic[comp_colum] = df_comp
exp_resis_predic[columns_names] = x_all
exp_resis_predic['resistivity'] = exp_data


with open(PREDICTIONS_DIR / "experimental_resis_prediction.pkl", "wb") as f:
    pickle.dump(exp_resis_predic, f)
