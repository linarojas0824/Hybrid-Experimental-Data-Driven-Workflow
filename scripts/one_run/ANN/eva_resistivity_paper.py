
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import json

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# -------------- DATA --------------------------- #
# load experimental dataset used for the model training
train = pd.read_csv("Train.csv")
val   = pd.read_csv("Val.csv")

norm_list = ["r", "del_r", "del_EN", "S", "VEC"]

Xtr_raw = train[norm_list].to_numpy(dtype=float)
Xva_raw = val[norm_list].to_numpy(dtype=float)

# IMPORTANT: fit scaler on TRAIN only (match typical training workflow)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(Xtr_raw)
X_val   = scaler.transform(Xva_raw)

y_train = train["resistivity"].to_numpy(dtype=float)
y_val   = val["resistivity"].to_numpy(dtype=float)


RUN_DIR = "model"
model = tf.keras.models.load_model(os.path.join(RUN_DIR, "ANN_customized.h5"))

train_pred = model(tf.convert_to_tensor(X_train, dtype=tf.float32)).numpy()
val_pred   = model(tf.convert_to_tensor(X_val, dtype=tf.float32)).numpy()

# Predict
train_pred = model.predict(X_train, verbose=0).reshape(-1)
val_pred   = model.predict(X_val,   verbose=0).reshape(-1)

train_r2 = r2_score(y_train, train_pred)
val_r2   = r2_score(y_val, val_pred)

# Save results
metrics = {"train_r2": float(train_r2), "val_r2": float(val_r2)}
with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

np.save(os.path.join(RUN_DIR, "train_pred.npy"), train_pred)
np.save(os.path.join(RUN_DIR, "val_pred.npy"), val_pred)