import os
import numpy as np
import tensorflow as tf
import pandas as pd
import json

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# -------------- DATA --------------------------- #
# load experimental dataset used for the model training
train = pd.read_csv("cualni_compositions.csv")


norm_list = ["r", "del_r", "del_EN", "S", "VEC"]

Xtr_raw = train[norm_list].to_numpy(dtype=float)


# IMPORTANT: fit scaler on TRAIN only (match typical training workflow)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(Xtr_raw)


RUN_DIR = "model"
model = tf.keras.models.load_model(os.path.join(RUN_DIR, "ANN_customized.h5"))

train_pred = model(tf.convert_to_tensor(X_train, dtype=tf.float32)).numpy()


# Predict
train_pred = model.predict(X_train, verbose=0).reshape(-1)


np.save(os.path.join(RUN_DIR, "train_pred.npy"), train_pred)
