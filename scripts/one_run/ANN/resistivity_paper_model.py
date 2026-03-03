import os
import json
import time

import pandas as pd
import numpy as np

import tensorflow as tf
from model.ANN import Net
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import warnings
#from silence_tensorflow import silence_tensorflow
#silence_tensorflow()
warnings.filterwarnings("ignore")

# -------------- DATA --------------------------- #
# load experimental dataset used for the model training
train = pd.read_csv('Train.csv')
val = pd.read_csv('Val.csv')
total = pd.concat([train, val], axis=0).reset_index(drop=True)

norm_list = ['r', 'del_r', 'del_EN', 'S','VEC'] # Set compositional input features thin-film alloys you have
normalize = total[norm_list]

# Normalization
scaler = MinMaxScaler()
normalize[:] = scaler.fit_transform(normalize[:])

X_train = normalize.iloc[0:len(train),:]
X_val = normalize.iloc[len(train):,:]

y_train = train['resistivity']
y_val = val['resistivity']

#--------------- PATH ------------------------------------ #

RUN_NAME = time.strftime("run_%Y%m%d_%H%M%S")
RUN_DIR = os.path.join("runs", RUN_NAME)
os.makedirs(RUN_DIR, exist_ok=True)

# -------------- Save configuration --------------------- #

config = dict(
    input_dim=int(X_train.shape[1]),
    num_dense_layers=3,
    num_dense_nodes=512,
    lr=5e-4,
    epochs=1000,
)
with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)


# -------------- BUILD MODEL ------------------------------ #

model = Net(input_dim=len(X_train.T), num_dense_layers=3, num_dense_nodes=512)

# -------------- CHECKPOINT -CALLBACKS --------------------- #

BEST_WEIGHTS_PATH = os.path.join(RUN_DIR, "best.weights.h5")
LAST_WEIGHTS_PATH = os.path.join(RUN_DIR, "last.weights.h5")
HISTORY_CSV_PATH  = os.path.join(RUN_DIR, "history.csv")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_WEIGHTS_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,   # safest for custom/subclassed Net
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(HISTORY_CSV_PATH, append=False),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min",
        patience=50, restore_best_weights=True
    ),
]

# -------------- TRAIN ------------------------------ #

batch_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(512)
batch_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(512)



history = model.train(num_epochs=1000, 
                      batch_train=batch_train, 
                      batch_val=batch_val, 
                      optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4))

# Save last weights (end of training)
model.model.save_weights(LAST_WEIGHTS_PATH)

# Save the history
if history is not None and hasattr(history, "history"):
    with open(os.path.join(RUN_DIR, "history.json"), "w") as f:
        json.dump(history.history, f, indent=2)
        
# Save the model
try:
    model.model.save(os.path.join(RUN_DIR, "model.keras"))
except Exception as e:
    with open(os.path.join(RUN_DIR, "model_save_error.txt"), "w") as f:
        f.write(str(e) + "\n")
        
# -------------- TEST ------------------------------ #
train_pred = model(tf.convert_to_tensor(X_train, dtype=tf.float32)).numpy()
val_pred   = model(tf.convert_to_tensor(X_val, dtype=tf.float32)).numpy()

train_r2 = r2_score(y_train, train_pred)
val_r2   = r2_score(y_val, val_pred)

# Save results
metrics = {
    "train_r2": float(train_r2),
    "val_r2": float(val_r2),
}

with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# Optional: save predictions for parity plots later
np.save(os.path.join(RUN_DIR, "train_pred.npy"), train_pred)
np.save(os.path.join(RUN_DIR, "val_pred.npy"), val_pred)