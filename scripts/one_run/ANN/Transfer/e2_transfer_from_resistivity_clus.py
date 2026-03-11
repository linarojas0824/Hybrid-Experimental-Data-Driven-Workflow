
# Paths
from src.utils.paths import MODELS_DIR, SAMPLING_DIR, RAW_DIR, PREDICTIONS_DIR

# Libraries
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import numpy as np

PRETRAINED_PATH = MODELS_DIR/"ANN_customized.h5"  

#========= LOAD DATA RANDOM SPLIT====================#

comp_space = pd.read_csv(RAW_DIR/"CuNiAl_descriptors.csv")

with open(SAMPLING_DIR/"cluster_split", "rb") as f:
    cluster_split = pickle.load(f)

X_train_random = cluster_split['X_train_split']
X_test_random = cluster_split['X_test_split']

columns_names = ["r", "del_r", "del_EN", "S", "VEC"]
X_train_random = X_train_random[columns_names]
X_test_random = X_test_random[columns_names]



X = np.asarray(X_train_random, dtype=float)
X_test = np.asarray(X_test_random, dtype=float)
X_all =np.asarray(comp_space[columns_names], dtype=float)

y = np.asarray(cluster_split['y_train'], dtype=float)
y_test = np.asarray(cluster_split['y_test'], dtype=float)

#========= SCALE ====================#
x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)
X_scaled_test = x_scaler.transform(X_test)
X_scaled_all = x_scaler.transform(X_all)

#========== LOAD PRETRAINED MODEL =========#
base_model = tf.keras.models.load_model(PRETRAINED_PATH)


#============== FREEZE BASE LAYERS =====================#
for layer in base_model.layers:
    layer.trainable = False

#============== NEW OUTPUT HEAD =====================#
x = base_model.layers[-2].output
new_out = tf.keras.layers.Dense(1, name="new_head")(x)
model = tf.keras.Model(inputs=base_model.input, outputs=new_out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse"
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=40,
        restore_best_weights=True
    )
]

model.fit(
    X_scaled, y,
    epochs=1000,
    batch_size=4,
    callbacks=callbacks,
    verbose=1
)
#============== SAVE MODELS =====================#
model.save(MODELS_DIR/"e2_TL_rand_cluster.keras", include_optimizer=False)

#============== PREDICTIONS  =====================#

ann_cluster_split = {}
ann_cluster_split['x_space'] = X_all
test_pred = model(tf.convert_to_tensor(X_scaled_test, dtype=tf.float32)).numpy()
all_pred = model(tf.convert_to_tensor(X_scaled_all, dtype=tf.float32)).numpy()

ann_cluster_split['y_test_pred'] = test_pred
ann_cluster_split['all_space_pred'] = all_pred

with open(PREDICTIONS_DIR / "ann_cluster_split.pkl", "wb") as f:
    pickle.dump(ann_cluster_split, f)



