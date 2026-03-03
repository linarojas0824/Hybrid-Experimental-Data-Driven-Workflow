
# Paths
from src.utils.paths import DB_FILE
from src.preprocessing.database_manager import DatabaseManager
from src.preprocessing.DescriptorEngineer import ElementPropertyLoader
from src.preprocessing.DescriptorEngineer import AlloyDescriptorCalculator
from src.utils.paths import MODELS_DIR
from src.utils.paths import PREDICTIONS_DIR
from src.utils.paths import DATA_DIR

#libreries
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


PRETRAINED_PATH = MODELS_DIR/"ANN_customized.h5"  
N_OUTPUTS = 1                              
BATCH_SIZE = 4
EPOCHS = 1000
LR = 1e-3 # Learning rate

MONITOR = "loss" #. for the LOO approach

# -----------------------------
# Load the data
# -----------------------------

db = DatabaseManager(DB_FILE)
df_optical = db.table_dataframe('Optical_properties')
df_compositions = db.table_dataframe('compositions')
db.close()

df_tern_1_rou = pd.merge(df_optical,df_compositions, on='ID')
tern_1_1550 = df_tern_1_rou[df_tern_1_rou["wavelength_nm"] == 1552]
df_comp = tern_1_1550[['Cu','Ni','Al']]

path_properties = DATA_DIR/'raw'
loader = ElementPropertyLoader(path_properties)
elem_props = loader.load()

calc = AlloyDescriptorCalculator(elem_props=elem_props)
df_ternary_1f_properties = calc.add_all_descriptors(df_comp)

columns_names = ["r", "del_r", "del_EN", "S", "VEC"]
df_ternary_1f_properties = df_ternary_1f_properties[columns_names]

X = df_ternary_1f_properties.to_numpy() # Input features
y = tern_1_1550['e2'].to_numpy() # output variable

# -----------------------------
# LOOCV LOOP
# -----------------------------
loo = LeaveOneOut()

y_true_all = []
y_pred_all = []

mse_all = []
mae_all = []

for fold, (train_idx, val_idx) in enumerate(loo.split(X), start=1):
    # SPLIT
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # NORMALIZATION
    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s   = x_scaler.transform(X_val)
    
    # TRANSFER MODEL
    base_model = tf.keras.models.load_model(PRETRAINED_PATH)
    x = base_model.layers[-2].output
    new_out = tf.keras.layers.Dense(N_OUTPUTS, name="new_head")(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=new_out)
    
    # FREEZE ALL BASE LAYERS
    for layer in base_model.layers:
        layer.trainable = False
    
    # CALLBACKS (Prevent overfitting)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=MONITOR,
            patience=40,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=MONITOR,
            factor=0.5,
            patience=15,
            min_lr=1e-6
        ),
    ]
    
    # TRAIN HEAD OF THE MODEL
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="mse",
        metrics=[tf.keras.metrics.MAE]
    )

    model.fit(
        X_train_s, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=callbacks
    )
    
    # PREDICT LEFT-OUT POINT
    pred = model.predict(X_val_s, verbose=0)
    
    y_true = y_val.reshape(-1)
    y_pred = pred.reshape(-1)
    
    y_true_all.append(y_true)
    y_pred_all.append(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    mse_all.append(mse)
    mae_all.append(mae)

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)

rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
mae_global = mean_absolute_error(y_true_all, y_pred_all)
# SAVE RESULTS

results = {
    "y_true": y_true_all,
    "y_pred": y_pred_all,
    "mse_per_fold": np.array(mse_all),
    "mae_per_fold": np.array(mae_all),
    "rmse_global": rmse,
    "mae_global": mae_global,
}

params = {
    "pretrained_model": PRETRAINED_PATH,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LR,
    "monitor": MONITOR,
    "n_samples": len(y_true_all),
}

np.savez(PREDICTIONS_DIR/
    "loocv_results_e2.npz",
    results=results,
    params=params
)