
# Paths
from src.utils.paths import DB_FILE
from src.preprocessing.database_manager import DatabaseManager
from src.preprocessing.DescriptorEngineer import ElementPropertyLoader
from src.preprocessing.DescriptorEngineer import AlloyDescriptorCalculator
from src.utils.paths import MODELS_DIR
from src.utils.paths import DATA_DIR

# Libraries
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd

PRETRAINED_PATH = MODELS_DIR/"ANN_customized.h5"  

#========= LOAD DATA ====================#

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

#========= SCALE ====================#
x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)

#========== LOAD PRETRAINED MODEL =========#
base_model = tf.keras.models.load_model(PRETRAINED_PATH)

#============== OUTPUT =====================#
x = base_model.layers[-2].output
new_out = tf.keras.layers.Dense(1, name="new_head")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=new_out)

#============== FREEZE BASE LAYERS =====================#
for layer in base_model.layers:
    layer.trainable = False

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
#============== SAVE MODEL =====================#
model.save(MODELS_DIR/"TL_Small_Similar_e2.keras", include_optimizer=False)

#============== PREDICT SPACE  =====================#
