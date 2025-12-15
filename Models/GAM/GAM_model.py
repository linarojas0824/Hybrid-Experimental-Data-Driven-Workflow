## Libraries
import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from sklearn.model_selection import train_test_split
import pickle

## Load the data
df_merged = pd.read_pickle("GAM_data.pkl")
## Split data for training and testing

X = df_merged[['Cu','Ni','Al','wavelength_nm','e1','e2','Resistivity']]
y = df_merged[['Relax_time']]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)

## Train the model
gam = LinearGAM(s(0)+s(1)+s(2)+s(3)+s(4)+s(5)+s(6))
## Fit the model to the data
gam.fit(X_train,y_train)
## Hyperparameter tuning
gam.gridsearch(X_train, y_train)

## Save the model 
with open("gam_model.pkl","wb") as f:
    pickle.dump(gam,f)