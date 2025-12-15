import requests
import pandas as pd
import numpy as np
import json
from urllib.request import urlopen
import pickle

def request_function(property,specie,page):
    url = (
    "http://aflowlib.duke.edu/search/API/?"
    f"{property}(!null),"
    "compound,"
    "stoichiometry,"
    f"$paging({page}),"
    f"species({specie})")
    
    r = requests.get(url)
    try:
     data = r.json()
    except:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    return df

species = ['Cu', 'Ni','Al']
all_data = {}

for specie in species:
    data_specie = pd.DataFrame(columns=['compound','agl_thermal_conductivity_300K','stoichiometry'])

    for i in range(90):
        df = request_function("agl_thermal_conductivity_300K",specie,i)
        
        if df.empty:
            continue
        
        df_v2 = df[["compound","agl_thermal_conductivity_300K","stoichiometry"]]
        data_specie = pd.concat([data_specie,df], ignore_index=True)
    
    all_data[specie] = data_specie

## Save the dictory as a pickle document
with open("data_AFLOW.pkl","wb") as f:
    pickle.dump(all_data,f)