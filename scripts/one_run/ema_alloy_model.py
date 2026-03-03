from database_manager import DatabaseManager
import pandas as pd
import numpy as np
import sqlite3

## Connecting to the database and creating the dataframes
db = DatabaseManager("combinatorial_data.db")
optical = db.table_dataframe ('optical_properties')
db.close()

db = DatabaseManager("Synthetic_data.db")
compositions = db.table_dataframe ('compositions')
db.close()

waves = optical['wavelength_nm'].unique()

## Creating the loop to calculate
cu = []
ni =[]
al = []
wavelength_list = []
e1 = []
e2 = []
composition_id = []

for comp_index in range(len(compositions)):
    comp_row = compositions.iloc[comp_index]
    for wave in waves:
        cu.append(comp_row['Cu'])
        ni.append(comp_row['Ni'])
        al.append(comp_row['Al'])
        composition_id.append(comp_row['composition_id'])
        wavelength_list.append(wave)
        
        ## Calculate weighted e1 and e2
        e1_val = (comp_row['Cu']*optical[(optical['composition_id']==1)&(optical['wavelength_nm']==wave)]['e1'].values[0]+
            comp_row['Ni']*optical[(optical['composition_id']==2)&(optical['wavelength_nm']==wave)]['e1'].values[0] +
            comp_row['Al']*optical[(optical['composition_id']==3)&(optical['wavelength_nm']==wave)]['e1'].values[0])
        
        e2_val = (comp_row['Cu']*optical[(optical['composition_id']==1)&(optical['wavelength_nm']==wave)]['e2'].values[0]+
            comp_row['Ni']*optical[(optical['composition_id']==2)&(optical['wavelength_nm']==wave)]['e2'].values[0] +
            comp_row['Al']*optical[(optical['composition_id']==3)&(optical['wavelength_nm']==wave)]['e2'].values[0]) 
        e1.append(e1_val)
        e2.append(e2_val)

## Creating the final dataframe
optical_sythe = pd.DataFrame({
    'comp_id':composition_id,
    'Cu': cu,
    'Ni': ni,
    'Al': al,
    'wavelength_nm': wavelength_list,
    'e1':e1,
    'e2':e2
})

## Save the dataframe
optical_sythe.to_pickle('synthetic_data.pkl')