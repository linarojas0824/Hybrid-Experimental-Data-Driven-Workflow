from database_manager import DatabaseManager
import pandas as pd
import numpy as np
import sqlite3
import pickle

class EMA_Properties():
    
    def __init__(self):
        """Initialize the dataset connection"""
        db_pure = "/Users/linarojas/Desktop/Research/Papers/Combinatorial_Ternary/Ellipsometry/Python/Python_scripts/Database/Pure_elements.db"
        self.db_pure = db_pure
        self.conn = sqlite3.connect( self.db_pure)
        self.cursor = self.conn.cursor() 
        print (f"Connected to database: {self.db_pure}")
        
    
    def table_dataframe(self,table_name):
        ## Return the table as a dataframe
        df = pd.read_sql_query(f"SELECT * FROM {table_name};", self.conn)
        return df
    
    def EMA_dataframe(self,df_compositions,action='return'):
        
        optical = self.table_dataframe ('optical_properties')
        df_relax_time = self.table_dataframe ('compositions')
        
        ## Relaxation time from the pure components
        rt = df_relax_time.set_index('id')['Relax_time'].to_dict()
        rho = df_relax_time.set_index('id')['Resistivity'].to_dict()
        
        # Calculate the weighted relaxation time
        df_compositions['Relax_time_weighted'] = (
            df_compositions['Cu'] * rt[1] +
            df_compositions['Ni'] * rt[2] +
            df_compositions['Al'] * rt[3] )
        
        #Calculate the weighted resistivity
        df_compositions['Resistivity_weighted'] = (
            df_compositions['Cu'] * rho[1] +
            df_compositions['Ni'] * rho[2] +
            df_compositions['Al'] * rho[3]
        )
        
                ## Cartesian product (composition x wavelength)
        waves = optical['wavelength_nm'].unique()

        df = (
            df_compositions[['ID','Cu','Ni','Al','Relax_time_weighted','Resistivity_weighted']]
            .assign(key=1)
            .merge(pd.DataFrame({'wavelength_nm': waves, 'key': 1}), on='key')
            .drop(columns='key')
        )

        ## Extract pure optical data
        opt_Cu = optical[optical['composition_id']==1][['wavelength_nm','e1','e2']].rename(
            columns={'e1':'e1_Cu','e2':'e2_Cu'}
        )

        opt_Ni = optical[optical['composition_id']==2][['wavelength_nm','e1','e2']].rename(
            columns={'e1':'e1_Ni','e2':'e2_Ni'}
        )

        opt_Al = optical[optical['composition_id']==3][['wavelength_nm','e1','e2']].rename(
            columns={'e1':'e1_Al','e2':'e2_Al'}
        )

        df = df.merge(opt_Cu, on='wavelength_nm')
        df = df.merge(opt_Ni, on='wavelength_nm')
        df = df.merge(opt_Al, on='wavelength_nm')

        ## compute weighted e1 and e2
        df['e1'] = (
            df['Cu'] * df['e1_Cu'] +
            df['Ni'] * df['e1_Ni'] +
            df['Al'] * df['e1_Al']
        )

        df['e2'] = (
            df['Cu'] * df['e2_Cu'] +
            df['Ni'] * df['e2_Ni'] +
            df['Al'] * df['e2_Al']
        )

        optical_sythe = df[['ID', 'Cu','Ni','Al',
                            'wavelength_nm', 'e1','e2',
                            'Relax_time_weighted','Resistivity_weighted']]

        ## Rename the columns
        optical_sythe = optical_sythe.rename(columns={
            'Relax_time_weighted': 'Relax_time',
            'Resistivity_weighted': 'Resistivity'
        })
        if action=='return':
            return optical_sythe
        else:
            optical_sythe.to_pickle('EMA_properties.pkl')