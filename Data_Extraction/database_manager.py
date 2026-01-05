import pandas as pd
import numpy as np
import sqlite3

class DatabaseManager():
    def __init__(self, db_name):
        """Initialize the database connection."""
        self.db_name = db_name
        self.conn = sqlite3.connect( self.db_name)
        self.cursor = self.conn.cursor()
        print (f"Connected to database: {self.db_name}")
    
    def list_tables(self):
        """Return a list of all tables in database"""
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in self.cursor.fetchall()]
        return tables
    
    def create_table(self, table_name, columns):
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});"
        self.cursor.execute(sql)
        self.conn.commit()
        print(f"Table '{table_name}' created successfully.")
    
    def create_table_df(self, df, table_name):
        df.to_sql(table_name, self.conn, if_exists='replace',index=False)
        print(f"Table '{table_name}' created successfully.")
    
    def table_dataframe(self,table_name):
        ## Return the table as a dataframe
        df = pd.read_sql_query(f"SELECT * FROM {table_name};", self.conn)
        return df
    
    def col_names(self,table_name):
        "Return the name of the columns in a database"
        self.cursor.execute(f'PRAGMA table_info({table_name})')
        columns = [col[1] for col in self.cursor.fetchall()]
        print("Colum names:",columns)
    
    def view_data(self,table_name,column_name,order="ASC",limit=10):
        order = order.upper()
        if order not in ('ASC','DESC'):
            raise ValueError ("Order must be 'ASC or 'DESC'")
        query = f"""SELECT * FROM {table_name} 
               ORDER BY {column_name} {order}
               LIMIT ?;"""
        self.cursor.execute(query,(limit,))
        rows = self.cursor.fetchall()
        for row in rows:
            print(row)
    
    def extract_pure_values(self,table_name, column_name, compo):
        query = f"SELECT {column_name} FROM {table_name} WHERE {compo} =1"
        df_resistiviy = pd.read_sql_query(query,self.conn)
        
        if not df_resistiviy.empty:
            value = df_resistiviy.iloc[0][column_name]
            return value
        else:
            return None
    
    def close(self):
        """Close the database connection"""
        self.conn.close()
        print("Database connection closed")
    
