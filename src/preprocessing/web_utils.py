import requests
import pandas as pd
import numpy as np
from io import StringIO

# ====================== Valence Electron ================================
class ElementTableProp:
    def __init__(self, url, table_index=0, timeout=30):
        self.url = url
        self.table_index = table_index
        self.timeout = timeout

    def read_table(self):
        r = requests.get(self.url, headers={"User-Agent": "Mozilla/5.0"}, timeout=self.timeout)
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
        if self.table_index >= len(tables):
            raise IndexError(f"Requested table {self.table_index}, but only {len(tables)} tables found.")
        return tables[self.table_index].reset_index(drop=True)

    @staticmethod
    def _find_symbol_col(df):
        for c in df.columns:
            if str(c).strip().lower() == "symbol":
                return c
        raise KeyError(f"No Symbol/symbol column found. Columns: {list(df.columns)}")

    def extract_property(
        self,
        value_col,
        out_value_name=None,
        dropna=True,
        numeric=True,
        extract_digits=False,
    ):
        df = self.read_table()

        sym_col = self._find_symbol_col(df)

        out_value_name = out_value_name or value_col

        out = df[[sym_col, value_col]].copy()
        out = out.rename(columns={sym_col: "symbol", value_col: out_value_name})

        if extract_digits:
            out[out_value_name] = (
                out[out_value_name].astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
            )

        if numeric:
            out[out_value_name] = pd.to_numeric(out[out_value_name], errors="coerce")

        if dropna:
            out = out.dropna(subset=[out_value_name])

        out["symbol"] = out["symbol"].astype(str).str.strip()
        out = out.drop_duplicates(subset=["symbol"]).reset_index(drop=True)

        return out
    @staticmethod
    def _last_numeric(row):
        nums = pd.to_numeric(row, errors="coerce").dropna()
        return nums.iloc[-1] if len(nums) else np.nan

    def extract_valence_electrons(self, legend_col="Legend", name_row_start=1, row_step=3, valence_row_offset=3):
        df = self.read_table()

        if legend_col not in df.columns:
            raise KeyError(f"Column '{legend_col}' not found. Columns: {list(df.columns)}")

        df_name = df.iloc[name_row_start::row_step].copy()
        df_val = df.iloc[valence_row_offset::row_step].copy()

        df_name["symbol"] = df_name[legend_col].astype(str).str.extract(r"\b\d+\s+([A-Z][a-z]?)\b")
        df_name = df_name.reset_index(drop=True)

        df_val["valence"] = df_val.apply(self._last_numeric, axis=1)
        df_val = df_val.reset_index(drop=True)

        out = pd.DataFrame({"symbol": df_name["symbol"], "valence": df_val["valence"]})
        out = out.dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"]).reset_index(drop=True)
        return out
    
    @staticmethod
    def to_map(df, value_col):
        return dict(zip(df["symbol"], df[value_col]))

#==================== AFLOW Data Extraction ============================

def get_species_df(sp, page=1):
    url = f"http://aflow.org/API/aflux/?species({sp}),compound,agl_thermal_conductivity_300K,stoichiometry,paging({page}),format(json)"
    return pd.DataFrame(requests.get(url).json())

dfs = []
for sp in ["Cu","Ni","Al"]:
    df = get_species_df(sp, page=1)
    if not df.empty:
        df["species_query"] = sp
        dfs.append(df)

df_final = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
df_final.head(7)

def read_tab_website (url, no_tab):
    headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    html = StringIO(response.text)
    tables = pd.read_html(html)

    df = tables[no_tab]
    return df

