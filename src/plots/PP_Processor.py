from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union,Sequence

import numpy as np
import pandas as pd


from preprocessing.SignalNoise_Process import SignalProcessor



# -------- Data loading + cleaning ------------------------- #

@dataclass
class DataCollector:
    index_col: str = "0.00000E+0"
    sp= SignalProcessor()
    
    def extract_file_names(self, folder, pattern, select_power="max"):

        rows = []
        folder = Path(folder)

        for fp in folder.glob("*.csv"):
            m = pattern.search(fp.name)
            if not m:
                continue

            rows.append({
                "file": fp.name,
                "sample": int(m.group("sample")),
                "power_mW": int(m.group("power")),
                "path": str(fp),
            })

        files_index = pd.DataFrame(rows).sort_values(["sample", "power_mW"])

        # -------- selection logic --------

        if select_power == "max":
            idx = files_index.groupby("sample")["power_mW"].idxmax()
            selected = files_index.loc[idx]

        elif select_power == "min":
            idx = files_index.groupby("sample")["power_mW"].idxmin()
            selected = files_index.loc[idx]

        elif isinstance(select_power, (int, float)):
            selected = files_index[files_index["power_mW"] == select_power]

        else:
            raise ValueError("select_power must be 'max', 'min', or a numeric value")

        selected = selected.sort_values("sample")

        return files_index, selected
    
    def load_csv(self,
        path: str,
        round_cols: Optional[int] = 1,
        round_index: Optional[int] = 1,
        header: Union[int, None] = 0,
        low:Optional[float] = 0,
        high:Optional[float] = 39
    ) -> pd.DataFrame:
        df = pd.read_csv(path, header=header)

        # convert everything to numeric when possible
        df = df.apply(pd.to_numeric, errors="coerce")
        
        if self.index_col not in df.columns:
            raise KeyError(
                f"Index column '{self.index_col}' not found. Available: {list(df.columns)[:10]} ..."
        )
            
        df = df.set_index(self.index_col)
        df.index = pd.to_numeric(df.index, errors="coerce").round(round_index)
        
        colnums = pd.to_numeric(df.columns, errors="coerce").round(round_cols)
        mask = (colnums >= low) & (colnums <= high)
        df = df.loc[:, mask]
        
        df.columns = colnums[mask]
        
        df = df.loc[:, ~df.columns.duplicated()]
        
        df = df.dropna()

        return df

    def build_master(self,selected, smooth="wvl", **smooth_kwargs):

        all_data = []

        for _, row in selected.iterrows():

            df = self.load_csv(row["path"], round_cols=3, low=0, high=38.8)

            if smooth == "wvl":
                df = self.sp.denoise_df_by_wvl(df, **smooth_kwargs)

            elif smooth == "time":
                df = self.sp.denoise_df_time(df, **smooth_kwargs)

            df = df.assign(
                sample=int(row["sample"]),
                power_mW=float(row["power_mW"])
            )

            all_data.append(df)

        master = pd.concat(all_data)

        master_dfs = {
            sample: g.copy()
            for sample, g in master.groupby("sample")
        }

        return master_dfs





