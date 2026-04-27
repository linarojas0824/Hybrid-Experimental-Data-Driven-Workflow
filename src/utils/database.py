import sqlite3
import pandas as pd
import numpy as np

def save_split_to_database(
    split_dict,
    experiment_name,
    db_path,
    table_name="splitting",
    if_exists="append"
):
    train_df = split_dict["X_train"].copy()
    test_df = split_dict["X_test"].copy()

    train_df["y"] = split_dict["y_train"]
    train_df["split"] = "train"
    train_df["experiment_name"] = experiment_name

    test_df["y"] = split_dict["y_test"]
    test_df["split"] = "test"
    test_df["experiment_name"] = experiment_name

    splitting_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    conn = sqlite3.connect(db_path)

    splitting_df.to_sql(
        table_name,
        conn,
        if_exists=if_exists,
        index=False
    )

    conn.close()

    return splitting_df

def save_predictions_database(
    pred_df,
    experiment_name,
    db_path,
    table_name="exp_predictions",
    comp_cols=("Cu", "Ni", "Al"),
    pred_col="e2_pred",
    std_col="std_pred"
):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    comp_cols = list(comp_cols)

    pred_exp_col = f"{pred_col}_{experiment_name}"
    std_exp_col = f"{std_col}_{experiment_name}"

    # 1. Create table if it does not exist
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            Cu REAL,
            Ni REAL,
            Al REAL,
            UNIQUE(Cu, Ni, Al)
        )
    """)

    # 2. Check existing columns
    existing_cols = pd.read_sql(
        f"PRAGMA table_info({table_name})",
        conn
    )["name"].tolist()

    # 3. Add prediction columns if needed
    if pred_exp_col not in existing_cols:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {pred_exp_col} REAL")

    if std_exp_col not in existing_cols:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {std_exp_col} REAL")

    # 4. Insert compositions if they do not exist
    for _, row in pred_df.iterrows():
        cursor.execute(f"""
            INSERT OR IGNORE INTO {table_name} (Cu, Ni, Al)
            VALUES (?, ?, ?)
        """, (
            row["Cu"],
            row["Ni"],
            row["Al"]
        ))

        # 5. Update prediction columns for that composition
        cursor.execute(f"""
            UPDATE {table_name}
            SET {pred_exp_col} = ?,
                {std_exp_col} = ?
            WHERE Cu = ? AND Ni = ? AND Al = ?
        """, (
            row[pred_col],
            row[std_col],
            row["Cu"],
            row["Ni"],
            row["Al"]
        ))

    conn.commit()
    conn.close()