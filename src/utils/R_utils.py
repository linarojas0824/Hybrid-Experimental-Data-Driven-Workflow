from pathlib import Path

def save_data_R(
    x_train,
    x_test,
    y_train,
    y_test,
    x_space,
    r_path
):
    r_path = Path(r_path)
    r_path.mkdir(parents=True, exist_ok=True)

    x_train.to_csv(r_path / "x_train.csv", index=False)
    x_test.to_csv(r_path / "x_test.csv", index=False)
    y_train.to_csv(r_path / "y_train.csv", index=False)
    y_test.to_csv(r_path / "y_test.csv", index=False)
    x_space.to_csv(r_path / "x_space.csv", index=False)