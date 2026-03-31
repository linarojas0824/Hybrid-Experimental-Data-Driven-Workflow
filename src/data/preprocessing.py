import numpy as np


def prepare_train_test(split_dict, calculator, feature_cols, scaler=None):
    df_train = calculator.transform(split_dict["X_train"])
    df_test = calculator.transform(split_dict["X_test"])

    X_train = df_train[feature_cols].to_numpy()
    X_test = df_test[feature_cols].to_numpy()

    y_train = np.asarray(split_dict["y_train"]).ravel()
    y_test = np.asarray(split_dict["y_test"]).ravel()

    fitted_scaler = None
    if scaler is not None:
        fitted_scaler = scaler.fit(X_train)
        X_train = fitted_scaler.transform(X_train)
        X_test = fitted_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, fitted_scaler