from sklearn.inspection import permutation_importance

def compute_permutation_importance(model, X, y, n_repeats=10, random_state=42):
    
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="r2"
    )

    return {
        "importances_mean": result.importances_mean,
        "importances_std": result.importances_std,
    }

def importance_to_df(importances, feature_names):
    import pandas as pd

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances["importances_mean"],
        "std": importances["importances_std"]
    })

    return df.sort_values("importance", ascending=False)