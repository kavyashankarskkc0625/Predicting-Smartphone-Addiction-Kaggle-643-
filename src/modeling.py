"""Model definitions and training helpers."""
from catboost import CatBoostClassifier


def train_catboost(
    X_train,
    y_train,
    X_valid,
    y_valid,
    categorical_features,
    depth=7,
    iterations=1000,
    learning_rate=0.05,
    random_seed=42,
):
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_seed,
        verbose=100,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
    )

    return model