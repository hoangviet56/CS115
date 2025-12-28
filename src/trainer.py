import torch


# ============================================================
# Utility functions for training & evaluation
# ============================================================

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    Fit model (closed-form) và trả về:
        train_mse, test_mse
    """
    model.fit(X_train, y_train)

    train_mse = model.mse(X_train, y_train)
    test_mse = model.mse(X_test, y_test)

    return train_mse, test_mse


def sweep_feature_dimension(
    model_class,
    feature_dims,
    X_train,
    y_train,
    X_test,
    y_test,
    model_kwargs_fn
):
    """
    Sweep số chiều feature để quan sát double descent.

    Args:
        model_class: LinearRegressionModel hoặc RFFModel
        feature_dims: list các giá trị dimension
        model_kwargs_fn: function(d) -> dict
            (dùng để khởi tạo model theo dimension)

    Returns:
        results: list of dict
            {
              "dim": d,
              "train_mse": ...,
              "test_mse": ...
            }
    """
    results = []

    for d in feature_dims:
        model = model_class(**model_kwargs_fn(d))

        train_mse, test_mse = train_and_evaluate(
            model, X_train, y_train, X_test, y_test
        )

        results.append({
            "dim": d,
            "train_mse": train_mse,
            "test_mse": test_mse
        })

        print(
            f"d={d:4d} | train MSE={train_mse:.4e} | test MSE={test_mse:.4e}"
        )

    return results
