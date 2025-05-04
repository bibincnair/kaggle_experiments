import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from tqdm import tqdm


def cross_validate_model(
    model_cls,
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    random_seed: int,
    **fit_kwargs,
):
    """Generic K‐fold CV, returns list of fitted models and OOF score."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    oof_preds = np.zeros_like(y, dtype=float)
    oof_true = np.zeros_like(y, dtype=float)
    models = []
    fold_scores = []
    for fold, (tr, va) in enumerate(kf.split(X, y)):
        m = model_cls(**params)
        m.fit(X[tr], y[tr], **fit_kwargs)
        p = m.predict(X[va])
        oof_preds[va] = p
        oof_true[va] = y[va]
        score = np.sqrt(mean_squared_log_error(np.expm1(y[va]), np.expm1(p)))
        fold_scores.append(score)
        models.append(m)
    overall = np.sqrt(mean_squared_log_error(np.expm1(oof_true), np.expm1(oof_preds)))
    return models, overall, fold_scores
