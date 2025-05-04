from sklearn.metrics import cohen_kappa_score
import numpy as np
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer



def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def threshold_rounder(predictions, thresholds):
    return np.where(
        predictions < thresholds[0],
        0,
        np.where(
            predictions < thresholds[1], 1, np.where(predictions < thresholds[2], 2, 3)
        ),
    )


def optimize_thresholds(y_true, predictions):
    def objective(thresholds):
        rounded_preds = threshold_rounder(predictions, thresholds)
        return -quadratic_weighted_kappa(y_true, rounded_preds)

    result = minimize(objective, x0=[0.5, 1.5, 2.5], method="Nelder-Mead")
    return result.x if result.success else [0.5, 1.5, 2.5]