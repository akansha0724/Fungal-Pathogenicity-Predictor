"""
Model definitions and cross-validated benchmarking.

Three models:
  1. Random Forest         — strong baseline for mixed categorical/text data
  2. Gradient Boosting     — HistGradientBoosting (10-50x faster, handles NaN
                             natively, class_weight support since sklearn 1.2)
  3. SVM (RBF)             — non-linear baseline; requires scaled features

All models use class_weight='balanced' to counter the heavy
leaf/fruit/seed_pathogen dominance (~78% of labeled samples).
Evaluation uses 5-fold stratified CV so each fold mirrors the full
class distribution.
"""

import pandas as pd
import joblib
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CV_FOLDS     = 5
RANDOM_STATE = 42

SCORING = {
    "accuracy"   : "accuracy",
    "macro_f1"   : "f1_macro",
    "weighted_f1": "f1_weighted",
}


def _rf():
    return Pipeline([("clf", RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    ))])


def _hgb():
    return Pipeline([("clf", HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.1,
        class_weight="balanced", random_state=RANDOM_STATE,
    ))])


def _svm():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10.0, gamma="scale",
                    class_weight="balanced", random_state=RANDOM_STATE,
                    probability=True)),
    ])


MODEL_REGISTRY = {
    "Random Forest"     : _rf,
    "Gradient Boosting" : _hgb,
    "SVM (RBF)"         : _svm,
}


def cross_validate_all(X, y) -> pd.DataFrame:
    cv   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, builder in MODEL_REGISTRY.items():
        print(f"  [{name}] cross-validating ...")
        res = cross_validate(builder(), X, y, cv=cv, scoring=SCORING,
                             return_train_score=False, n_jobs=-1)
        rows.append({
            "Model"          : name,
            "Accuracy"       : res["test_accuracy"].mean(),
            "Accuracy Std"   : res["test_accuracy"].std(),
            "Macro F1"       : res["test_macro_f1"].mean(),
            "Macro F1 Std"   : res["test_macro_f1"].std(),
            "Weighted F1"    : res["test_weighted_f1"].mean(),
            "Weighted F1 Std": res["test_weighted_f1"].std(),
        })
        print(f"       Acc={res['test_accuracy'].mean():.4f}  "
              f"Macro-F1={res['test_macro_f1'].mean():.4f}  "
              f"Weighted-F1={res['test_weighted_f1'].mean():.4f}")
    return pd.DataFrame(rows).set_index("Model")


def train_model(X_train, y_train, model_name: str):
    model = MODEL_REGISTRY[model_name]()
    model.fit(X_train, y_train)
    return model


def save_model(model, path: str):
    joblib.dump(model, path)
    print(f"  Saved -> {path}")
