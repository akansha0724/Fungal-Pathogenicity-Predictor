"""
Test-set evaluation and SHAP-based interpretability.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder


def evaluate(model, X_test, y_test, label_encoder: LabelEncoder) -> dict:
    y_pred      = model.predict(X_test)
    labels      = np.unique(y_test)
    class_names = label_encoder.inverse_transform(labels)

    report = classification_report(
        y_test, y_pred, labels=labels, target_names=class_names,
        zero_division=0, output_dict=True,
    )
    return {
        "accuracy"   : accuracy_score(y_test, y_pred),
        "macro_f1"   : f1_score(y_test, y_pred, average="macro",    zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "report"     : report,
        "cm"         : confusion_matrix(y_test, y_pred, labels=labels),
        "class_names": class_names,
        "labels"     : labels,
        "y_pred"     : y_pred,
    }


def print_summary(metrics: dict, model_name: str):
    print(f"\n{'='*62}")
    print(f"  {model_name} -- Test Set Results")
    print(f"{'='*62}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  Macro F1    : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1 : {metrics['weighted_f1']:.4f}")
    print(f"\n  Per-Class Report:")
    rdf = pd.DataFrame(metrics["report"]).T
    rdf = rdf.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
    print(rdf[["precision", "recall", "f1-score", "support"]].round(3).to_string())
    print(f"{'='*62}\n")


def compute_shap_rf(rf_model, X_train, X_test, feature_names,
                    n_background: int = 150):
    """
    SHAP TreeExplainer for a Random Forest pipeline.
    Returns (shap_values_3d, feature_names) where shap_values_3d has
    shape (n_samples, n_features, n_classes).
    """
    import shap

    clf    = rf_model.named_steps["clf"]
    bg_idx = np.random.default_rng(42).choice(
        len(X_train), min(n_background, len(X_train)), replace=False
    )
    explainer   = shap.TreeExplainer(clf, data=X_train[bg_idx],
                                     feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test)

    # Normalise to always-3D: (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)   # list[n_classes] of (n,f) -> (n,f,c)
    elif shap_values.ndim == 2:
        shap_values = shap_values[:, :, np.newaxis]

    return shap_values, feature_names
