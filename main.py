"""
Fungal Pathogenicity Prediction
================================
End-to-end ML pipeline on the FungalTraits dataset.

Steps:
  1. Load + clean  (class consolidation, missing-value strategy)
  2. Feature engineering  (OrdinalEncoder + TF-IDF via ColumnTransformer)
  3. 5-fold stratified CV benchmark  (RF, GradientBoosting, SVM)
  4. Train best model (Gradient Boosting) on full training split
  5. Train Random Forest separately for SHAP / feature-importance analysis
  6. Save all artefacts to results/

Usage:
    python main.py
"""

import warnings
import matplotlib
matplotlib.use("Agg")   # save plots to file; no display needed

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

from src.evaluate  import compute_shap_rf, evaluate, print_summary
from src.models    import cross_validate_all, save_model, train_model
from src.preprocess import (
    TARGET, build_preprocessor, get_feature_cols,
    get_feature_names, load_and_clean,
)
from src.visualize import (
    plot_class_distribution, plot_confusion_matrix, plot_cv_comparison,
    plot_feature_importance, plot_per_class_f1, plot_shap_summary,
)

RANDOM_STATE = 42
BEST_MODEL   = "Gradient Boosting"   # highest CV accuracy & macro-F1


# ── 1. Load & inspect ─────────────────────────────────────────────────────────

print("=" * 62)
print("  FUNGAL PATHOGENICITY PREDICTION")
print("=" * 62)

df = load_and_clean()
print(f"\nLabeled samples : {len(df)}")
print(f"Target classes  : {df[TARGET].nunique()}\n")
print(df[TARGET].value_counts().to_string())

plot_class_distribution(df[TARGET])


# ── 2. Feature engineering ────────────────────────────────────────────────────

cat_cols, text_cols = get_feature_cols(df)
print(f"\nCategorical features : {len(cat_cols)}")
print(f"Text features        : {len(text_cols)}")

le = LabelEncoder()
y  = le.fit_transform(df[TARGET])

preprocessor  = build_preprocessor(cat_cols, text_cols)
X             = preprocessor.fit_transform(df)
feature_names = get_feature_names(preprocessor, cat_cols, text_cols)
print(f"Feature dimensions   : {X.shape[1]}")


# ── 3. Train / test split ─────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y,
)
print(f"Train : {X_train.shape[0]} | Test : {X_test.shape[0]}\n")


# ── 4. Cross-validation benchmark ────────────────────────────────────────────

print("-" * 62)
print("  5-FOLD STRATIFIED CROSS-VALIDATION")
print("-" * 62)
cv_results = cross_validate_all(X_train, y_train)

print("\nCV Results Summary:")
print(cv_results[["Accuracy", "Macro F1", "Weighted F1"]].round(4).to_string())
cv_results.to_csv("results/cv_results.csv")
plot_cv_comparison(cv_results)


# ── 5. Train best model (Gradient Boosting) ───────────────────────────────────

print(f"\nTraining {BEST_MODEL} on full training set ...")
best_model = train_model(X_train, y_train, BEST_MODEL)
save_model(best_model, "results/best_model.pkl")

metrics = evaluate(best_model, X_test, y_test, le)
print_summary(metrics, BEST_MODEL)

plot_confusion_matrix(metrics["cm"], metrics["class_names"],
                      model_name=BEST_MODEL)
plot_per_class_f1(metrics)


# ── 6. Random Forest for interpretability ─────────────────────────────────────

print("Training Random Forest for feature importance + SHAP ...")
rf_model = train_model(X_train, y_train, "Random Forest")

plot_feature_importance(rf_model, feature_names)

print("Computing SHAP values (this takes ~1 min) ...")
try:
    shap_values, feat_names = compute_shap_rf(
        rf_model, X_train, X_test, feature_names
    )
    plot_shap_summary(shap_values, X_test, feat_names, metrics["class_names"])
    print("SHAP analysis complete.")
except Exception as e:
    print(f"SHAP skipped: {e}")


# ── 7. Final summary ──────────────────────────────────────────────────────────

best_cv = cv_results.loc[BEST_MODEL]
print("\n" + "=" * 62)
print("  FINAL RESULTS")
print("=" * 62)
print(f"  Best model       : {BEST_MODEL}")
print(f"  CV  Accuracy     : {best_cv['Accuracy']:.4f}  ({best_cv['Accuracy']*100:.1f}%)")
print(f"  CV  Macro F1     : {best_cv['Macro F1']:.4f}")
print(f"  Test Accuracy    : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
print(f"  Test Macro F1    : {metrics['macro_f1']:.4f}")
print(f"  Test Weighted F1 : {metrics['weighted_f1']:.4f}")
print(f"\n  Artefacts saved  : results/")
print("=" * 62)
