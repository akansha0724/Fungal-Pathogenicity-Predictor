"""
Publication-quality visualisations — all plots saved to results/.
Uses Agg backend (no display required).
"""

import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
    "figure.dpi"        : 130,
})

_C = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6",
      "#1abc9c", "#e67e22", "#34495e", "#e91e63"]


def _save(fig, name):
    path = f"{RESULTS_DIR}/{name}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_class_distribution(y_series: pd.Series):
    counts = y_series.value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(counts.index, counts.values,
                   color=_C[1], edgecolor="white", height=0.65)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)
    ax.set_xlabel("Sample Count")
    ax.set_title("Class Distribution -- Plant Pathogenic Capacity",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, counts.max() * 1.15)
    plt.tight_layout()
    _save(fig, "01_class_distribution.png")


def plot_cv_comparison(cv_df: pd.DataFrame):
    metrics = ["Accuracy", "Macro F1", "Weighted F1"]
    stds    = ["Accuracy Std", "Macro F1 Std", "Weighted F1 Std"]
    x       = np.arange(len(metrics))
    width   = 0.25
    models  = cv_df.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model, color) in enumerate(zip(models, _C)):
        means  = [cv_df.loc[model, m] for m in metrics]
        errors = [cv_df.loc[model, s] for s in stds]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=model,
                      color=color, alpha=0.88, edgecolor="white",
                      yerr=errors, capsize=4,
                      error_kw={"elinewidth": 1.5})
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.013,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score  (5-Fold CV Mean +/- Std)")
    ax.set_title("Model Benchmark -- 5-Fold Stratified Cross-Validation",
                 fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    _save(fig, "02_model_comparison.png")


def plot_confusion_matrix(cm, class_names, model_name="Best Model"):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"label": "Recall (row-normalised)"})
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix -- {model_name}",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    _save(fig, "03_confusion_matrix.png")


def plot_feature_importance(rf_model, feature_names, top_n=20):
    clf = rf_model.named_steps["clf"]
    importances = pd.Series(clf.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    colors = [_C[2] if "__" in n else _C[1] for n in top.index]
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top.index, top.values, color=colors,
                   edgecolor="white", height=0.7)
    for bar, val in zip(bars, top.values):
        ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    ax.legend(handles=[
        mpatches.Patch(color=_C[1], label="Categorical feature"),
        mpatches.Patch(color=_C[2], label="TF-IDF (text) feature"),
    ], loc="lower right")
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title(f"Top {top_n} Feature Importances -- Random Forest",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "04_feature_importance.png")


def plot_shap_summary(shap_values_3d, X_test, feature_names,
                      class_names, top_n=15):
    """
    shap_values_3d: ndarray (n_samples, n_features, n_classes)
    Plots global mean-|SHAP| bar chart (aggregated across all classes).
    """
    import shap

    # Mean absolute SHAP across samples and classes -> (n_features,)
    mean_abs = np.abs(shap_values_3d).mean(axis=(0, 2))
    top_idx  = np.argsort(mean_abs)[-top_n:]

    top_names  = [feature_names[i] for i in top_idx]
    top_values = mean_abs[top_idx]

    # Sort for horizontal bar chart
    order = np.argsort(top_values)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([top_names[i] for i in order],
            [top_values[i] for i in order],
            color=_C[0], edgecolor="white", height=0.7)
    ax.set_xlabel("Mean |SHAP value| (averaged over all classes)")
    ax.set_title(f"Top {top_n} Features by SHAP Impact -- Random Forest",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "05_shap_importance.png")

    # Beeswarm for the dominant class (leaf/fruit/seed_pathogen = class 0)
    dominant_idx = 0
    sv_dom = shap_values_3d[:, :, dominant_idx]   # (n_samples, n_features)
    top_feat_idx = np.argsort(np.abs(sv_dom).mean(axis=0))[-top_n:]

    fig2 = plt.figure(figsize=(10, 7))
    shap.summary_plot(
        sv_dom[:, top_feat_idx],
        X_test[:, top_feat_idx],
        feature_names=[feature_names[i] for i in top_feat_idx],
        show=False,
        plot_type="dot",
        max_display=top_n,
    )
    plt.title(
        f"SHAP Beeswarm -- class: {class_names[dominant_idx] if dominant_idx < len(class_names) else 'dominant'}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig2, "06_shap_beeswarm.png")


def plot_per_class_f1(metrics: dict):
    rdf = pd.DataFrame(metrics["report"]).T
    rdf = rdf.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
    rdf = rdf.sort_values("f1-score", ascending=True)

    colors = [_C[1] if v >= 0.7 else _C[2] for v in rdf["f1-score"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(rdf.index, rdf["f1-score"],
                   color=colors, edgecolor="white", height=0.65)
    for bar, val in zip(bars, rdf["f1-score"]):
        ax.text(bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.axvline(0.7, linestyle="--", color="gray", alpha=0.6, label="F1 = 0.70")
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("F1-Score")
    ax.set_title("Per-Class F1-Score -- Best Model (Test Set)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, "07_per_class_f1.png")
