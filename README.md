# Fungal Pathogenicity Prediction

End-to-end ML pipeline for classifying fungal plant pathogenicity from ecological and morphological trait data, using the [FungalTraits v1.2](https://link.springer.com/article/10.1007/s13225-020-00466-2) dataset.

## Results

| Model | CV Accuracy | CV Macro F1 | CV Weighted F1 |
|---|---|---|---|
| **Gradient Boosting** | **92.2%** | **0.683** | **0.915** |
| Random Forest | 89.2% | 0.655 | 0.893 |
| SVM (RBF) | 72.5% | 0.561 | 0.784 |

*5-fold stratified cross-validation on training set (80/20 split).*  
*Macro F1 is the primary metric — accuracy alone is misleading given the 78% class imbalance.*

**Test set (held-out 20%):**

| Accuracy | Macro F1 | Weighted F1 |
|---|---|---|
| 91.2% | 0.736 | 0.913 |

## Dataset

- **Source**: FungalTraits v1.2 — 10,771 fungal genera with ecological and morphological traits
- **Labeled samples**: 2,054 genera with confirmed plant pathogenic capacity
- **Target classes**: 9 classes after consolidating rare / multi-label entries
- **Class imbalance**: `leaf/fruit/seed_pathogen` accounts for ~78% of labeled data

## Feature Engineering

Two-stage hybrid pipeline built with `sklearn.ColumnTransformer` (no data leakage):

1. **Categorical features (19 columns)** — `OrdinalEncoder` with `NaN` treated as its own category (`encoded_missing_value=-2`). Missingness is biologically meaningful: a null `Decay_substrate` signals the fungus is not a decomposer.

2. **Free-text features (2 columns)** — `TfidfVectorizer` (max 100 features, `min_df=2`) on host-name and lifestyle-comment fields.

Total feature dimensions: 88

## Key Design Decisions

| Decision | Reason |
|---|---|
| NaN → own category (not imputed) | Missingness is biologically informative, not random |
| Class consolidation (multi-label → `other_pathogen`) | Enables stratified CV; keeps class boundaries clean |
| `class_weight='balanced'` on all models | Prevents the 78% majority class from dominating |
| Macro F1 as primary metric | Weights all 9 classes equally; not skewed by majority |
| Gradient Boosting as best model | Consistently higher CV accuracy (+3%) and Macro F1 (+3%) vs RF |
| RF trained separately for SHAP | `TreeExplainer` gives fast, exact SHAP values; used for interpretability |

## Interpretability (SHAP)

SHAP TreeExplainer on Random Forest identifies the top predictors:

- `primary_lifestyle` and `Secondary_lifestyle` — strongest single features; encode the fundamental ecological strategy
- Taxonomic features (`Class`, `Order`, `Family`) — capture phylogenetic signal; related fungi share pathogenic strategies
- `Decay_substrate_template` — sharply separates saprotrophs from pathogens
- `Endophytic_interaction_capability_template` — distinguishes obligate pathogens from facultative ones

## Project Structure

```
.
├── main.py                  # Orchestration — runs the full pipeline
├── src/
│   ├── preprocess.py        # load_and_clean(), build_preprocessor(), NaN-safe TF-IDF
│   ├── models.py            # RF / GradBoost / SVM definitions + 5-fold CV
│   ├── evaluate.py          # Metrics, classification report, SHAP computation
│   └── visualize.py        # 7 publication-quality plots saved to results/
├── results/                 # Generated artefacts (plots, model .pkl, CV CSV)
├── fungal_trait.csv         # Raw FungalTraits dataset
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python main.py
```

All 7 plots and the trained model are saved to `results/`.

## Generated Outputs

| File | Description |
|---|---|
| `01_class_distribution.png` | Sample counts per class |
| `02_model_comparison.png` | CV benchmark — all 3 models, 3 metrics with error bars |
| `03_confusion_matrix.png` | Row-normalised confusion matrix (Gradient Boosting) |
| `04_feature_importance.png` | Top 20 RF feature importances |
| `05_shap_importance.png` | Mean absolute SHAP values (aggregated across all classes) |
| `06_shap_beeswarm.png` | SHAP beeswarm for dominant class |
| `07_per_class_f1.png` | Per-class F1 breakdown |
| `best_model.pkl` | Serialised Gradient Boosting pipeline |
| `cv_results.csv` | Full CV results table |
