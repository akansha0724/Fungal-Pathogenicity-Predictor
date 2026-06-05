"""
Data loading, class consolidation, and feature pipeline construction.

Design decisions:
  - "MISSING" is encoded as a valid category — missingness is biologically
    informative (e.g., null Decay_substrate means the fungus is not a decomposer).
  - Multi-label / rare target classes (<= MIN_CLASS_SIZE samples) are grouped
    into 'other_pathogen' to make stratified CV feasible.
  - Text columns are TF-IDF encoded (max 100 features, min_df=2) — capped to
    avoid overfitting on the ~2k labeled sample set.
  - OrdinalEncoder is preferred over OneHotEncoder for tree-based models;
    SVM path applies StandardScaler downstream.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


TARGET         = "Plant_pathogenic_capacity_template"
MIN_CLASS_SIZE = 10

CATEGORICAL_COLS = [
    "Phylum", "Class", "Order", "Family", "GENUS",
    "primary_lifestyle", "Secondary_lifestyle",
    "Endophytic_interaction_capability_template",
    "Decay_substrate_template", "Decay_type_template",
    "Aquatic_habitat_template", "Animal_biotrophic_capacity_template",
    "Growth_form_template", "Fruitbody_type_template", "Hymenium_type_template",
    "Ectomycorrhiza_exploration_type_template", "Ectomycorrhiza_lineage_template",
    "primary_photobiont", "secondary_photobiont",
]

TEXT_COLS = [
    "Comment_on_lifestyle_template",
    "Specific_hosts",
]


def load_and_clean(path: str = "fungal_trait.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    df = df.dropna(subset=[TARGET]).copy()

    counts = df[TARGET].value_counts()
    rare   = counts[counts < MIN_CLASS_SIZE].index
    df[TARGET] = df[TARGET].apply(lambda v: "other_pathogen" if v in rare else v)

    # Drop any bucket that still ended up too small for stratified CV
    final_counts = df[TARGET].value_counts()
    df = df[~df[TARGET].isin(final_counts[final_counts < 2].index)]
    return df


def get_feature_cols(df: pd.DataFrame):
    cat  = [c for c in CATEGORICAL_COLS if c in df.columns]
    text = [c for c in TEXT_COLS        if c in df.columns]
    return cat, text


class _NanSafeTfidf(BaseEstimator, TransformerMixin):
    """TF-IDF wrapper that coerces NaN -> '' before vectorising."""

    def __init__(self, max_features=100):
        self.max_features = max_features   # stored for BaseEstimator.get_params

    def _clean(self, X):
        if hasattr(X, "iloc"):
            series = X.iloc[:, 0]
        else:
            series = pd.Series(np.array(X).ravel())
        return series.fillna("").astype(str).tolist()

    def _make_tfidf(self):
        return TfidfVectorizer(max_features=self.max_features,
                               stop_words="english", min_df=2)

    def fit(self, X, y=None):
        self.tfidf_ = self._make_tfidf()
        self.tfidf_.fit(self._clean(X))
        return self

    def transform(self, X):
        return self.tfidf_.transform(self._clean(X)).toarray()

    def get_feature_names_out(self):
        return self.tfidf_.get_feature_names_out()


def build_preprocessor(cat_cols, text_cols):
    transformers = [
        ("cat", Pipeline([
            ("enc", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-2,
            )),
        ]), cat_cols),
    ]
    for col in text_cols:
        transformers.append((f"tfidf_{col}", _NanSafeTfidf(max_features=100), [col]))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def get_feature_names(preprocessor, cat_cols, text_cols):
    names = list(cat_cols)
    for col in text_cols:
        vocab = preprocessor.named_transformers_[f"tfidf_{col}"].get_feature_names_out()
        names.extend([f"{col}__{w}" for w in vocab])
    return names
