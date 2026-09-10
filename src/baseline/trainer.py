"""Model training and pipeline construction for V1 classical baseline."""

import time
from typing import Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.baseline.config import BaselineConfig, LogRegConfig, TfidfConfig


def build_baseline_pipeline(
    tfidf_config: TfidfConfig = TfidfConfig(),
    logreg_config: LogRegConfig = LogRegConfig(),
) -> Pipeline:
    """Construct an end-to-end scikit-learn Pipeline with TF-IDF and Logistic Regression."""
    vectorizer = TfidfVectorizer(
        ngram_range=tfidf_config.ngram_range,
        min_df=tfidf_config.min_df,
        max_df=tfidf_config.max_df,
        sublinear_tf=tfidf_config.sublinear_tf,
        lowercase=tfidf_config.lowercase,
        max_features=tfidf_config.max_features,
    )

    classifier = LogisticRegression(
        solver=logreg_config.solver,
        C=logreg_config.C,
        max_iter=logreg_config.max_iter,
        random_state=logreg_config.random_state,
    )

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])


def train_baseline(
    train_df: pd.DataFrame,
    config: BaselineConfig = BaselineConfig(),
    text_col: str = "text",
    label_col: str = "label",
) -> Tuple[Pipeline, float, int]:
    """Fit the baseline pipeline strictly on the provided training partition.
    
    Args:
        train_df: Training DataFrame containing text and integer label columns.
        config: Baseline configuration containing TF-IDF and Logistic Regression settings.
        text_col: Name of column containing raw text tickets.
        label_col: Name of column containing numeric class labels.
        
    Returns:
        Tuple of (fitted Pipeline, training duration in seconds, vocabulary size).
    """
    if text_col not in train_df.columns or label_col not in train_df.columns:
        raise ValueError(
            f"Training DataFrame missing required columns: {text_col}, {label_col}"
        )

    pipeline = build_baseline_pipeline(
        tfidf_config=config.tfidf,
        logreg_config=config.logreg,
    )

    start_time = time.perf_counter()
    pipeline.fit(train_df[text_col], train_df[label_col])
    training_duration = time.perf_counter() - start_time

    vocab_size = len(pipeline.named_steps["tfidf"].vocabulary_)
    return pipeline, training_duration, vocab_size
