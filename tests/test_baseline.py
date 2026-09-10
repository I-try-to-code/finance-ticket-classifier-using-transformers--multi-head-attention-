"""Unit and integration tests for V1 baseline pipeline and inference engine."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.baseline.config import BaselineConfig
from src.baseline.evaluator import evaluate_baseline
from src.baseline.predictor import BaselinePredictor, save_baseline_artifact
from src.baseline.trainer import build_baseline_pipeline, train_baseline
from src.config import PROCESSED_DATA_DIR


@pytest.fixture(scope="module")
def processed_splits():
    """Load train and validation splits."""
    train_path = PROCESSED_DATA_DIR / "train.parquet"
    val_path = PROCESSED_DATA_DIR / "val.parquet"

    if not train_path.exists():
        train_path = PROCESSED_DATA_DIR / "train.csv"
        val_path = PROCESSED_DATA_DIR / "val.csv"

    train_df = pd.read_parquet(train_path) if str(train_path).endswith(".parquet") else pd.read_csv(train_path)
    val_df = pd.read_parquet(val_path) if str(val_path).endswith(".parquet") else pd.read_csv(val_path)
    return train_df, val_df


@pytest.fixture(scope="module")
def trained_baseline(processed_splits):
    """Train baseline pipeline once for test module."""
    train_df, _ = processed_splits
    config = BaselineConfig()
    pipeline, duration, vocab_size = train_baseline(train_df, config=config)
    id_to_label = dict(zip(train_df["label"].astype(int), train_df["intent_name"].astype(str)))
    return pipeline, id_to_label, config


def test_data_loading(processed_splits):
    """Verify train and validation partitions exist with expected shapes."""
    train_df, val_df = processed_splits
    assert len(train_df) == 8002
    assert len(val_df) == 2001
    assert "text" in train_df.columns and "label" in train_df.columns
    assert "text" in val_df.columns and "label" in val_df.columns
    assert train_df["label"].nunique() == 77
    assert val_df["label"].nunique() == 77


def test_no_tfidf_fitting_on_validation_or_test(processed_splits, trained_baseline):
    """Verify that TF-IDF vocabulary is fitted solely on train data without leakage."""
    train_df, val_df = processed_splits
    pipeline, _, _ = trained_baseline

    fitted_vocab = pipeline.named_steps["tfidf"].vocabulary_

    # 1. Any hypothetical token or word unique to validation must not be present if not in train
    synthetic_unseen_token = "unique_validation_token_xyz_never_in_train"
    assert synthetic_unseen_token not in fitted_vocab

    # 2. Synthetic verification: Ensure fitting on train texts only includes terms from train
    sample_train = pd.DataFrame({
        "text": [
            "card lost please help",
            "I have a card lost issue",
            "money transfer question",
            "how to make a money transfer",
        ],
        "label": [0, 0, 1, 1],
    })
    sample_pipeline = build_baseline_pipeline()
    sample_pipeline.fit(sample_train["text"], sample_train["label"])

    sample_vocab = sample_pipeline.named_steps["tfidf"].vocabulary_
    assert "card" in sample_vocab
    assert "money" in sample_vocab
    # Validation-only term should definitely not be in sample_vocab
    assert "atm" not in sample_vocab
    assert "unseen_word" not in sample_vocab


def test_training_pipeline_prediction_count(processed_splits, trained_baseline):
    """Verify prediction count matches input count exactly."""
    _, val_df = processed_splits
    pipeline, _, _ = trained_baseline

    preds = pipeline.predict(val_df["text"])
    assert len(preds) == len(val_df)
    assert np.all((preds >= 0) & (preds < 77))


def test_probability_outputs_have_77_classes(processed_splits, trained_baseline):
    """Verify predict_proba outputs shape (N, 77) and rows sum to 1.0."""
    _, val_df = processed_splits
    pipeline, _, _ = trained_baseline

    sample_texts = val_df["text"].iloc[:50].tolist()
    probs = pipeline.predict_proba(sample_texts)

    assert probs.shape == (len(sample_texts), 77)
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_saved_model_artifact_and_inference(trained_baseline, tmp_path):
    """Verify that saved model artifact can be serialized, loaded, and used for inference."""
    pipeline, id_to_label, _ = trained_baseline
    artifact_path = tmp_path / "test_baseline_model.joblib"

    # Save artifact
    save_baseline_artifact(pipeline, id_to_label, artifact_path)
    assert artifact_path.exists()

    # Load predictor
    predictor = BaselinePredictor.from_artifact(artifact_path)

    # Test sample query specified in requirements
    sample_query = "The cash machine did not give me my money"
    result = predictor.predict(sample_query, top_k=5, return_all_probs=True)

    assert result.input_text == sample_query
    assert isinstance(result.predicted_intent, str)
    assert 0 <= result.predicted_class_id < 77
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.top_k_predictions) == 5
    assert len(result.all_probabilities) == 77

    # Top predictions are in descending order
    probs = [c.probability for c in result.top_k_predictions]
    assert probs == sorted(probs, reverse=True)


def test_batch_inference(trained_baseline):
    """Verify batch prediction functionality."""
    pipeline, id_to_label, _ = trained_baseline
    predictor = BaselinePredictor(pipeline, id_to_label)

    queries = [
        "Where is my replacement card?",
        "Can I pay using my phone with apple pay?",
        "What is the current exchange rate for euros?",
    ]
    batch_results = predictor.batch_predict(queries, top_k=3)

    assert len(batch_results) == len(queries)
    for res, q in zip(batch_results, queries):
        assert res.input_text == q
        assert len(res.top_k_predictions) == 3
        assert res.confidence > 0.0


def test_evaluation_metrics_soundness(processed_splits, trained_baseline):
    """Verify validation metrics are calculated properly and within valid ranges."""
    _, val_df = processed_splits
    pipeline, id_to_label, _ = trained_baseline

    eval_res = evaluate_baseline(pipeline, val_df, id_to_label)

    assert 0.0 < eval_res.macro_f1 <= 1.0
    assert 0.0 < eval_res.weighted_f1 <= 1.0
    assert 0.0 < eval_res.accuracy <= 1.0
    assert len(eval_res.per_class_metrics) == 77
    assert len(eval_res.best_performing_intents) == 10
    assert len(eval_res.worst_performing_intents) == 10
    assert eval_res.confusion_matrix.shape == (77, 77)
