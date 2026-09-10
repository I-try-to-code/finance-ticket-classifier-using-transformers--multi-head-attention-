"""CLI runner for BANKING77 V1 Baseline (TF-IDF + Logistic Regression)."""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline.config import BaselineConfig
from src.baseline.evaluator import (
    evaluate_baseline,
    export_confusion_pairs_csv,
    generate_baseline_evaluation_report,
    plot_and_save_confusion_matrix,
)
from src.baseline.predictor import (
    BaselinePredictor,
    save_baseline_artifact,
)
from src.baseline.tracker import (
    create_experiment_record,
    save_experiment_record,
)
from src.baseline.trainer import train_baseline


def run_baseline_pipeline(config: BaselineConfig = BaselineConfig()) -> int:
    """Execute complete training, evaluation, artifact export, and demo inference."""
    config.ensure_directories()

    print("\n" + "="*70)
    print("BANKING77 V1 BASELINE: TF-IDF + LOGISTIC REGRESSION")
    print("="*70)

    # 1. Load Data
    train_path = config.data_dir / "train.parquet"
    val_path = config.data_dir / "val.parquet"

    if not train_path.exists() or not val_path.exists():
        # Fall back to CSV if parquet does not exist
        train_path = config.data_dir / "train.csv"
        val_path = config.data_dir / "val.csv"

    print(f"[1/6] Loading data partitions...")
    train_df = pd.read_parquet(train_path) if str(train_path).endswith(".parquet") else pd.read_csv(train_path)
    val_df = pd.read_parquet(val_path) if str(val_path).endswith(".parquet") else pd.read_csv(val_path)

    print(f"      Train samples (for fitting): {len(train_df):,}")
    print(f"      Val samples (for evaluation): {len(val_df):,}")
    print(f"      Official Test Split        : REMAINING UNTOUCHED (3,080 samples)")

    # Build id_to_label mapping from train split
    id_to_label = dict(zip(train_df["label"].astype(int), train_df["intent_name"].astype(str)))
    num_classes = len(id_to_label)

    # 2. Train Pipeline
    print(f"\n[2/6] Training baseline pipeline on train split only...")
    print(f"      TF-IDF settings  : ngram_range={config.tfidf.ngram_range}, min_df={config.tfidf.min_df}, sublinear_tf={config.tfidf.sublinear_tf}")
    print(f"      LogReg settings  : solver={config.logreg.solver}, C={config.logreg.C}, max_iter={config.logreg.max_iter}")

    pipeline, train_duration, vocab_size = train_baseline(train_df, config=config)
    print(f"      Training finished in: {train_duration:.2f}s")
    print(f"      Learned vocabulary  : {vocab_size:,} n-gram features")

    # 3. Evaluate on Validation Set
    print(f"\n[3/6] Evaluating on validation set...")
    eval_res = evaluate_baseline(pipeline, val_df, id_to_label)
    print(f"      Macro F1 (PRIMARY) : {eval_res.macro_f1:.4f} ({eval_res.macro_f1*100:.2f}%)")
    print(f"      Weighted F1        : {eval_res.weighted_f1:.4f} ({eval_res.weighted_f1*100:.2f}%)")
    print(f"      Accuracy           : {eval_res.accuracy:.4f} ({eval_res.accuracy*100:.2f}%)")

    # 4. Save Artifacts & Visualizations
    print(f"\n[4/6] Exporting reports, confusion matrix, and top confusion pairs...")
    plot_and_save_confusion_matrix(
        eval_res.confusion_matrix,
        output_path=config.confusion_matrix_png,
    )
    print(f"      Saved confusion matrix plot: {config.confusion_matrix_png}")

    export_confusion_pairs_csv(
        eval_res.top_confusion_pairs,
        output_path=config.top_confusions_csv,
    )
    print(f"      Saved top confusion pairs  : {config.top_confusions_csv}")

    generate_baseline_evaluation_report(
        eval_res=eval_res,
        tfidf_params=config.tfidf.to_dict(),
        logreg_params=config.logreg.to_dict(),
        train_size=len(train_df),
        val_size=len(val_df),
        vocab_size=vocab_size,
        training_duration=train_duration,
        output_path=config.evaluation_report_path,
    )
    print(f"      Saved evaluation report    : {config.evaluation_report_path}")

    # 5. Serialize Model Artifact & Log Experiment
    print(f"\n[5/6] Serializing model artifact and tracking experiment run...")
    model_path = save_baseline_artifact(
        pipeline=pipeline,
        id_to_label=id_to_label,
        output_path=config.model_artifact_path,
        metadata={
            "vocab_size": vocab_size,
            "train_duration": train_duration,
            "macro_f1": eval_res.macro_f1,
            "accuracy": eval_res.accuracy,
        },
    )
    print(f"      Saved model artifact: {model_path}")

    exp_record = create_experiment_record(
        experiment_name="baseline_v1_tfidf_logreg",
        dataset_name="banking77",
        train_count=len(train_df),
        val_count=len(val_df),
        test_count=3080,
        num_classes=num_classes,
        random_seed=config.logreg.random_state,
        tfidf_params=config.tfidf.to_dict(),
        logreg_params=config.logreg.to_dict(),
        training_duration=train_duration,
        vocab_size=vocab_size,
        val_metrics=eval_res.to_dict(),
        artifacts={
            "model_path": str(config.model_artifact_path),
            "report_path": str(config.evaluation_report_path),
            "confusion_matrix_png": str(config.confusion_matrix_png),
            "top_confusions_csv": str(config.top_confusions_csv),
        },
    )
    save_experiment_record(exp_record, config.experiment_log_path)
    print(f"      Saved experiment run: {config.experiment_log_path}")

    # 6. Test Inference with Sample Query
    print(f"\n[6/6] Testing inference path on sample query...")
    predictor = BaselinePredictor.from_artifact(config.model_artifact_path)
    sample_query = "The cash machine did not give me my money"
    result = predictor.predict(sample_query, top_k=5)

    print(f"      Query          : \"{result.input_text}\"")
    print(f"      Predicted Class: {result.predicted_intent} (id: {result.predicted_class_id})")
    print(f"      Confidence     : {result.confidence:.4f}")
    print("      Top-5 Probabilities:")
    for c in result.top_k_predictions:
        print(f"        - {c.intent_name:45s}: {c.probability:.4f} (id: {c.class_id})")

    print("\n" + "="*70)
    print("V1 BASELINE PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run BANKING77 V1 Baseline Pipeline")
    parser.add_argument("--c", type=float, default=1.0, help="Logistic Regression C regularization parameter")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum optimizer iterations")
    parser.add_argument("--min-df", type=int, default=2, help="TF-IDF min_df")
    args = parser.parse_args()

    config = BaselineConfig()
    config.logreg = config.logreg.__class__(
        solver=config.logreg.solver,
        C=args.c,
        max_iter=args.max_iter,
        random_state=config.logreg.random_state,
    )
    config.tfidf = config.tfidf.__class__(
        ngram_range=config.tfidf.ngram_range,
        min_df=args.min_df,
        max_df=config.tfidf.max_df,
        sublinear_tf=config.tfidf.sublinear_tf,
        lowercase=config.tfidf.lowercase,
    )

    sys.exit(run_baseline_pipeline(config))


if __name__ == "__main__":
    main()
