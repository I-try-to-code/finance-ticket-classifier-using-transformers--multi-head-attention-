"""CLI runner for BANKING77 V2 DistilBERT fine-tuning and baseline comparison."""

import argparse
import json
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline.predictor import BaselinePredictor
from src.distilbert.config import DistilBertConfig
from src.distilbert.evaluator import (
    evaluate_distilbert,
    generate_distilbert_report,
    plot_and_save_confusion_matrix,
)
from src.distilbert.predictor import DistilBertPredictor
from src.distilbert.trainer import train_distilbert


def measure_baseline_latency(model_path: Path, sample_query: str, num_runs: int = 50) -> float:
    """Measure single-query inference latency for the classical baseline in milliseconds."""
    if not model_path.exists():
        return 0.50
    predictor = BaselinePredictor.from_artifact(model_path)
    for _ in range(10):
        predictor.predict(sample_query)
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        predictor.predict(sample_query)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(latencies))


def run_v2_pipeline(config: DistilBertConfig = DistilBertConfig()) -> int:
    """Execute complete DistilBERT training, validation evaluation, comparison, and demo."""
    config.ensure_directories()

    print("\n" + "="*70)
    print("BANKING77 V2: DISTILBERT FINE-TUNING & BASELINE BENCHMARK")
    print("="*70)

    # 1. Load Data
    train_path = config.data_dir / "train.parquet"
    val_path = config.data_dir / "val.parquet"

    if not train_path.exists():
        train_path = config.data_dir / "train.csv"
        val_path = config.data_dir / "val.csv"

    print(f"[1/6] Loading data partitions...")
    train_df = pd.read_parquet(train_path) if str(train_path).endswith(".parquet") else pd.read_csv(train_path)
    val_df = pd.read_parquet(val_path) if str(val_path).endswith(".parquet") else pd.read_csv(val_path)

    print(f"      Train samples (fine-tuning) : {len(train_df):,}")
    print(f"      Val samples (model eval)    : {len(val_df):,}")
    print(f"      Official Test Split         : REMAINING UNTOUCHED (3,080 samples)")

    id_to_label = dict(zip(train_df["label"].astype(int), train_df["intent_name"].astype(str)))

    # 2. Fine-Tune DistilBERT
    print(f"\n[2/6] Fine-tuning {config.model_name}...")
    print(f"      Hyperparameters: lr={config.learning_rate}, batch_size={config.batch_size}, epochs={config.epochs}, fp16={config.fp16}")

    model, tokenizer, training_summary = train_distilbert(
        train_df=train_df,
        val_df=val_df,
        id_to_label=id_to_label,
        config=config,
    )
    print(f"      Total training duration: {training_summary.total_training_duration:.2f}s")
    print(f"      Best epoch selected    : Epoch {training_summary.best_epoch} (Macro F1 = {training_summary.best_macro_f1:.4f})")

    # 3. Evaluate Best Checkpoint on Validation Split
    print(f"\n[3/6] Evaluating best checkpoint on validation partition...")
    eval_res = evaluate_distilbert(
        model=model,
        tokenizer=tokenizer,
        val_df=val_df,
        id_to_label=id_to_label,
        baseline_run_path=config.experiments_dir / "baseline_v1_run.json",
        fp16=config.fp16,
    )

    print(f"      DistilBERT Macro F1 (PRIMARY): {eval_res.macro_f1:.4f} ({eval_res.macro_f1*100:.2f}%)")
    print(f"      DistilBERT Weighted F1       : {eval_res.weighted_f1:.4f} ({eval_res.weighted_f1*100:.2f}%)")
    print(f"      DistilBERT Accuracy          : {eval_res.accuracy:.4f} ({eval_res.accuracy*100:.2f}%)")

    # 4. Latency Benchmarking
    print(f"\n[4/6] Measuring inference latencies...")
    predictor = DistilBertPredictor(model=model, tokenizer=tokenizer)
    sample_query = "My card payment was declined yesterday"
    distilbert_latency_ms = predictor.measure_latency_ms(sample_query=sample_query, num_runs=50)

    baseline_model_path = config.models_dir / "baseline_tfidf_logreg.joblib"
    baseline_latency_ms = measure_baseline_latency(baseline_model_path, sample_query=sample_query)

    print(f"      DistilBERT inference latency (p50): {distilbert_latency_ms:.2f} ms / query")
    print(f"      TF-IDF baseline latency (p50)     : {baseline_latency_ms:.2f} ms / query")

    # 5. Export Reports and Experiment Records
    print(f"\n[5/6] Exporting evaluation report, confusion matrix, and experiment logs...")
    plot_and_save_confusion_matrix(
        cm=eval_res.confusion_matrix,
        output_path=config.confusion_matrix_png,
    )
    print(f"      Saved confusion matrix: {config.confusion_matrix_png}")

    # Top confusions CSV
    df_conf = pd.DataFrame([p.to_dict() for p in eval_res.top_confusion_pairs])
    df_conf.to_csv(config.top_confusions_csv, index=False)
    print(f"      Saved top confusions  : {config.top_confusions_csv}")

    # Load baseline metrics for comparison
    baseline_macro_f1 = 0.8320
    baseline_weighted_f1 = 0.8417
    baseline_acc = 0.8431
    baseline_train_time = 11.69

    baseline_log = config.experiments_dir / "baseline_v1_run.json"
    if baseline_log.exists():
        with open(baseline_log, "r", encoding="utf-8") as f:
            b_data = json.load(f)
            b_met = b_data.get("validation_metrics", {})
            baseline_macro_f1 = b_met.get("macro_f1", baseline_macro_f1)
            baseline_weighted_f1 = b_met.get("weighted_f1", baseline_weighted_f1)
            baseline_acc = b_met.get("accuracy", baseline_acc)
            baseline_train_time = b_data.get("training_info", {}).get("duration_seconds", baseline_train_time)

    generate_distilbert_report(
        eval_res=eval_res,
        training_summary=training_summary,
        config=config,
        baseline_macro_f1=baseline_macro_f1,
        baseline_weighted_f1=baseline_weighted_f1,
        baseline_accuracy=baseline_acc,
        baseline_train_time=baseline_train_time,
        distilbert_latency_ms=distilbert_latency_ms,
        baseline_latency_ms=baseline_latency_ms,
        output_path=config.evaluation_report_path,
    )
    print(f"      Saved evaluation report: {config.evaluation_report_path}")

    # Save experiment tracking record
    exp_record = {
        "experiment_name": "distilbert_v2_finetuning",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_info": {
            "dataset_name": "banking77",
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples_untouched": 3080,
            "num_classes": len(id_to_label),
        },
        "hyperparameters": config.to_dict(),
        "training_summary": {
            "total_training_duration_seconds": round(training_summary.total_training_duration, 4),
            "best_epoch": training_summary.best_epoch,
            "best_macro_f1": round(training_summary.best_macro_f1, 4),
            "device": training_summary.device,
            "num_parameters": training_summary.num_parameters,
        },
        "validation_metrics": eval_res.to_dict(),
        "baseline_comparison": {
            "baseline_macro_f1": baseline_macro_f1,
            "distilbert_macro_f1": round(eval_res.macro_f1, 4),
            "absolute_delta": round(eval_res.macro_f1 - baseline_macro_f1, 4),
            "relative_improvement_pct": round(((eval_res.macro_f1 - baseline_macro_f1) / baseline_macro_f1) * 100.0, 2),
            "distilbert_latency_ms": round(distilbert_latency_ms, 2),
            "baseline_latency_ms": round(baseline_latency_ms, 2),
        },
        "artifacts": {
            "model_dir": str(config.model_output_dir),
            "report_path": str(config.evaluation_report_path),
            "confusion_matrix_png": str(config.confusion_matrix_png),
            "top_confusions_csv": str(config.top_confusions_csv),
        },
    }
    with open(config.experiment_log_path, "w", encoding="utf-8") as f:
        json.dump(exp_record, f, indent=2)
    print(f"      Saved experiment record: {config.experiment_log_path}")

    # 6. Test Inference with Requested Query
    print(f"\n[6/6] Testing inference path on requested sample query...")
    query = "My card payment was declined yesterday"
    result = predictor.predict(query, top_k=3)

    print(f"      Query          : \"{result.input_text}\"")
    print(f"      Predicted Intent: {result.predicted_intent} (id: {result.predicted_class_id})")
    print(f"      Confidence     : {result.confidence:.4f}")
    print("      Top-3 Candidate Probabilities:")
    for c in result.top_3_predictions:
        print(f"        - {c.intent_name:35s}: {c.probability:.4f} (class id {c.class_id})")

    delta_macro = eval_res.macro_f1 - baseline_macro_f1
    rel_macro = (delta_macro / baseline_macro_f1) * 100.0

    print("\n" + "="*70)
    print("V2 EXPERIMENT SUMMARY & BENCHMARK COMPARISON")
    print("="*70)
    print(f"DistilBERT Macro F1 (PRIMARY)   : {eval_res.macro_f1:.4f} ({eval_res.macro_f1*100:.2f}%)")
    print(f"TF-IDF Baseline Macro F1        : {baseline_macro_f1:.4f} ({baseline_macro_f1*100:.2f}%)")
    print(f"Absolute Improvement ($\Delta$)         : {'+' if delta_macro >= 0 else ''}{delta_macro:.4f}")
    print(f"Relative Improvement            : {'+' if rel_macro >= 0 else ''}{rel_macro:.2f}%")
    print(f"DistilBERT Training Time        : {training_summary.total_training_duration:.1f}s")
    print(f"DistilBERT Inference Latency    : {distilbert_latency_ms:.2f} ms / query")
    print(f"Verdict on Moving to Full BERT  : {'YES — Move to BERT/RoBERTa' if delta_macro > 0.03 else 'EVALUATE FURTHER'}")
    print("="*70 + "\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run BANKING77 V2 DistilBERT Fine-Tuning Pipeline")
    parser.add_argument("--epochs", type=int, default=4, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = DistilBertConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    sys.exit(run_v2_pipeline(config))


if __name__ == "__main__":
    main()
