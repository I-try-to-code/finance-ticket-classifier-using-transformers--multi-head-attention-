"""Comprehensive dataset audit and text quality diagnostics for BANKING77."""

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Any, Dict, List, Set, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LengthStats:
    """Statistical summary of text lengths."""
    min: int
    max: int
    mean: float
    median: float
    std: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min": self.min,
            "max": self.max,
            "mean": round(self.mean, 2),
            "median": round(self.median, 2),
            "std": round(self.std, 2),
        }


@dataclass(frozen=True)
class ClassDistribution:
    """Class frequency distribution summary."""
    num_classes: int
    min_count: int
    max_count: int
    mean_count: float
    median_count: float
    std_count: float
    counts_per_class: Dict[str, int]
    percentages_per_class: Dict[str, float]


@dataclass
class QualityDiagnostics:
    """Data hygiene and text-quality diagnostics."""
    null_count: int = 0
    empty_count: int = 0
    whitespace_only_count: int = 0
    control_char_text_count: int = 0
    control_chars_found: List[str] = field(default_factory=list)
    html_markup_text_count: int = 0
    html_markup_examples: List[str] = field(default_factory=list)
    non_ascii_text_count: int = 0
    non_ascii_char_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    malformed_record_count: int = 0
    malformed_reasons: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SplitAudit:
    """Audit findings for an individual dataset split."""
    split_name: str
    num_examples: int
    char_length_stats: LengthStats
    word_length_stats: LengthStats
    class_distribution: ClassDistribution
    quality_diagnostics: QualityDiagnostics
    exact_duplicate_texts: int
    unique_duplicate_queries: int


@dataclass(frozen=True)
class DatasetAuditResult:
    """Complete dataset audit encompassing train, test, and cross-split findings."""
    dataset_name: str
    num_unique_intents: int
    intent_names: List[str]
    train_audit: SplitAudit
    test_audit: SplitAudit
    cross_split_duplicate_count: int
    cross_split_duplicate_examples: List[str]
    environment_metadata: Dict[str, Any]


def compute_length_stats(series: pd.Series) -> LengthStats:
    """Compute min, max, mean, median, and std for numeric series."""
    if series.empty:
        return LengthStats(min=0, max=0, mean=0.0, median=0.0, std=0.0)
    return LengthStats(
        min=int(series.min()),
        max=int(series.max()),
        mean=float(series.mean()),
        median=float(series.median()),
        std=float(series.std(ddof=1)) if len(series) > 1 else 0.0,
    )


def compute_class_distribution(df: pd.DataFrame, label_col: str = "intent_name") -> ClassDistribution:
    """Compute per-class counts, percentages, and overall balance statistics."""
    counts = df[label_col].value_counts().to_dict()
    total = len(df)
    percentages = {label: round((count / total) * 100.0, 4) for label, count in counts.items()}
    count_values = list(counts.values())

    return ClassDistribution(
        num_classes=len(counts),
        min_count=int(np.min(count_values)),
        max_count=int(np.max(count_values)),
        mean_count=float(np.mean(count_values)),
        median_count=float(np.median(count_values)),
        std_count=float(np.std(count_values, ddof=1)) if len(count_values) > 1 else 0.0,
        counts_per_class=counts,
        percentages_per_class=percentages,
    )


def perform_quality_diagnostics(df: pd.DataFrame, text_col: str = "text") -> QualityDiagnostics:
    """Audit text records for nulls, whitespace-only, control chars, HTML markup, and unicode."""
    diagnostics = QualityDiagnostics()

    html_tag_pattern = re.compile(r"<[^>]+>")
    html_entity_pattern = re.compile(r"&(?:[a-zA-Z]+|#\d+|#x[a-fA-F0-9]+);")

    control_chars_seen: Set[str] = set()
    non_ascii_seen: Dict[str, Dict[str, Any]] = {}

    for idx, raw_val in enumerate(df[text_col]):
        # 1. Null / NaN check
        if pd.isna(raw_val):
            diagnostics.null_count += 1
            diagnostics.malformed_record_count += 1
            diagnostics.malformed_reasons.append(f"Row {idx}: NaN/Null value encountered")
            continue

        if not isinstance(raw_val, str):
            diagnostics.malformed_record_count += 1
            diagnostics.malformed_reasons.append(f"Row {idx}: Non-string value type {type(raw_val)}")
            raw_val = str(raw_val)

        # 2. Empty and whitespace-only
        if len(raw_val) == 0:
            diagnostics.empty_count += 1
        elif raw_val.strip() == "":
            diagnostics.whitespace_only_count += 1

        # 3. HTML markup / entities
        if html_tag_pattern.search(raw_val) or html_entity_pattern.search(raw_val):
            diagnostics.html_markup_text_count += 1
            if len(diagnostics.html_markup_examples) < 5:
                diagnostics.html_markup_examples.append(raw_val)

        # 4. Control characters (Unicode category Cc/Cf excluding standard whitespace: \t, \n, \r)
        ctrl_in_text = [
            c for c in raw_val
            if unicodedata.category(c) in ("Cc", "Cf") and c not in ("\n", "\t", "\r")
        ]
        if ctrl_in_text:
            diagnostics.control_char_text_count += 1
            control_chars_seen.update(ctrl_in_text)

        # 5. Non-ASCII character audit (currency symbols, non-breaking space, typographical symbols)
        has_non_ascii = False
        for c in raw_val:
            code_point = ord(c)
            if code_point >= 128:
                has_non_ascii = True
                if c not in non_ascii_seen:
                    non_ascii_seen[c] = {
                        "char": c,
                        "code_point": code_point,
                        "hex": hex(code_point),
                        "unicode_name": unicodedata.name(c, "UNKNOWN"),
                        "frequency": 0,
                    }
                non_ascii_seen[c]["frequency"] += 1

        if has_non_ascii:
            diagnostics.non_ascii_text_count += 1

    diagnostics.control_chars_found = sorted(list(control_chars_seen))
    diagnostics.non_ascii_char_details = non_ascii_seen
    return diagnostics


def audit_single_split(df: pd.DataFrame, split_name: str, text_col: str = "text", label_col: str = "intent_name") -> SplitAudit:
    """Run comprehensive audit on a single DataFrame split."""
    # Length metrics
    char_lens = df[text_col].astype(str).str.len()
    word_lens = df[text_col].astype(str).apply(lambda s: len(s.split()))

    char_stats = compute_length_stats(char_lens)
    word_stats = compute_length_stats(word_lens)

    # Class distribution
    class_dist = compute_class_distribution(df, label_col=label_col)

    # Hygiene diagnostics
    diagnostics = perform_quality_diagnostics(df, text_col=text_col)

    # Duplicates within split
    # exact duplicate rows (all occurrences of non-unique texts)
    exact_duplicates = int(df.duplicated(subset=[text_col], keep=False).sum())
    # number of duplicate texts beyond the first occurrence
    unique_duplicate_queries = int(df.duplicated(subset=[text_col], keep="first").sum())

    return SplitAudit(
        split_name=split_name,
        num_examples=len(df),
        char_length_stats=char_stats,
        word_length_stats=word_stats,
        class_distribution=class_dist,
        quality_diagnostics=diagnostics,
        exact_duplicate_texts=exact_duplicates,
        unique_duplicate_queries=unique_duplicate_queries,
    )


def audit_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    intent_names: List[str],
    dataset_name: str = "banking77",
    environment_metadata: Dict[str, Any] = None,
) -> DatasetAuditResult:
    """Execute complete dataset audit for train and test splits plus cross-split overlap."""
    if environment_metadata is None:
        environment_metadata = {}

    train_audit = audit_single_split(train_df, split_name="train")
    test_audit = audit_single_split(test_df, split_name="test")

    # Cross-split overlap check
    train_texts: Set[str] = set(train_df["text"].astype(str))
    test_texts: Set[str] = set(test_df["text"].astype(str))
    overlap = train_texts.intersection(test_texts)

    return DatasetAuditResult(
        dataset_name=dataset_name,
        num_unique_intents=len(intent_names),
        intent_names=intent_names,
        train_audit=train_audit,
        test_audit=test_audit,
        cross_split_duplicate_count=len(overlap),
        cross_split_duplicate_examples=sorted(list(overlap))[:10],
        environment_metadata=environment_metadata,
    )
