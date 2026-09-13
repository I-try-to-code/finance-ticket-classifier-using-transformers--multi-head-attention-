"""Error analysis package comparing DistilBERT and BERT on Banking77 validation split."""

from src.analysis.collector import (
    ModelMetricSummary,
    ValidationPredictionRow,
    collect_validation_predictions,
)
from src.analysis.confusion import (
    ConfusionPairMetric,
    export_confusion_pairs_csv,
    extract_top_confusion_pairs,
    plot_confusion_matrix,
)
from src.analysis.focused import (
    FOCUSED_INTENTS,
    FocusedExampleRecord,
    export_focused_examples_csv,
    extract_focused_intent_examples,
)
from src.analysis.confidence import (
    HighConfidenceErrorRecord,
    LowConfidenceCorrectRecord,
    SemanticUncertaintyRecord,
    export_high_confidence_errors_csv,
    extract_confidence_analysis,
)
from src.analysis.categorizer import (
    CategorizedConfusionEvidence,
    build_qualitative_error_taxonomy,
)
from src.analysis.reporter import generate_error_analysis_markdown

__all__ = [
    "ValidationPredictionRow",
    "ModelMetricSummary",
    "collect_validation_predictions",
    "ConfusionPairMetric",
    "extract_top_confusion_pairs",
    "export_confusion_pairs_csv",
    "plot_confusion_matrix",
    "FOCUSED_INTENTS",
    "FocusedExampleRecord",
    "extract_focused_intent_examples",
    "export_focused_examples_csv",
    "HighConfidenceErrorRecord",
    "LowConfidenceCorrectRecord",
    "SemanticUncertaintyRecord",
    "extract_confidence_analysis",
    "export_high_confidence_errors_csv",
    "CategorizedConfusionEvidence",
    "build_qualitative_error_taxonomy",
    "generate_error_analysis_markdown",
]
