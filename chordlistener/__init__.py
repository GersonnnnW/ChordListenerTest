"""ChordListener: audio -> notes, chords and sheet music."""

__version__ = "0.1.0"

# Bump when the analysis output would change for the same audio. Cached results
# are keyed on this, so old cache entries are ignored after an engine change.
ENGINE_VERSION = "4"

from .pipeline import Analysis, AnalysisOptions, analyze, analyze_samples  # noqa: E402

__all__ = ["Analysis", "AnalysisOptions", "analyze", "analyze_samples", "__version__", "ENGINE_VERSION"]
