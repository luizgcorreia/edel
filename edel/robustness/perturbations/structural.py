"""Structural robustness metrics (requires_reembed=False)."""

from edel.robustness.base import RobustnessTest

class SentenceCountMetric(RobustnessTest):
    """Static metric to check sentence count correlation with aspect displacement."""
    
    name = "sentence_count"
    label = "Sentence Count"
    priority = "S"
    requires_reembed = False
    
    def perturb(self, texts: list[str], n: int) -> list[str]:
        # Since requires_reembed is False, this is not used for re-embedding.
        # Just return the texts as-is.
        return texts


class PMFIRatioDistribution(RobustnessTest):
    """Static metric to check PMFI length ratio correlation with displacement."""
    
    name = "pmfi_ratio"
    label = "PMFI Length Ratio"
    priority = "C"
    requires_reembed = False
    
    def perturb(self, texts: list[str], n: int) -> list[str]:
        # Since requires_reembed is False, this is not used for re-embedding.
        # Just return the texts as-is.
        return texts
