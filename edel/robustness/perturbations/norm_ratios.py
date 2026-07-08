"""Descriptive to Noun ratio structural metric."""

from edel.robustness.base import RobustnessTest

class DescriptiveNounRatio(RobustnessTest):
    """Static metric to check descriptive-to-noun ratio correlation with displacement."""
    
    name = "descriptive_noun_ratio"
    label = "Descriptive-to-Noun Ratio"
    priority = "C"
    requires_reembed = False
    
    def perturb(self, texts: list[str], n: int) -> list[str]:
        # Since requires_reembed is False, this is not used for re-embedding.
        # Just return the texts as-is.
        return texts
