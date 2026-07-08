"""Robustness test registry."""

from typing import cast
from edel.robustness.base import RobustnessTest
from edel.robustness.perturbations import (
    WordOrderShuffle,
    RandomTokenDeletion,
    NounMasking,
    NounSynonymSubstitution,
    VerbMasking,
    AdjectiveMasking,
    VerbSynonymSubstitution,
    AdjectiveSynonymSubstitution,
    HeadDeletion,
    TailDeletion,
    RandomExtension,
    GenerativeExtension,
    SentenceCountMetric,
    NumeralToWordSubstitution,
    WithinFieldDSLInjection,
    OutOfFieldDSLInjection,
    NounDuplication,
    AdjectiveDuplication,
    ConcretenessGradedDisplacement,
    SpecificityGradedDisplacement,
    DescriptiveNounRatio,
    PMFIRatioDistribution,
)

# Ordered list of instantiated test classes
ROBUSTNESS_REGISTRY: list[RobustnessTest] = [
    # Priority M
    WordOrderShuffle(),
    RandomTokenDeletion(),
    NounMasking(),
    NounSynonymSubstitution(),
    
    # Priority S
    VerbMasking(),
    AdjectiveMasking(),
    VerbSynonymSubstitution(),
    AdjectiveSynonymSubstitution(),
    HeadDeletion(),
    TailDeletion(),
    RandomExtension(),
    GenerativeExtension(),
    SentenceCountMetric(),
    NumeralToWordSubstitution(),
    WithinFieldDSLInjection(),
    OutOfFieldDSLInjection(),
    
    # Priority C
    NounDuplication(),
    AdjectiveDuplication(),
    ConcretenessGradedDisplacement(),
    SpecificityGradedDisplacement(),
    DescriptiveNounRatio(),
    PMFIRatioDistribution(),
]

def list_tests() -> list[str]:
    """Return the names of all registered tests."""
    return [test.name for test in ROBUSTNESS_REGISTRY]

def get_test(name: str) -> RobustnessTest:
    """Get a registered test by name."""
    for test in ROBUSTNESS_REGISTRY:
        if test.name == name:
            return test
    raise KeyError(f"Robustness test '{name}' not found.")
