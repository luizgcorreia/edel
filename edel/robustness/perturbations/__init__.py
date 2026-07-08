"""Perturbation implementations."""

from .word_order import WordOrderShuffle
from .token_deletion import RandomTokenDeletion
from .pos_masking import NounMasking, VerbMasking, AdjectiveMasking
from .pos_synonym import NounSynonymSubstitution, VerbSynonymSubstitution, AdjectiveSynonymSubstitution
from .targeted_deletion import HeadDeletion, TailDeletion
from .extension import RandomExtension, GenerativeExtension
from .structural import SentenceCountMetric, PMFIRatioDistribution
from .alphanumeric import NumeralToWordSubstitution
from .dsl import WithinFieldDSLInjection, OutOfFieldDSLInjection
from .duplication import NounDuplication, AdjectiveDuplication
from .norms import ConcretenessGradedDisplacement, SpecificityGradedDisplacement
from .norm_ratios import DescriptiveNounRatio

__all__ = [
    "WordOrderShuffle",
    "RandomTokenDeletion",
    "NounMasking",
    "VerbMasking",
    "AdjectiveMasking",
    "NounSynonymSubstitution",
    "VerbSynonymSubstitution",
    "AdjectiveSynonymSubstitution",
    "HeadDeletion",
    "TailDeletion",
    "RandomExtension",
    "GenerativeExtension",
    "SentenceCountMetric",
    "PMFIRatioDistribution",
    "NumeralToWordSubstitution",
    "WithinFieldDSLInjection",
    "OutOfFieldDSLInjection",
    "NounDuplication",
    "AdjectiveDuplication",
    "ConcretenessGradedDisplacement",
    "SpecificityGradedDisplacement",
    "DescriptiveNounRatio",
]
