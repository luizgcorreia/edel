"""Targeted token deletion from head or tail."""

from edel.robustness.base import RobustnessTest

class HeadDeletion(RobustnessTest):
    """Delete N tokens from the head of the text."""
    
    name = "head_deletion"
    label = "Head Deletion"
    priority = "S"
    requires_reembed = True
    
    def perturb(self, texts: list[str], n: int) -> list[str]:
        if n <= 0:
            return texts
            
        perturbed_texts = []
        for text in texts:
            if not text or not text.strip():
                perturbed_texts.append(text)
                continue
                
            words = text.split()
            if len(words) <= n:
                perturbed_texts.append("")
            else:
                perturbed_texts.append(" ".join(words[n:]))
                
        return perturbed_texts


class TailDeletion(RobustnessTest):
    """Delete N tokens from the tail of the text."""
    
    name = "tail_deletion"
    label = "Tail Deletion"
    priority = "S"
    requires_reembed = True
    
    def perturb(self, texts: list[str], n: int) -> list[str]:
        if n <= 0:
            return texts
            
        perturbed_texts = []
        for text in texts:
            if not text or not text.strip():
                perturbed_texts.append(text)
                continue
                
            words = text.split()
            if len(words) <= n:
                perturbed_texts.append("")
            else:
                perturbed_texts.append(" ".join(words[:-n]))
                
        return perturbed_texts
