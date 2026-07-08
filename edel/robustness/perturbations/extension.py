"""Text extension perturbations (Random and Generative)."""

import random
import json
from edel.robustness.base import RobustnessTest

class RandomExtension(RobustnessTest):
    """Append N random tokens sampled from the corpus vocabulary."""
    
    name = "random_extension"
    label = "Random Extension"
    priority = "S"
    requires_reembed = True
    
    def perturb(self, texts: list[str], n: int) -> list[str]:
        if n <= 0:
            return texts
            
        # Build vocabulary from all texts
        vocab = []
        for text in texts:
            if text:
                vocab.extend(text.split())
                
        # Fallback if vocabulary is empty
        if not vocab:
            vocab = ["the", "a", "of", "and", "in", "to", "is", "that", "it", "was"]
            
        perturbed_texts = []
        for text in texts:
            if not text or not text.strip():
                perturbed_texts.append(text)
                continue
                
            rng = random.Random(hash(text) + n)
            # Sample n words with replacement from vocab
            sampled = rng.choices(vocab, k=n)
            
            # Append to text
            perturbed_texts.append(text + " " + " ".join(sampled))
            
        return perturbed_texts


class GenerativeExtension(RobustnessTest):
    """Append N words of coherent continuation using an LLM."""
    
    name = "generative_extension"
    label = "Generative Extension (LLM)"
    priority = "S"
    requires_reembed = True
    
    def perturb(self, texts: list[str], n: int) -> list[str]:
        if n <= 0:
            return texts
            
        # Retrieve LLM client if attached
        client = getattr(self, "llm_client", None)
        if client is None:
            # Fallback to a static readable mock continuation to prevent errors
            # when LLM is unavailable (e.g. offline, local tests)
            return [text + f" [generative extension of {n} words]" for text in texts]
            
        perturbed_texts = []
        for text in texts:
            if not text or not text.strip():
                perturbed_texts.append(text)
                continue
                
            prompt = (
                f"You are a text completion assistant. Complete the following text "
                f"by appending exactly {n} words. The continuation must be coherent, grammatically correct, "
                f"and match the scientific/technical tone of the original text. Do not repeat the original text. "
                f"Your response must be a JSON object with a single key 'continuation' containing only "
                f"the generated words.\n\n"
                f"Text: {text}"
            )
            
            try:
                res_str = client.generate(prompt)
                try:
                    res_dict = json.loads(res_str)
                    continuation = res_dict.get("continuation", "")
                except json.JSONDecodeError:
                    # Fallback to using the raw text response if not valid JSON
                    continuation = res_str.strip()
                
                # Clean up prompt prefix/postfix formatting or backticks if LLM returns them
                if continuation.startswith("```"):
                    lines = continuation.split('\n')
                    clean_lines = [l for l in lines if not l.strip().startswith("```")]
                    continuation = "\n".join(clean_lines).strip()
                    try:
                        res_dict = json.loads(continuation)
                        continuation = res_dict.get("continuation", "")
                    except json.JSONDecodeError:
                        pass
                
                perturbed_texts.append(text + " " + continuation)
            except Exception as e:
                print(f"Error in GenerativeExtension LLM call: {e}")
                perturbed_texts.append(text + f" [failed continuation of {n} words]")
                
        return perturbed_texts
