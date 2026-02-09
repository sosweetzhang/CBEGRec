"""Neuro-symbolic translator: neural outputs -> symbolic LLM input."""
from typing import Dict, List, Tuple
import numpy as np

class NSTranslator:
    """Neuro-symbolic semantic translator."""
    
    def __init__(self, 
                 threshold_low: float = 0.3,
                 threshold_high: float = 0.7):
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
    
    def quantize(self, mastery_prob: float) -> str:
        """Map continuous probability to discrete level (Novice/Developing/Master)."""
        if mastery_prob < self.threshold_low:
            return "Novice"
        elif mastery_prob <= self.threshold_high:
            return "Developing"
        else:
            return "Master"
    
    def generate_profile(self,
                        mastery_probs: Dict[int, float],
                        concept_names: Dict[int, str],
                        target_concept: int = None) -> str:
        """Generate structured natural language profile."""
        mastered = []
        developing = []
        novice = []
        
        for c_id, prob in mastery_probs.items():
            level = self.quantize(prob)
            concept_name = concept_names.get(c_id, f"Concept_{c_id}")
            
            if level == "Master":
                mastered.append(concept_name)
            elif level == "Developing":
                developing.append(concept_name)
            else:
                novice.append(concept_name)
        profile_parts = []
        if mastered:
            profile_parts.append(f"Mastered: {', '.join(mastered)}")
        if developing:
            profile_parts.append(f"Developing: {', '.join(developing)}")
        if novice:
            profile_parts.append(f"Novice: {', '.join(novice)}")
        
        profile_text = f"Student Profile: [{'; '.join(profile_parts)}]"
        
        if target_concept is not None:
            target_name = concept_names.get(target_concept, f"Concept_{target_concept}")
            profile_text += f". Target Goal: {target_name}."
        if developing:
            profile_text += f" The student is uncertain about: {', '.join(developing)}."
        
        return profile_text
    
    def translate_bundle_to_text(self,
                                bundle: List[int],
                                mastery_probs: Dict[int, float],
                                concept_names: Dict[int, str]) -> str:
        """Convert bundle to text description."""
        bundle_names = [concept_names.get(c_id, f"Concept_{c_id}") for c_id in bundle]
        bundle_mastery = {c_id: mastery_probs.get(c_id, 0.0) for c_id in bundle}
        
        text = f"Concept Bundle: {', '.join(bundle_names)}. "
        text += f"Mastery levels: {bundle_mastery}"
        return text
