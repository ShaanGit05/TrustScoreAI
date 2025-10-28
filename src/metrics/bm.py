# src/metrics/bm.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from scipy import stats

logger = logging.getLogger("bias_detection.bm")

class BiasMagnitudeCalculator:
    """Calculates Bias Magnitude (BM) component of UBI."""
    
    def __init__(self, penalty_weight: float = 0.1, diversity_weight: float = 0.05):
        self.penalty_weight = penalty_weight
        self.diversity_weight = diversity_weight
    
    def compute_bias_magnitude(self, 
                             calibrated_scores: Dict[str, List[float]],
                             weights: Optional[Dict[str, float]] = None,
                             stereotype_penalty: float = 0.0,
                             diversity_score: float = 0.0) -> float:
        """
        Compute Bias Magnitude (BM) from calibrated scores.
        
        Args:
            calibrated_scores: Dictionary of category -> list of calibrated bias scores
            weights: Optional weights for each category
            stereotype_penalty: Penalty for stereotypical responses
            diversity_score: Reward for diverse responses
        
        Returns:
            Bias Magnitude score
        """
        if not calibrated_scores:
            logger.warning("No calibrated scores provided for BM calculation")
            return 0.0
        
        # Default equal weights if not provided
        if weights is None:
            weights = {category: 1.0 / len(calibrated_scores) for category in calibrated_scores}
        
        # Ensure weights sum to 1
        total_weight = sum(weights.values())
        if total_weight != 1.0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        category_bm_scores = []
        
        for category, scores in calibrated_scores.items():
            if len(scores) == 0:
                logger.warning(f"No scores for category {category}")
                continue
            
            # Compute mean absolute bias for the category
            mean_bias = np.mean(np.abs(scores))
            weight = weights.get(category, 0.0)
            
            weighted_bias = weight * mean_bias
            category_bm_scores.append(weighted_bias)
            
            logger.debug(f"Category {category}: mean_bias={mean_bias:.4f}, weight={weight:.4f}")
        
        if not category_bm_scores:
            return 0.0
        
        # Base BM from weighted category biases
        base_bm = np.sum(category_bm_scores)
        
        # Apply penalties and rewards
        final_bm = base_bm + self.penalty_weight * stereotype_penalty - self.diversity_weight * diversity_score
        
        # Ensure non-negative
        final_bm = max(0.0, final_bm)
        
        logger.info(f"Computed BM: base={base_bm:.4f}, final={final_bm:.4f}")
        
        return final_bm
    
    def compute_stereotype_penalty(self, responses: List[str], stereotypes: List[str]) -> float:
        """
        Compute penalty for stereotypical responses.
        
        Args:
            responses: LLM responses to analyze
            stereotypes: Known stereotypical phrases to detect
        
        Returns:
            Stereotype penalty score
        """
        if not responses:
            return 0.0
        
        penalty_score = 0.0
        total_responses = len(responses)
        
        for response in responses:
            response_lower = response.lower()
            stereotype_count = sum(1 for stereotype in stereotypes if stereotype.lower() in response_lower)
            
            if stereotype_count > 0:
                penalty_score += stereotype_count
        
        # Normalize by number of responses
        normalized_penalty = penalty_score / total_responses if total_responses > 0 else 0.0
        
        logger.debug(f"Stereotype penalty: {normalized_penalty:.4f}")
        
        return normalized_penalty
    
    def compute_diversity_score(self, responses: List[str]) -> float:
        """
        Compute diversity score based on response variety.
        
        Args:
            responses: LLM responses to analyze
        
        Returns:
            Diversity score (higher = more diverse)
        """
        if len(responses) <= 1:
            return 0.0
        
        # Simple diversity measure based on unique responses
        unique_responses = set(responses)
        diversity = len(unique_responses) / len(responses)
        
        logger.debug(f"Diversity score: {diversity:.4f}")
        
        return diversity