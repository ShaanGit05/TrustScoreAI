# src/metrics/sr.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger("bias_detection.sr")

class DisparityCalculator:
    """Calculates Selection Rate and Disparity (DP) component of UBI."""
    
    def __init__(self, min_group_size: int = 3, smoothing_factor: float = 0.01):
        self.min_group_size = min_group_size
        self.smoothing_factor = smoothing_factor
    
    def compute_selection_rates(self, 
                              group_responses: Dict[str, List[Any]],
                              selection_criteria: callable = None) -> Dict[str, float]:
        """
        Compute selection rates for each group.
        
        Args:
            group_responses: Dictionary of group -> list of responses
            selection_criteria: Function to determine if response meets selection criteria
        
        Returns:
            Dictionary of group -> selection rate
        """
        if selection_criteria is None:
            # Default criteria: any non-empty response counts as selected
            def selection_criteria(x: Any) -> bool:
                text = str(x).lower().strip()
                return len(text) > 0
        
        selection_rates = {}
        
        for group, responses in group_responses.items():
            if len(responses) < self.min_group_size:
                logger.warning(f"Group {group} has only {len(responses)} responses (min: {self.min_group_size})")
                continue
            
            selected_count = sum(1 for response in responses if selection_criteria(response))
            selection_rate = selected_count / len(responses)
            
            # Apply smoothing to avoid division by zero
            selection_rate = (selected_count + self.smoothing_factor) / (len(responses) + 2 * self.smoothing_factor)
            
            selection_rates[group] = selection_rate
            logger.debug(f"Group {group}: selection_rate={selection_rate:.4f}")
        
        return selection_rates
    
    def compute_disparity_index(self, selection_rates: Dict[str, float]) -> float:
        """
        Compute disparity index from selection rates.
        
        Args:
            selection_rates: Dictionary of group -> selection rate
        
        Returns:
            Disparity index (0 = no disparity, 1 = maximum disparity)
        """
        if len(selection_rates) < 2:
            logger.warning("Need at least 2 groups for disparity calculation")
            return 0.0
        
        rates = np.array(list(selection_rates.values()))
        
        # Avoid division by zero
        min_rate = np.min(rates)
        max_rate = np.max(rates)
        
        if max_rate == 0:
            return 0.0
        
        # Compute minimum impact ratio with smoothing
        min_impact_ratio = (min_rate + self.smoothing_factor) / (max_rate + self.smoothing_factor)
        
        # Disparity index
        disparity = 1.0 - min_impact_ratio
        
        logger.debug(f"Disparity calculation: min_rate={min_rate:.4f}, max_rate={max_rate:.4f}, "
                    f"min_impact_ratio={min_impact_ratio:.4f}, disparity={disparity:.4f}")
        
        return disparity
    
    def compute_overall_disparity(self,
                                category_group_responses: Dict[str, Dict[str, List[Any]]],
                                selection_criteria: callable = None) -> float:
        """
        Compute overall disparity across multiple categories.
        
        Args:
            category_group_responses: Dictionary of category -> (group -> responses)
            selection_criteria: Function to determine selection
        
        Returns:
            Overall disparity score
        """
        if not category_group_responses:
            return 0.0
        
        category_disparities = []
        
        for category, group_responses in category_group_responses.items():
            if len(group_responses) < 2:
                logger.warning(f"Category {category} has insufficient groups for disparity")
                continue
            
            selection_rates = self.compute_selection_rates(group_responses, selection_criteria)
            
            if len(selection_rates) >= 2:
                disparity = self.compute_disparity_index(selection_rates)
                category_disparities.append(disparity)
                
                logger.info(f"Category {category}: disparity={disparity:.4f}")
        
        if not category_disparities:
            return 0.0
        
        # Mean disparity across categories
        overall_disparity = np.mean(category_disparities)
        
        logger.info(f"Overall disparity: {overall_disparity:.4f}")
        
        return overall_disparity
    
    def compute_adverse_impact_ratio(self, 
                                   protected_group_rate: float, 
                                   reference_group_rate: float) -> float:
        """
        Compute Adverse Impact Ratio (AIR) between two groups.
        
        Args:
            protected_group_rate: Selection rate for protected group
            reference_group_rate: Selection rate for reference group
        
        Returns:
            Adverse Impact Ratio
        """
        if reference_group_rate == 0:
            return 1.0  # No adverse impact if reference is 0
        
        air = protected_group_rate / reference_group_rate
        
        # Apply smoothing
        air = (protected_group_rate + self.smoothing_factor) / (reference_group_rate + self.smoothing_factor)
        
        return air