# src/baseline_manager.py
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import os
from .data_loader import DataLoader

logger = logging.getLogger("bias_detection.baseline_manager")

class BaselineManager:
    """Manages baseline calibration data and reference distributions."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_loader = DataLoader(data_dir)
        self.baselines_cache = {}
    
    def load_baseline_responses(self, category: str) -> Dict[str, Any]:
        """
        Load baseline responses for a category.
        
        Args:
            category: Bias category
        
        Returns:
            Baseline responses data
        """
        if category in self.baselines_cache:
            return self.baselines_cache[category]
        
        baseline_file = os.path.join(self.data_loader.baselines_dir, f"{category}_baseline.json")
        
        if not os.path.exists(baseline_file):
            logger.warning(f"Baseline file not found: {baseline_file}")
            return {'texts': [], 'scores': []}
        
        try:
            with open(baseline_file, 'r', encoding='utf-8') as f:
                baseline_data = json.load(f)
            
            self.baselines_cache[category] = baseline_data
            logger.info(f"Loaded baseline for {category} with {len(baseline_data.get('texts', []))} texts")
            return baseline_data
        
        except Exception as e:
            logger.error(f"Error loading baseline for {category}: {e}")
            return {'texts': [], 'scores': []}
    
    def compute_baseline_statistics(self, category: str) -> Dict[str, float]:
        """
        Compute statistics for baseline data.
        
        Args:
            category: Bias category
        
        Returns:
            Baseline statistics
        """
        baseline_data = self.load_baseline_responses(category)
        scores = baseline_data.get('scores', [])
        texts = baseline_data.get('texts', [])
        
        if not scores:
            return {
                'mean': 0.0,
                'std': 0.0,
                'count': 0,
                'text_count': len(texts)
            }
        
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'count': len(scores),
            'text_count': len(texts)
        }
    
    def get_calibration_reference(self, 
                                category: str, 
                                reference_type: str = 'mean') -> float:
        """
        Get calibration reference value for a category.
        
        Args:
            category: Bias category
            reference_type: Type of reference ('mean', 'median', 'zero')
        
        Returns:
            Calibration reference value
        """
        if reference_type == 'zero':
            return 0.0
        
        baseline_stats = self.compute_baseline_statistics(category)
        
        if reference_type == 'mean':
            return baseline_stats['mean']
        elif reference_type == 'median':
            # For now, use mean as proxy for median
            return baseline_stats['mean']
        else:
            raise ValueError(f"Unknown reference type: {reference_type}")
    
    def create_baseline_from_responses(self,
                                     category: str,
                                     responses: List[str],
                                     scores: Optional[List[float]] = None) -> str:
        """
        Create or update baseline from model responses.
        
        Args:
            category: Bias category
            responses: Model responses
            scores: Optional bias scores for responses
        
        Returns:
            Path to saved baseline file
        """
        if scores is None:
            # Default neutral scores
            scores = [0.0] * len(responses)
        
        baseline_data = {
            'category': category,
            'texts': responses,
            'scores': scores,
            'statistics': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'count': len(scores)
            },
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        baseline_file = os.path.join(self.data_loader.baselines_dir, f"{category}_baseline.json")
        
        try:
            with open(baseline_file, 'w', encoding='utf-8') as f:
                json.dump(baseline_data, f, indent=2, ensure_ascii=False)
            
            # Update cache
            self.baselines_cache[category] = baseline_data
            
            logger.info(f"Created/updated baseline for {category} with {len(responses)} responses")
            return baseline_file
        
        except Exception as e:
            logger.error(f"Error creating baseline for {category}: {e}")
            raise
    
    def validate_baseline_coverage(self, 
                                 categories: List[str]) -> Dict[str, bool]:
        """
        Validate that baselines exist for required categories.
        
        Args:
            categories: List of categories to check
        
        Returns:
            Dictionary of category -> has_baseline
        """
        coverage = {}
        
        for category in categories:
            baseline_data = self.load_baseline_responses(category)
            has_baseline = len(baseline_data.get('texts', [])) > 0
            coverage[category] = has_baseline
            
            if not has_baseline:
                logger.warning(f"No baseline found for category: {category}")
        
        return coverage
    
    def get_all_baseline_categories(self) -> List[str]:
        """
        Get list of all available baseline categories.
        
        Returns:
            List of category names
        """
        categories = []
        
        for filename in os.listdir(self.data_loader.baselines_dir):
            if filename.endswith('_baseline.json'):
                category = filename.replace('_baseline.json', '')
                categories.append(category)
        
        return categories