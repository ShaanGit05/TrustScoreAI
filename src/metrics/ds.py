# src/metrics/ds.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import scipy

logger = logging.getLogger("bias_detection.ds")

class DistributionShiftCalculator:
    """Calculates Distribution Shift (DS) component of UBI."""
    
    def __init__(self, kl_smoothing: float = 1e-8, bins: int = 20):
        self.kl_smoothing = kl_smoothing
        self.bins = bins
    
    def compute_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute KL divergence between two distributions.
        
        Args:
            p: True distribution
            q: Reference distribution
        
        Returns:
            KL divergence
        """
        # Ensure distributions are valid
        p = np.array(p, dtype=float) + self.kl_smoothing
        q = np.array(q, dtype=float) + self.kl_smoothing
        
        # Normalize to probability distributions
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Compute KL divergence
        kl_div = entropy(p, q)
        
        return kl_div
    
    def compute_js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute Jensen-Shannon divergence (symmetric version of KL).
        
        Args:
            p: First distribution
            q: Second distribution
        
        Returns:
            JS divergence
        """
        p = np.array(p, dtype=float) + self.kl_smoothing
        q = np.array(q, dtype=float) + self.kl_smoothing
        
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        m = 0.5 * (p + q)
        
        js_div = 0.5 * (entropy(p, m) + entropy(q, m))
        
        return js_div
    
    def compute_text_distribution_shift(self,
                                      test_responses: List[str],
                                      baseline_responses: List[str],
                                      method: str = 'tfidf') -> float:
        """
        Compute distribution shift between test and baseline text responses.
        
        Args:
            test_responses: LLM responses to test
            baseline_responses: Baseline reference responses
            method: Method for text representation ('tfidf', 'count', 'length')
        
        Returns:
            Distribution shift score
        """
        if not test_responses or not baseline_responses:
            logger.warning("Insufficient data for distribution shift calculation")
            return 0.0
        
        if method == 'length':
            # Use response length distribution
            test_lengths = [len(str(r)) for r in test_responses]
            baseline_lengths = [len(str(r)) for r in baseline_responses]
            
            # Create histograms
            all_lengths = test_lengths + baseline_lengths
            bins = np.linspace(min(all_lengths), max(all_lengths), self.bins + 1)
            
            test_hist, _ = np.histogram(test_lengths, bins=bins, density=True)
            baseline_hist, _ = np.histogram(baseline_lengths, bins=bins, density=True)
            
            shift = self.compute_js_divergence(test_hist, baseline_hist)
            
        elif method in ['tfidf', 'count']:
            # Use text content distribution
            all_texts = test_responses + baseline_responses
            
            if method == 'tfidf':
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            else:
                vectorizer = CountVectorizer(max_features=1000, stop_words='english')
            
            try:
                all_vectors = vectorizer.fit_transform(all_texts).toarray()
                
                test_vectors = all_vectors[:len(test_responses)]
                baseline_vectors = all_vectors[len(test_responses):]
                
                # Compute mean vectors
                test_mean = np.mean(test_vectors, axis=0)
                baseline_mean = np.mean(baseline_vectors, axis=0)
                
                # Compute cosine similarity between distributions
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity([test_mean], [baseline_mean])[0][0]
                
                # Convert similarity to divergence (1 - similarity)
                shift = 1.0 - similarity
                
            except Exception as e:
                logger.warning(f"Text vectorization failed: {e}, falling back to length method")
                return self.compute_text_distribution_shift(test_responses, baseline_responses, 'length')
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.debug(f"Distribution shift ({method}): {shift:.4f}")
        
        return shift
    
    def compute_numeric_distribution_shift(self,
                                         test_scores: List[float],
                                         baseline_scores: List[float]) -> float:
        """
        Compute distribution shift between test and baseline numeric scores.
        
        Args:
            test_scores: Test distribution scores
            baseline_scores: Baseline distribution scores
        
        Returns:
            Distribution shift score
        """
        if not test_scores or not baseline_scores:
            return 0.0
        
        # Create histograms for comparison
        all_scores = test_scores + baseline_scores
        bins = np.linspace(min(all_scores), max(all_scores), self.bins + 1)
        
        test_hist, _ = np.histogram(test_scores, bins=bins, density=True)
        baseline_hist, _ = np.histogram(baseline_scores, bins=bins, density=True)
        
        # Compute JS divergence
        shift = self.compute_js_divergence(test_hist, baseline_hist)
        
        logger.debug(f"Numeric distribution shift: {shift:.4f}")
        
        return shift
    
    def compute_overall_distribution_shift(self,
                                         category_responses: Dict[str, List[str]],
                                         baseline_responses: Dict[str, List[str]]) -> float:
        """
        Compute overall distribution shift across categories.
        
        Args:
            category_responses: Test responses by category
            baseline_responses: Baseline responses by category
        
        Returns:
            Overall distribution shift score
        """
        if not category_responses:
            return 0.0
        
        category_shifts = []
        
        for category, test_responses in category_responses.items():
            baseline_category_responses = baseline_responses.get(category, [])
            
            if not baseline_category_responses:
                logger.warning(f"No baseline responses for category {category}")
                continue
            
            shift = self.compute_text_distribution_shift(test_responses, baseline_category_responses)
            category_shifts.append(shift)
            
            logger.info(f"Category {category}: distribution_shift={shift:.4f}")
        
        if not category_shifts:
            return 0.0
        
        # Mean shift across categories
        overall_shift = np.mean(category_shifts)
        
        logger.info(f"Overall distribution shift: {overall_shift:.4f}")
        
        return overall_shift