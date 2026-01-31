"""
Stream B: Belief Engine - Relevance Flow.

Implements Bayesian belief optimization for iterative video search.
Maintains a belief curve (mu) that is updated based on VLM observations.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from ..utils.math_ops import gaussian_kernel, softmax_sharpen, find_peaks


@dataclass
class BeliefHistory:
    """Stores the history of belief updates for visualization."""
    steps: List[int] = field(default_factory=list)
    mu_snapshots: List[np.ndarray] = field(default_factory=list)
    sigma_snapshots: List[np.ndarray] = field(default_factory=list)
    observations: List[dict] = field(default_factory=list)


class BeliefEngine:
    """
    Belief Engine for Bayesian search optimization.
    
    Maintains belief distributions (mu, sigma) over the video timeline
    and updates them based on VLM observations.
    
    Attributes:
        mu: Expected relevance at each time point [T].
        sigma: Uncertainty at each time point [T].
    """
    
    def __init__(self):
        """Initialize Belief Engine."""
        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None
        self.T: int = 0
        self.history: BeliefHistory = BeliefHistory()
        self._step_count: int = 0
    
    def init_prior(
        self,
        video_feats: np.ndarray,
        visual_keywords: List[str],
        encoder,
        temperature: float = 0.1
    ) -> np.ndarray:
        """
        Initialize belief prior based on visual keyword similarity.
        
        Args:
            video_feats: Video frame features [T, D], L2-normalized.
            visual_keywords: List of visual keywords for search.
            encoder: VideoEncoder instance for encoding keywords.
            temperature: Temperature for softmax sharpening.
        
        Returns:
            Initialized mu distribution [T].
        """
        self.T = len(video_feats)
        
        if self.T == 0:
            self.mu = np.array([])
            self.sigma = np.array([])
            return self.mu
        
        # Initialize sigma (uncertainty)
        self.sigma = np.ones(self.T)
        
        if len(visual_keywords) == 0:
            # No keywords, use uniform prior
            self.mu = np.ones(self.T) / self.T
            return self.mu
        
        # Encode keywords to text features
        text_feats = encoder.encode_text(visual_keywords)  # [K, D]
        
        if len(text_feats) == 0:
            self.mu = np.ones(self.T) / self.T
            return self.mu
        
        # Compute similarity matrix [K, T]
        similarity_matrix = encoder.compute_similarity(video_feats, text_feats)
        
        # Max pooling across keywords: S_prior[t] = max_k(Sim[k, t])
        S_prior = np.max(similarity_matrix, axis=0)  # [T]
        
        # Apply softmax sharpening
        self.mu = softmax_sharpen(S_prior, temperature=temperature)
        
        # Record initial state
        self._record_snapshot(t_obs=None, relevance=None, offset=None)
        
        return self.mu
    
    def suggest_next_step(
        self,
        beta: float = 1.0,
        visited: Optional[List[int]] = None,
        min_distance: int = 3
    ) -> int:
        """
        Suggest next time point to explore.
        
        Uses Upper Confidence Bound (UCB) strategy:
            score = mu + beta * sigma
        
        Args:
            beta: Exploration-exploitation tradeoff. Higher = more exploration.
            visited: List of already visited time points to avoid.
            min_distance: Minimum distance from visited points.
        
        Returns:
            Suggested time index.
        """
        if self.mu is None or self.T == 0:
            return 0
        
        # Compute acquisition function
        scores = self.mu + beta * self.sigma
        
        # Mask visited regions
        if visited and len(visited) > 0:
            mask = np.ones(self.T, dtype=bool)
            for v in visited:
                start = max(0, v - min_distance)
                end = min(self.T, v + min_distance + 1)
                mask[start:end] = False
            
            # If all masked, just use original scores
            if np.any(mask):
                scores = np.where(mask, scores, -np.inf)
        
        return int(np.argmax(scores))
    
    def update(
        self,
        t_obs: int,
        relevance: float,
        offset: float = 0.0,
        sigma_suppress_width: float = 4.0,
        mu_excite_width: float = 1.5,
        mu_inhibit_width: float = 4.0,
        hit_threshold: float = 0.6
    ) -> None:
        """
        Update belief based on VLM observation.
        
        Args:
            t_obs: Observed time point.
            relevance: VLM confidence score (0.0-1.0).
            offset: Time offset from VLM (in frame indices).
            sigma_suppress_width: Sigma for uncertainty suppression kernel.
            mu_excite_width: Sigma for excitation kernel (on hit).
            mu_inhibit_width: Sigma for inhibition kernel (on miss).
            hit_threshold: Relevance threshold for hit/miss.
        """
        if self.mu is None or self.T == 0:
            return
        
        self._step_count += 1
        
        # 1. Suppress uncertainty around observation point
        k_suppress = gaussian_kernel(
            self.T, center=t_obs, sigma=sigma_suppress_width, height=1.0
        )
        self.sigma = self.sigma * (1 - k_suppress)
        
        # 2. Update expected value (mu)
        if relevance > hit_threshold:
            # HIT: Excite around target
            t_target = t_obs + offset
            t_target = np.clip(t_target, 0, self.T - 1)
            
            k_excite = gaussian_kernel(
                self.T, center=t_target, sigma=mu_excite_width, height=relevance
            )
            self.mu = self.mu + k_excite
        else:
            # MISS: Inhibit around observation point
            k_inhibit = gaussian_kernel(
                self.T, center=t_obs, sigma=mu_inhibit_width, height=0.5
            )
            self.mu = self.mu * (1 - k_inhibit)
        
        # Clip to valid range
        self.mu = np.clip(self.mu, 0, 1)
        
        # Record history
        self._record_snapshot(t_obs, relevance, offset)
    
    def _record_snapshot(
        self,
        t_obs: Optional[int],
        relevance: Optional[float],
        offset: Optional[float]
    ) -> None:
        """Record current state for visualization."""
        self.history.steps.append(self._step_count)
        self.history.mu_snapshots.append(self.mu.copy() if self.mu is not None else np.array([]))
        self.history.sigma_snapshots.append(self.sigma.copy() if self.sigma is not None else np.array([]))
        self.history.observations.append({
            "t_obs": t_obs,
            "relevance": relevance,
            "offset": offset
        })
    
    def get_top_indices(
        self,
        k: int,
        min_distance: int = 1
    ) -> List[int]:
        """
        Get top-k most relevant time indices.
        
        Args:
            k: Number of indices to return.
            min_distance: Minimum distance between selected indices.
        
        Returns:
            List of top-k indices sorted by position.
        """
        if self.mu is None or self.T == 0:
            return []
        
        return find_peaks(self.mu, k=k, min_distance=min_distance).tolist()
    
    def get_confidence(self) -> float:
        """
        Get overall confidence in current belief.
        
        Returns:
            Maximum mu value (peak confidence).
        """
        if self.mu is None or self.T == 0:
            return 0.0
        
        return float(np.max(self.mu))
    
    def should_early_stop(
        self,
        relevance_threshold: float = 0.9,
        confidence_threshold: float = 0.95
    ) -> bool:
        """
        Check if search should terminate early.
        
        Args:
            relevance_threshold: Minimum relevance from last observation.
            confidence_threshold: Minimum peak mu value.
        
        Returns:
            True if early stopping is recommended.
        """
        if self.mu is None or len(self.history.observations) == 0:
            return False
        
        last_obs = self.history.observations[-1]
        last_relevance = last_obs.get("relevance")
        
        if last_relevance is None:
            return False
        
        peak_mu = np.max(self.mu)
        
        return last_relevance > relevance_threshold and peak_mu > confidence_threshold
    
    def get_history_dict(self) -> dict:
        """
        Get history as dictionary for debugging/visualization.
        
        Returns:
            Dictionary containing belief evolution history.
        """
        return {
            "steps": self.history.steps,
            "mu_snapshots": [m.tolist() if len(m) > 0 else [] for m in self.history.mu_snapshots],
            "sigma_snapshots": [s.tolist() if len(s) > 0 else [] for s in self.history.sigma_snapshots],
            "observations": self.history.observations
        }
    
    def reset(self) -> None:
        """Reset engine state."""
        self.mu = None
        self.sigma = None
        self.T = 0
        self.history = BeliefHistory()
        self._step_count = 0
