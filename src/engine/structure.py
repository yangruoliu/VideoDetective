"""
Stream A: Structure Engine - Coverage Flow.

Uses spectral clustering to extract video "chapter" anchor points,
ensuring global narrative structure coverage.
"""

from typing import List, Optional

import numpy as np
from sklearn.cluster import SpectralClustering

from ..utils.math_ops import cosine_similarity_matrix, temporal_decay_mask


class StructureEngine:
    """
    Structure Engine for extracting video anchor points.
    
    Uses spectral clustering on affinity matrix (cosine similarity * temporal decay)
    to find medoid frames representing each video "chapter".
    """
    
    def __init__(
        self,
        gamma: float = 1e-3,
        random_state: int = 42
    ):
        """
        Initialize Structure Engine.
        
        Args:
            gamma: Temporal decay rate for affinity matrix. Default 1e-3.
            random_state: Random state for reproducibility.
        """
        self.gamma = gamma
        self.random_state = random_state
    
    def get_anchors(
        self,
        feats: np.ndarray,
        k: int = 8,
        query_features: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Extract anchor frame indices using spectral clustering.
        
        Args:
            feats: Feature matrix [T, D], should be L2-normalized.
            k: Number of anchor points (clusters). Default 8.
            query_features: Optional [1, D] vector. If provided, selects best frame per cluster.
        
        Returns:
            Sorted list of anchor frame indices.
        """
        T = len(feats)
        
        if T == 0:
            return []
        
        if T <= k:
            return list(range(T))
        
        # 1. Build Affinity Matrix
        S = cosine_similarity_matrix(feats, normalize=False)
        M = temporal_decay_mask(T, gamma=self.gamma)
        A = S * M
        A = (A + A.T) / 2
        A = np.clip(A, 0, None)
        
        # 2. Spectral Clustering
        try:
            clustering = SpectralClustering(
                n_clusters=k,
                affinity='precomputed',
                random_state=self.random_state,
                assign_labels='kmeans'
            )
            labels = clustering.fit_predict(A)
        except Exception as e:
            # Fallback to uniform sampling if clustering fails
            return np.linspace(0, T - 1, k).astype(int).tolist()
        
        # 3. Anchor Selection (Medoid vs Query-Aware)
        anchors = []
        for c in range(k):
            cluster_indices = np.where(labels == c)[0]
            if len(cluster_indices) == 0:
                continue
            
            best_idx = None
            
            if query_features is not None:
                # Query-Guided Coverage (Optimization V9)
                # Compute sim(cluster_frames, query)
                # query_features: [1, D] or [D]
                q = query_features.flatten()
                
                # Extract feats for this cluster
                cluster_feats = feats[cluster_indices]
                
                # Dot product (both are normalized) - no clip needed, argmax is scale-invariant
                scores = np.dot(cluster_feats, q)
                max_score_idx = np.argmax(scores)
                best_idx = cluster_indices[max_score_idx]
                
            else:
                # Standard Medoid
                if len(cluster_indices) == 1:
                    best_idx = cluster_indices[0]
                else:
                    best_idx = None
                    min_dist_sum = float('inf')
                    for i in cluster_indices:
                        dist_sum = 0.0
                        for j in cluster_indices:
                            if i != j:
                                dist_sum += (1 - S[i, j])
                        
                        if dist_sum < min_dist_sum:
                            min_dist_sum = dist_sum
                            best_idx = i
            
            if best_idx is not None:
                anchors.append(int(best_idx))
        
        return sorted(anchors)
    
    def get_anchors_with_diversity(
        self,
        feats: np.ndarray,
        k: int = 8,
        diversity_weight: float = 0.3
    ) -> List[int]:
        """
        Extract anchor frames with additional diversity constraint.
        
        This variant penalizes anchors that are too similar to already
        selected anchors, encouraging visual diversity.
        
        Args:
            feats: Feature matrix [T, D].
            k: Number of anchors.
            diversity_weight: Weight for diversity penalty (0-1).
        
        Returns:
            Sorted list of anchor frame indices.
        """
        T = len(feats)
        
        if T == 0:
            return []
        
        if T <= k:
            return list(range(T))
        
        # Compute similarity matrix
        S = cosine_similarity_matrix(feats, normalize=False)
        
        # Greedy selection with diversity
        selected = []
        remaining = set(range(T))
        
        # Start with the frame most representative of the whole video
        scores = np.mean(S, axis=1)
        first_idx = int(np.argmax(scores))
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        while len(selected) < k and len(remaining) > 0:
            # For each remaining frame, compute:
            # score = representativeness - diversity_penalty
            best_idx = None
            best_score = float('-inf')
            
            for idx in remaining:
                # Representativeness: average similarity to all frames
                rep_score = scores[idx]
                
                # Diversity penalty: max similarity to already selected
                if len(selected) > 0:
                    sims_to_selected = [S[idx, s] for s in selected]
                    diversity_penalty = max(sims_to_selected)
                else:
                    diversity_penalty = 0
                
                combined = rep_score - diversity_weight * diversity_penalty
                
                if combined > best_score:
                    best_score = combined
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break
        
        return sorted(selected)
