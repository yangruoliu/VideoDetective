"""
Graph Belief Engine for VideoDetective v4.0.

Implements semantic graph construction and matrix belief propagation.
Replaces the continuous Bayesian belief engine from v3.0.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class GraphBeliefEngine:
    """
    Graph-based belief propagation engine for video understanding.
    
    Maintains a semantic graph where nodes are video chunks and edges
    represent visual similarity + temporal proximity.
    
    Key features:
        - Similarity matrix (W_sim) from cosine similarity
        - Time matrix (W_time) with exponential decay
        - Top-K sparsification to prevent noise diffusion
        - Laplacian-normalized propagation
        - Multi-channel priors for each option (A/B/C/D)
    """
    
    def __init__(
        self,
        node_feats: np.ndarray,
        chunks: List[Tuple[int, int]],
        alpha: float = 0.7,
        tau: float = 10.0,
        top_k: int = 5
    ):
        """
        Initialize the graph belief engine.
        
        Args:
            node_feats: Mean-pooled chunk features [K, D], L2-normalized.
            chunks: List of (start_frame, end_frame) for each chunk.
            alpha: Weight for similarity vs time in affinity matrix.
            tau: Time decay constant for temporal affinity.
            top_k: Keep only top-K neighbors per node (sparsification).
        """
        self.node_feats = node_feats
        self.chunks = chunks
        self.K = len(chunks)  # Number of nodes
        self.alpha = alpha
        self.tau = tau
        self.top_k = top_k
        
        if self.K == 0:
            self.W = np.array([[]])
            self.W_norm = np.array([[]])
            self.Y = np.array([])
            self.F = np.array([])
            self.visited = np.array([], dtype=bool)
            self.channel_scores = {}
            return
        
        # Build affinity matrix
        self._build_affinity_matrix()
        
        # Initialize belief state
        self.Y = np.zeros(self.K)  # Observations (VLM scores)
        self.F = np.zeros(self.K)  # Propagated beliefs
        self.visited = np.zeros(self.K, dtype=bool)
        
        # Multi-channel scores (per option)
        self.channel_scores: Dict[str, np.ndarray] = {}
        
        # V4.3: Semantic captions for text-based retrieval
        self.semantic_captions: Dict[int, str] = {}  # node_idx -> event description
        
        # V4.3: Semantic channel scores (per option, from semantic matching)
        self.semantic_channel_scores: Dict[str, np.ndarray] = {}
    
    def _build_affinity_matrix(self):
        """Build the affinity matrix W = α*W_sim + (1-α)*W_time."""
        K = self.K
        
        # 1. Similarity matrix (cosine similarity)
        # W_sim[i,j] = node_feats[i] @ node_feats[j]
        W_sim = self.node_feats @ self.node_feats.T  # [K, K]
        
        # Normalize W_sim to [0, 1] to match W_time scale
        W_sim = np.clip(W_sim, 0.0, 1.0)
        
        # 2. Time matrix (exponential decay based on chunk center distance)
        chunk_centers = np.array([(s + e) / 2 for s, e in self.chunks])
        time_diff = np.abs(chunk_centers[:, None] - chunk_centers[None, :])
        W_time = np.exp(-time_diff / self.tau)  # [K, K], already in [0, 1]
        
        # 3. Fuse affinity matrix (both now in [0, 1])
        W = self.alpha * W_sim + (1 - self.alpha) * W_time
        
        # 4. Zero-out self-loops
        np.fill_diagonal(W, 0)
        
        # 5. Top-K sparsification (per row)
        W_sparse = np.zeros_like(W)
        for i in range(K):
            row = W[i]
            if len(row) <= self.top_k:
                W_sparse[i] = row
            else:
                # Keep top-K values
                top_k_idx = np.argsort(row)[-self.top_k:]
                W_sparse[i, top_k_idx] = row[top_k_idx]
        
        # Make symmetric
        W_sparse = (W_sparse + W_sparse.T) / 2
        
        self.W = W_sparse
        
        # 6. Laplacian normalization: W_norm = D^(-1/2) @ W @ D^(-1/2)
        D = np.sum(W_sparse, axis=1)
        D_inv_sqrt = np.zeros_like(D)
        nonzero_mask = D > 1e-8
        D_inv_sqrt[nonzero_mask] = 1.0 / np.sqrt(D[nonzero_mask])
        D_inv_sqrt_mat = np.diag(D_inv_sqrt)
        
        self.W_norm = D_inv_sqrt_mat @ W_sparse @ D_inv_sqrt_mat
    
    def init_multi_channel_prior(
        self,
        option_keywords_map: Dict[str, List[str]],
        encoder,
        query_keywords: List[str] = None  # V4.3: Add query keywords to global prior
    ):
        """
        Initialize multi-channel priors for each option.
        
        Args:
            option_keywords_map: {"A": [kw1, kw2], "B": [...], ...}
            encoder: VideoEncoder instance for encoding keywords.
            query_keywords: Keywords from the question itself.
        """
        if self.K == 0:
            return
        
        # Compute prior for each option
        for opt_id, keywords in option_keywords_map.items():
            if not keywords:
                self.channel_scores[opt_id] = np.zeros(self.K)
                continue
            
            # Encode keywords
            text_feats = encoder.encode_text(keywords)  # [len(kw), D]
            
            # Compute max similarity for each node
            sim_matrix = text_feats @ self.node_feats.T  # [len(kw), K]
            P_opt = np.max(sim_matrix, axis=0)  # [K] - max over keywords
            
            # Normalize to [0, 1]
            P_opt = np.clip(P_opt, 0.0, 1.0)
            
            self.channel_scores[opt_id] = P_opt
        
        # V4.3: Compute global prior from ALL keywords (option + query)
        all_kw: List[str] = []
        
        # Add query keywords first (they are important!)
        if query_keywords:
            all_kw.extend(query_keywords)
        
        # Add option keywords
        for kws in option_keywords_map.values():
            all_kw.extend(kws or [])
        
        # Deduplicate preserve order
        seen = set()
        all_kw = [k for k in all_kw if k and not (k in seen or seen.add(k))]

        if all_kw:
            text_feats = encoder.encode_text(all_kw)
            sim_matrix = text_feats @ self.node_feats.T
            global_prior = np.max(sim_matrix, axis=0)
            
            # Normalize to [0, 1]
            global_prior = np.clip(global_prior, 0.0, 1.0)
            
            self.Y = global_prior.copy()
        
        # Run initial propagation
        self.propagate()
    
    def propagate(self, iterations: int = 7, alpha: float = 0.6):
        """
        Matrix belief propagation.
        
        Formula: F = α * W_norm @ F + (1-α) * Y
        
        Args:
            iterations: Number of propagation iterations.
            alpha: Propagation vs observation weight (60% neighbor propagation, 40% self observation).
        """
        if self.K == 0:
            return
        
        # Initialize F from Y if not set
        if np.sum(np.abs(self.F)) < 1e-8:
            self.F = self.Y.copy()
        
        for _ in range(iterations):
            self.F = alpha * (self.W_norm @ self.F) + (1 - alpha) * self.Y
        
        # Note: F is NOT clipped here to allow belief accumulation across updates
        # F values can slightly exceed [0,1] due to propagation, which is intentional
    
    def update(self, node_idx: int, score: float):
        """
        Update belief after observing a node.
        
        Args:
            node_idx: Index of observed node.
            score: VLM relevance score (0.0-1.0).
        """
        if node_idx < 0 or node_idx >= self.K:
            return
        
        self.Y[node_idx] = score
        self.visited[node_idx] = True
        self.propagate()
    
    def search_by_text(
        self,
        keywords: List[str],
        encoder,
        exclude_visited: bool = True
    ) -> int:
        """
        Find the best matching unvisited node for given keywords.
        
        Args:
            keywords: List of keywords to search for.
            encoder: VideoEncoder instance.
            exclude_visited: If True, exclude already visited nodes.
        
        Returns:
            Index of best matching node.
        """
        if self.K == 0 or not keywords:
            return 0
        
        text_feats = encoder.encode_text(keywords)
        sim_matrix = text_feats @ self.node_feats.T  # [len(kw), K]
        scores = np.max(sim_matrix, axis=0)  # [K]
        
        # Note: No normalization needed here - argmax is scale-invariant
        
        if exclude_visited:
            scores = scores * (1 - self.visited.astype(float))
        
        return int(np.argmax(scores))
    
    def get_candidates_for_option(
        self,
        opt_id: str,
        exclude_visited: bool = True
    ) -> List[int]:
        """
        Get all candidate nodes for a specific option, sorted by score.
        
        Args:
            opt_id: Option ID ("A", "B", "C", "D").
            exclude_visited: If True, exclude already visited nodes.
            
        Returns:
            List of node indices sorted by descending score.
        """
        if self.K == 0 or opt_id not in self.channel_scores:
            return []
            
        scores = self.channel_scores[opt_id]
        if exclude_visited:
            scores = scores * (1 - self.visited.astype(float))
            
        # Get indices sorted by score
        sorted_indices = np.argsort(-scores)
        
        # Filter out zero scores if needed, but for now just return all unvisited
        candidates = []
        for idx in sorted_indices:
            idx = int(idx)
            if scores[idx] > 0.0: # Only return nodes with some relevance
                candidates.append(idx)
                
        return candidates

    def get_option_anchors(self, exclude_visited: bool = True) -> Dict[str, int]:
        """
        Get the best node for each option based on channel scores.
        
        Ensures every option (A/B/C/D) gets a unique anchor, even if they share
        the same top-1 node or have score=0. Falls back to top-2, top-3, etc.
        
        Args:
            exclude_visited: If True, exclude already visited nodes.
        
        Returns:
            Dict mapping option (A/B/C/D) to node index.
        """
        if self.K == 0:
            return {}
        
        anchors = {}
        used_nodes = set()
        
        # Sort options by their max score (descending) to give priority to high-confidence options
        opt_scores = []
        for opt_id, scores in self.channel_scores.items():
            max_score = np.max(scores) if len(scores) > 0 else 0
            opt_scores.append((opt_id, max_score, scores))
        opt_scores.sort(key=lambda x: -x[1])
        
        for opt_id, _, scores in opt_scores:
            if exclude_visited:
                masked_scores = scores * (1 - self.visited.astype(float))
            else:
                masked_scores = scores.copy()
            
            # Get sorted indices by score (descending)
            sorted_indices = np.argsort(-masked_scores)
            
            # Find the first node not already used by another option
            found = False
            for idx in sorted_indices:
                idx = int(idx)
                if idx not in used_nodes:
                    anchors[opt_id] = idx
                    used_nodes.add(idx)
                    found = True
                    break
            
            # Fallback: if all nodes are used, just pick the top one anyway
            if not found:
                anchors[opt_id] = int(sorted_indices[0])
        
        return anchors
    
    def graph_nms(
        self,
        num_select: int = 8,
        suppression_factor: float = 0.0
    ) -> List[int]:
        """
        Graph-based Non-Maximum Suppression for final node selection.
        
        Iteratively selects highest-belief nodes and suppresses their neighbors.
        
        Args:
            num_select: Number of nodes to select.
            suppression_factor: Factor to multiply neighbor beliefs (0 = full suppression).
        
        Returns:
            List of selected node indices.
        """
        if self.K == 0:
            return []
        
        selected = []
        F_work = self.F.copy()
        
        for _ in range(min(num_select, self.K)):
            # Select highest belief node
            idx = int(np.argmax(F_work))
            
            if F_work[idx] <= 0:
                break
            
            selected.append(idx)
            
            # Suppress this node and its neighbors
            F_work[idx] = -np.inf
            
            # Suppress neighbors (those with high affinity)
            neighbors = self.W[idx] > 0
            F_work[neighbors] *= suppression_factor
        
        return selected
    
    def get_frames_from_chunks(
        self,
        chunk_indices: List[int],
        frames_per_chunk: int = 4,
        frame_scores: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Sample frame indices from selected chunks.
        
        Args:
            chunk_indices: List of chunk indices to sample from.
            frames_per_chunk: Number of frames to sample from each chunk.
            frame_scores: Optional per-frame relevance scores [T].
        
        Returns:
            List of frame indices, sorted chronologically.
        """
        frame_indices = []
        
        for chunk_idx in chunk_indices:
            if chunk_idx < 0 or chunk_idx >= len(self.chunks):
                continue
            
            start, end = self.chunks[chunk_idx]
            chunk_len = end - start + 1
            
            use_scores = (
                frame_scores is not None
                and hasattr(frame_scores, "__len__")
                and len(frame_scores) > end
            )
            if use_scores:
                candidates = list(range(start, end + 1))
                if chunk_len <= frames_per_chunk:
                    chosen = candidates
                else:
                    ranked = sorted(candidates, key=lambda i: frame_scores[i], reverse=True)
                    chosen = sorted(ranked[:frames_per_chunk])
                frame_indices.extend(chosen)
            else:
                if chunk_len <= frames_per_chunk:
                    # Use all frames
                    frame_indices.extend(range(start, end + 1))
                else:
                    # Uniformly sample
                    step = chunk_len / frames_per_chunk
                    for i in range(frames_per_chunk):
                        frame_idx = int(start + i * step)
                        frame_indices.append(min(frame_idx, end))
        
        return sorted(set(frame_indices))
    
    # ===== V4.3: Semantic Recall Methods =====
    
    def set_semantic_captions(self, captions: Dict[int, str]):
        """
        Set semantic captions for nodes from VideoSummarizer.
        
        Args:
            captions: Dict mapping node_idx to event description.
        """
        self.semantic_captions = captions
    
    def compute_semantic_scores(
        self,
        query: str,
        encoder=None
    ) -> np.ndarray:
        """
        Compute semantic matching scores between query and node captions.
        
        Uses text embedding similarity for efficiency.
        
        Args:
            query: Semantic query string.
            encoder: VideoEncoder with encode_text method.
        
        Returns:
            Semantic scores array [K].
        """
        if self.K == 0 or not query or encoder is None:
            return np.zeros(self.K)
        
        # Build caption list aligned with node indices
        captions = []
        for i in range(self.K):
            cap = self.semantic_captions.get(i, "")
            captions.append(cap if cap else "")
        
        # Skip if no captions
        if not any(captions):
            return np.zeros(self.K)
        
        try:
            # Encode query
            query_feat = encoder.encode_text([query])  # [1, D]
            
            # Encode captions (replace empty with placeholder)
            captions_for_encode = [c if c else "empty scene" for c in captions]
            caption_feats = encoder.encode_text(captions_for_encode)  # [K, D]
            
            # Cosine similarity
            scores = np.dot(query_feat, caption_feats.T)[0]  # [K]
            
            # Zero out nodes without captions
            for i in range(self.K):
                if not self.semantic_captions.get(i):
                    scores[i] = 0.0
            
            # Normalize to 0-1
            min_s, max_s = scores.min(), scores.max()
            if max_s > min_s:
                scores = (scores - min_s) / (max_s - min_s)
            else:
                # All scores are the same, clip to ensure [0, 1]
                scores = np.clip(scores, 0.0, 1.0)
            
            return scores
        
        except Exception as e:
            print(f"[GraphBeliefEngine] Semantic score computation failed: {e}")
            return np.zeros(self.K)
    
    def init_semantic_channel_prior(
        self,
        semantic_queries: Dict[str, str],
        encoder
    ):
        """
        Initialize semantic channel scores for each option.
        
        Args:
            semantic_queries: {"A": "query for A", "B": "query for B", ...}
            encoder: VideoEncoder instance.
        """
        if self.K == 0:
            return
        
        for opt_id, query in semantic_queries.items():
            if query:
                scores = self.compute_semantic_scores(query, encoder)
                self.semantic_channel_scores[opt_id] = scores
            else:
                self.semantic_channel_scores[opt_id] = np.zeros(self.K)
    
    def get_fused_option_anchors(
        self,
        alpha: float = 0.6,
        exclude_visited: bool = True
    ) -> Dict[str, int]:
        """
        Get best node for each option using fused visual + semantic scores.
        
        Args:
            alpha: Weight for visual scores (1-alpha for semantic).
            exclude_visited: If True, exclude already visited nodes.
        
        Returns:
            Dict mapping option (A/B/C/D) to node index.
        """
        if self.K == 0:
            return {}
        
        anchors = {}
        used_nodes = set()
        
        # Collect all options with their fused scores
        opt_fused = []
        for opt_id in self.channel_scores.keys():
            visual_scores = self.channel_scores.get(opt_id, np.zeros(self.K))
            semantic_scores = self.semantic_channel_scores.get(opt_id, np.zeros(self.K))
            
            # Fuse scores
            fused = alpha * visual_scores + (1 - alpha) * semantic_scores
            
            if exclude_visited:
                fused = fused * (1 - self.visited.astype(float))
            
            max_score = np.max(fused) if len(fused) > 0 else 0
            opt_fused.append((opt_id, max_score, fused))
        
        # Sort by max fused score (descending) for priority
        opt_fused.sort(key=lambda x: -x[1])
        
        for opt_id, _, fused in opt_fused:
            sorted_indices = np.argsort(-fused)
            
            # Find first unused node
            found = False
            for idx in sorted_indices:
                idx = int(idx)
                if idx not in used_nodes:
                    anchors[opt_id] = idx
                    used_nodes.add(idx)
                    found = True
                    break
            
            if not found:
                anchors[opt_id] = int(sorted_indices[0])
        
        return anchors
    
    def get_fused_candidates_for_option(
        self,
        opt_id: str,
        alpha: float = 0.6,
        exclude_visited: bool = True
    ) -> List[int]:
        """
        Get all candidate nodes for an option sorted by fused score.
        
        Args:
            opt_id: Option ID ("A", "B", "C", "D").
            alpha: Weight for visual scores.
            exclude_visited: If True, exclude already visited nodes.
        
        Returns:
            List of node indices sorted by descending fused score.
        """
        if self.K == 0:
            return []
        
        visual_scores = self.channel_scores.get(opt_id, np.zeros(self.K))
        semantic_scores = self.semantic_channel_scores.get(opt_id, np.zeros(self.K))
        
        fused = alpha * visual_scores + (1 - alpha) * semantic_scores
        
        if exclude_visited:
            fused = fused * (1 - self.visited.astype(float))
        
        sorted_indices = np.argsort(-fused)
        
        candidates = []
        for idx in sorted_indices:
            idx = int(idx)
            if fused[idx] > 0.0:
                candidates.append(idx)
        
        return candidates
