"""
Video Encoder using SigLIP/CLIP.

Extracts visual features from video frames for semantic search.
"""

import warnings
import os
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import torch
from PIL import Image

# Import based on available model
try:
    from transformers import (
        AutoProcessor,
        AutoModel,
        CLIPProcessor,
        CLIPModel,
    )
except ImportError:
    raise ImportError("transformers is required. Install with: pip install transformers")

from ..utils.video_io import load_video, compute_sample_indices, LazyVideoFrames, LazyImagePaths


class VideoEncoder:
    """
    Video encoder using SigLIP or CLIP vision models.
    
    Extracts L2-normalized visual features from video frames.
    
    Strategy:
        1. Try to load SigLIP from HuggingFace (with HF mirror)
        2. If SigLIP fails, fallback to local CLIP model
    """
    
    def __init__(
        self,
        siglip_model_id: str = "google/siglip-so400m-patch14-384",
        clip_local_path: str = "",
        cache_dir: str = "",
        device: Optional[str] = None
    ):
        """
        Initialize the video encoder.
        
        Args:
            siglip_model_id: HuggingFace model ID for SigLIP.
            clip_local_path: Local path to CLIP model (fallback).
            cache_dir: HuggingFace cache directory.
            device: Device to use ('cuda' or 'cpu'). Auto-detect if None.
        """
        self.siglip_model_id = siglip_model_id
        self.clip_local_path = clip_local_path
        self.cache_dir = cache_dir
        
        if device is None:
            # Allow forcing encoder device via env var to avoid CUDA OOM on busy GPUs.
            # Examples:
            #   ENCODER_DEVICE=cpu
            #   ENCODER_DEVICE=cuda
            env_dev = (os.getenv("ENCODER_DEVICE") or os.getenv("VIDEODETECTIVE_ENCODER_DEVICE") or "").strip().lower()
            if env_dev in ("cpu", "cuda"):
                self.device = env_dev
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.processor = None
        self.is_siglip = False
        
        self._load_model()
    
    def _is_cuda_oom(self, e: Exception) -> bool:
        """Best-effort detection of CUDA OOM across different torch error types."""
        msg = str(e).lower()
        if "out of memory" in msg or "cudaerrormemoryallocation" in msg:
            return True
        # torch.cuda.OutOfMemoryError exists in newer torch, but may not always be raised
        try:
            if isinstance(e, torch.cuda.OutOfMemoryError):  # type: ignore[attr-defined]
                return True
        except Exception:
            pass
        return False

    def _fallback_to_cpu(self, reason: str):
        """Switch encoder to CPU mode (used when GPU is too busy / OOM)."""
        if self.device != "cpu":
            print(f"[VideoEncoder] WARNING: CUDA OOM while loading encoder ({reason}). Falling back to CPU. This will be slower.")
        self.device = "cpu"
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    
    def _load_model(self):
        """Load the encoder model with fallback strategy."""
        # Try SigLIP first
        try:
            print(f"Attempting to load SigLIP model: {self.siglip_model_id}")
            self.processor = AutoProcessor.from_pretrained(
                self.siglip_model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.siglip_model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            try:
                self.model = self.model.to(self.device)
            except Exception as e:
                if self.device == "cuda" and self._is_cuda_oom(e):
                    self._fallback_to_cpu("SigLIP.to(cuda)")
                    self.model = self.model.to(self.device)
                else:
                    raise
            self.model.eval()
            self.is_siglip = True
            print("Successfully loaded SigLIP model")
            return
        except Exception as e:
            warnings.warn(f"Failed to load SigLIP model: {e}. Falling back to CLIP.")
        
        # Fallback to local CLIP
        try:
            print(f"Attempting to load local CLIP model: {self.clip_local_path}")
            
            def _is_valid_hf_snapshot_dir(p: Path) -> bool:
                """Heuristic: a usable CLIP snapshot should include processor + config files."""
                if not p.exists() or not p.is_dir():
                    return False
                required_any = ["preprocessor_config.json", "processor_config.json"]
                required_all = ["config.json"]
                has_any = any((p / f).exists() for f in required_any)
                has_all = all((p / f).exists() for f in required_all)
                return has_any and has_all

            def _iter_snapshot_candidates(snapshots_dir: Path):
                """Yield snapshot dirs in a stable order, preferring valid ones."""
                if not snapshots_dir.exists():
                    return
                subdirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                # Prefer directories that look complete; keep deterministic ordering
                valid = sorted([d for d in subdirs if _is_valid_hf_snapshot_dir(d)], key=lambda x: x.name)
                invalid = sorted([d for d in subdirs if not _is_valid_hf_snapshot_dir(d)], key=lambda x: x.name)
                for d in valid + invalid:
                    yield d

            # Check if local path exists
            clip_path = Path(self.clip_local_path)
            if not clip_path.exists():
                # Try to find the actual model directory
                # The path might be in HuggingFace cache format
                possible_paths = [
                    clip_path,
                    clip_path / "snapshots",
                    Path(self.cache_dir) / "models--openai--clip-vit-base-patch32",
                ]
                
                model_path: Optional[str] = None
                for p in possible_paths:
                    if p.exists():
                        # If it's a HF cache directory, find the snapshot
                        snapshots_dir = p / "snapshots" if (p / "snapshots").exists() else p
                        if snapshots_dir.exists():
                            # Prefer a complete snapshot directory if available
                            for subdir in _iter_snapshot_candidates(snapshots_dir):
                                model_path = str(subdir)
                                break
                        if model_path:
                            break
                
                if model_path is None:
                    # Try loading from HuggingFace ID as last resort
                    model_path = "openai/clip-vit-base-patch32"
            else:
                # Check if it's a HF cache format directory
                snapshots_dir = clip_path / "snapshots"
                if snapshots_dir.exists():
                    model_path = None
                    for subdir in _iter_snapshot_candidates(snapshots_dir):
                        model_path = str(subdir)
                        break
                    if model_path is None:
                        model_path = str(clip_path)
                else:
                    model_path = str(clip_path)

            # If user points to a specific snapshot dir but it is incomplete, auto-switch to a valid sibling snapshot.
            try:
                mp0 = Path(model_path)
                if mp0.exists() and mp0.is_dir() and mp0.parent.name == "snapshots" and not _is_valid_hf_snapshot_dir(mp0):
                    print(f"[VideoEncoder] WARNING: CLIP snapshot looks incomplete: {str(mp0)}")
                    sib_dir = mp0.parent
                    for d in _iter_snapshot_candidates(sib_dir):
                        if _is_valid_hf_snapshot_dir(d):
                            print(f"[VideoEncoder] Auto-switching to valid CLIP snapshot: {str(d)}")
                            model_path = str(d)
                            break
            except Exception:
                pass

            # If snapshots exist, try candidates until one loads successfully.
            # This handles partially-downloaded/invalid snapshots.
            tried = []
            candidate_paths: List[str] = []
            try:
                # If model_path points to a snapshots dir (common case), also try siblings.
                mp = Path(model_path)
                if mp.exists() and mp.is_dir() and mp.parent.name == "snapshots":
                    # Prefer valid snapshots only if any exist; otherwise include invalid as last resort.
                    sibs = list(_iter_snapshot_candidates(mp.parent))
                    valids = [d for d in sibs if _is_valid_hf_snapshot_dir(d)]
                    use = valids if valids else sibs
                    for d in use:
                        candidate_paths.append(str(d))
                candidate_paths.append(model_path)
            except Exception:
                candidate_paths = [model_path]

            last_err: Optional[Exception] = None
            errors: List[Tuple[str, str]] = []
            for cand in candidate_paths:
                if cand in tried:
                    continue
                tried.append(cand)
                try:
                    self.processor = CLIPProcessor.from_pretrained(
                        cand,
                        cache_dir=self.cache_dir
                    )
                    model = CLIPModel.from_pretrained(
                        cand,
                        cache_dir=self.cache_dir
                    )
                    try:
                        model = model.to(self.device)
                    except Exception as e:
                        if self.device == "cuda" and self._is_cuda_oom(e):
                            self._fallback_to_cpu("CLIP.to(cuda)")
                            model = model.to(self.device)
                        else:
                            raise
                    self.model = model
                    self.model.eval()
                    self.is_siglip = False
                    print(f"Successfully loaded CLIP model from: {cand} (device={self.device})")
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    errors.append((cand, str(e)))
                    continue

            if last_err is not None:
                # Provide actionable diagnostics to clean broken cache / pick correct snapshot.
                msg = "Failed to load CLIP from all candidate paths:\n" + "\n".join(
                    [f"- {p}: {err}" for (p, err) in errors[-8:]]  # keep tail to limit size
                )
                raise RuntimeError(msg) from last_err
        except Exception as e:
            raise RuntimeError(f"Failed to load both SigLIP and CLIP models: {e}")
    
    def encode(
        self,
        video_path: str,
        fps: float = 1.0,
        max_frames: Optional[int] = None,
        use_cache: bool = False  # Default to False for reproducibility
    ) -> Tuple[np.ndarray, List[Image.Image]]:
        """
        Encode video frames to feature vectors.
        
        Args:
            video_path: Path to video file.
            fps: Target FPS for frame sampling.
            max_frames: Maximum number of frames to extract.
            use_cache: If True, cache embeddings to disk for reuse. Default False.
        
        Returns:
            Tuple of:
                - feats: L2-normalized feature array [T, D]
                - frames: List of PIL.Image frames
        """
        # Decode frames from video
        print(f"[VideoEncoder] Extracting frames from video (fps={fps})...")
        frames, _timestamps = load_video(video_path, fps=fps, max_frames=max_frames)
        if len(frames) == 0:
            return np.array([]), []
        
        print(f"[VideoEncoder] Encoding {len(frames)} frames...")
        
        # Extract features
        feats = self._encode_images(frames)
        
        print(f"[VideoEncoder] Encoded {len(frames)} frames -> features shape {feats.shape}")
        
        return feats, frames
    
    def encode_frames(
        self,
        frames: List[Image.Image]
    ) -> np.ndarray:
        """
        Encode a list of frames to feature vectors.
        
        Args:
            frames: List of PIL.Image frames.
        
        Returns:
            L2-normalized feature array [T, D].
        """
        if len(frames) == 0:
            return np.array([])
        
        return self._encode_images(frames)
    
    def _encode_images(
        self,
        images: List[Image.Image],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode images to feature vectors with batching.
        
        Args:
            images: List of PIL.Image.
            batch_size: Batch size for processing.
        
        Returns:
            L2-normalized feature array [N, D].
        """
        all_feats = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                
                # Process images
                inputs = self.processor(
                    images=batch_images,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Extract features
                if self.is_siglip:
                    outputs = self.model.get_image_features(**inputs)
                else:
                    # CLIP
                    outputs = self.model.get_image_features(
                        pixel_values=inputs.get("pixel_values")
                    )
                
                all_feats.append(outputs.cpu().numpy())
        
        # Concatenate all batches
        feats = np.concatenate(all_feats, axis=0)
        
        # L2 normalize
        norms = np.linalg.norm(feats, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        feats = feats / norms
        
        return feats
    
    def encode_text(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """
        Encode text queries to feature vectors.
        
        Args:
            texts: List of text strings.
        
        Returns:
            L2-normalized text feature array [N, D].
        """
        if len(texts) == 0:
            return np.array([])
        
        with torch.no_grad():
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64  # Reduced to match model's max_position_embeddings
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if self.is_siglip:
                outputs = self.model.get_text_features(**inputs)
            else:
                # CLIP
                outputs = self.model.get_text_features(
                    input_ids=inputs.get("input_ids"),
                    attention_mask=inputs.get("attention_mask")
                )
            
            feats = outputs.cpu().numpy()
        
        # L2 normalize
        norms = np.linalg.norm(feats, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        feats = feats / norms
        
        return feats
    
    def compute_similarity(
        self,
        visual_feats: np.ndarray,
        text_feats: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between visual and text features.
        
        Args:
            visual_feats: Visual features [T, D].
            text_feats: Text features [K, D].
        
        Returns:
            Similarity matrix [K, T].
        """
        # Both should already be L2 normalized
        return text_feats @ visual_feats.T
    
    def chunk_video(
        self,
        feats: np.ndarray,
        sim_threshold: float = 0.85,
        min_len: int = 4
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Segment video into semantic chunks based on visual similarity (v4.0).
        
        This method divides the video into semantically coherent segments by detecting
        scene transitions (low similarity between adjacent frames).
        
        Args:
            feats: L2-normalized frame features [T, D].
            sim_threshold: Similarity threshold for cutting. Below this = new segment.
            min_len: Minimum chunk length. Shorter chunks are merged with neighbors.
        
        Returns:
            Tuple of:
                - chunks: List of (start_frame, end_frame) tuples (inclusive).
                - node_feats: Mean-pooled feature for each chunk [K, D].
        
        Algorithm:
            1. Compute cosine similarity between adjacent frames.
            2. Cut where similarity < threshold.
            3. Merge chunks shorter than min_len to more similar neighbor.
            4. Mean-pool features within each chunk.
        """
        T = feats.shape[0]
        if T == 0:
            return [], np.array([])
        
        if T == 1:
            return [(0, 0)], feats.copy()
        
        # Step 1: Compute adjacent frame similarities
        # sim[i] = cosine_similarity(feats[i], feats[i+1])
        adj_sim = np.sum(feats[:-1] * feats[1:], axis=1)  # [T-1]
        
        # Step 2: Find cut points (where similarity drops below threshold)
        cuts = [0]  # Start of first chunk
        for i in range(len(adj_sim)):
            if adj_sim[i] < sim_threshold:
                cuts.append(i + 1)  # Start of new chunk
        cuts.append(T)  # End marker
        
        # Step 3: Build initial chunks
        chunks = []
        for i in range(len(cuts) - 1):
            start = cuts[i]
            end = cuts[i + 1] - 1  # Inclusive end
            if start <= end:
                chunks.append((start, end))
        
        # Step 4: Merge short chunks
        def get_chunk_feat(chunk: Tuple[int, int]) -> np.ndarray:
            """Mean-pool features for a chunk."""
            start, end = chunk
            return feats[start:end + 1].mean(axis=0)
        
        merged = True
        while merged and len(chunks) > 1:
            merged = False
            new_chunks = []
            i = 0
            while i < len(chunks):
                chunk = chunks[i]
                chunk_len = chunk[1] - chunk[0] + 1
                
                if chunk_len < min_len:
                    # Need to merge with a neighbor
                    chunk_feat = get_chunk_feat(chunk)
                    
                    left_sim = -1.0
                    right_sim = -1.0
                    
                    # Check left neighbor
                    if len(new_chunks) > 0:
                        left_feat = get_chunk_feat(new_chunks[-1])
                        left_sim = float(np.dot(chunk_feat, left_feat))
                    
                    # Check right neighbor
                    if i + 1 < len(chunks):
                        right_feat = get_chunk_feat(chunks[i + 1])
                        right_sim = float(np.dot(chunk_feat, right_feat))
                    
                    if left_sim >= right_sim and left_sim > 0:
                        # Merge with left neighbor
                        prev = new_chunks.pop()
                        new_chunks.append((prev[0], chunk[1]))
                        merged = True
                    elif right_sim > 0:
                        # Merge with right neighbor (skip this chunk, extend next)
                        if i + 1 < len(chunks):
                            next_chunk = chunks[i + 1]
                            chunks[i + 1] = (chunk[0], next_chunk[1])
                            merged = True
                        else:
                            # No right neighbor, merge with left if possible
                            if len(new_chunks) > 0:
                                prev = new_chunks.pop()
                                new_chunks.append((prev[0], chunk[1]))
                                merged = True
                            else:
                                new_chunks.append(chunk)
                    else:
                        # No neighbors to merge with
                        new_chunks.append(chunk)
                else:
                    new_chunks.append(chunk)
                
                i += 1
            
            chunks = new_chunks
        
        # Step 5: Compute mean-pooled node features
        node_feats = []
        for start, end in chunks:
            chunk_feat = feats[start:end + 1].mean(axis=0)
            # Re-normalize
            norm = np.linalg.norm(chunk_feat)
            if norm > 1e-8:
                chunk_feat = chunk_feat / norm
            node_feats.append(chunk_feat)
        
        node_feats = np.array(node_feats)  # [K, D]
        
        return chunks, node_feats

