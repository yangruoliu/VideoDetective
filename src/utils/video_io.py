"""
Video I/O Utilities.

Provides functions for video loading and frame extraction using decord.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Union

import numpy as np
from PIL import Image

try:
    from decord import VideoReader, cpu
except ImportError:
    raise ImportError("decord is required. Install with: pip install decord")

IndexLike = Union[int, slice]


def compute_sample_indices(
    video_path: str,
    fps: float = 1.0,
    max_frames: Optional[int] = None
) -> Tuple[List[int], List[float]]:
    """
    Compute sampled frame indices (and timestamps) without decoding frames.

    This is much faster than decoding all sampled frames and is used to enable
    lazy frame access when embeddings cache is hit.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
    except Exception as e:
        raise ValueError(f"Failed to read video: {video_path}. Error: {e}")

    total_frames = len(vr)
    video_fps = vr.get_avg_fps()

    frame_interval = video_fps / fps
    indices = np.arange(0, total_frames, frame_interval).astype(int)

    if max_frames is not None and len(indices) > max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames).astype(int)

    indices = np.clip(indices, 0, total_frames - 1)
    indices = np.unique(indices)
    idx_list = indices.tolist()
    timestamps = [int(i) / video_fps for i in idx_list]
    return idx_list, timestamps


class LazyVideoFrames:
    """
    Lazy frame container that behaves like a list of PIL.Image, but only decodes frames on demand.

    Indices are positions in the sampled frame list (0..T-1). Internally we map to
    original video frame indices for decord decoding.
    """

    def __init__(self, video_path: str, sampled_indices: Sequence[int]):
        self.video_path = str(Path(video_path))
        self.sampled_indices = [int(i) for i in sampled_indices]
        self._vr: Optional[VideoReader] = None

    def __len__(self) -> int:
        return len(self.sampled_indices)

    def _get_vr(self) -> VideoReader:
        if self._vr is None:
            self._vr = VideoReader(self.video_path, ctx=cpu(0))
        return self._vr

    def get_batch(self, positions: Sequence[int]) -> List[Image.Image]:
        """Decode a batch of sampled-frame positions."""
        if not positions:
            return []
        vr = self._get_vr()
        vid_indices = [self.sampled_indices[int(p)] for p in positions if 0 <= int(p) < len(self.sampled_indices)]
        if not vid_indices:
            return []
        frames_array = vr.get_batch(vid_indices).asnumpy()
        return [Image.fromarray(frame) for frame in frames_array]

    def __getitem__(self, idx: IndexLike) -> Union[Image.Image, List[Image.Image]]:
        if isinstance(idx, slice):
            positions = list(range(*idx.indices(len(self))))
            return self.get_batch(positions)
        pos = int(idx)
        if pos < 0:
            pos = len(self.sampled_indices) + pos
        if pos < 0 or pos >= len(self.sampled_indices):
            raise IndexError(pos)
        return self.get_batch([pos])[0]


class LazyImagePaths:
    """
    Lazy image container backed by file paths. Only opens images on demand.
    """

    def __init__(self, image_paths: Sequence[str]):
        self.image_paths = [str(p) for p in image_paths]

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_batch(self, positions: Sequence[int]) -> List[Image.Image]:
        out: List[Image.Image] = []
        for p in positions:
            i = int(p)
            if i < 0 or i >= len(self.image_paths):
                continue
            fp = self.image_paths[i]
            try:
                with Image.open(fp) as im:
                    out.append(im.convert("RGB").copy())
            except Exception:
                # Skip unreadable frames
                continue
        return out

    def __getitem__(self, idx: IndexLike) -> Union[Image.Image, List[Image.Image]]:
        if isinstance(idx, slice):
            positions = list(range(*idx.indices(len(self))))
            return self.get_batch(positions)
        pos = int(idx)
        if pos < 0:
            pos = len(self.image_paths) + pos
        if pos < 0 or pos >= len(self.image_paths):
            raise IndexError(pos)
        batch = self.get_batch([pos])
        if not batch:
            raise IndexError(pos)
        return batch[0]


def load_video(
    video_path: str,
    fps: float = 1.0,
    max_frames: Optional[int] = None
) -> Tuple[List[Image.Image], List[float]]:
    """
    Load video and sample frames at specified FPS.
    
    Args:
        video_path: Path to the video file.
        fps: Target frames per second for sampling. Default is 1.0.
        max_frames: Maximum number of frames to extract. None for no limit.
    
    Returns:
        Tuple of:
            - frames: List of PIL.Image frames
            - timestamps: List of timestamps (in seconds) for each frame
    
    Raises:
        FileNotFoundError: If video file does not exist.
        ValueError: If video cannot be read.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
    except Exception as e:
        raise ValueError(f"Failed to read video: {video_path}. Error: {e}")
    
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    duration = total_frames / video_fps
    
    # Calculate frame indices based on target FPS
    frame_interval = video_fps / fps
    indices = np.arange(0, total_frames, frame_interval).astype(int)
    
    # Limit number of frames if specified
    if max_frames is not None and len(indices) > max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames).astype(int)
    
    # Ensure indices are valid
    indices = np.clip(indices, 0, total_frames - 1)
    indices = np.unique(indices)
    
    # Extract frames
    frames_array = vr.get_batch(indices.tolist()).asnumpy()
    
    # Convert to PIL Images
    frames = [Image.fromarray(frame) for frame in frames_array]
    
    # Calculate timestamps
    timestamps = [idx / video_fps for idx in indices]
    
    return frames, timestamps


def extract_frames_at_indices(
    video_path: str,
    indices: List[int]
) -> List[Image.Image]:
    """
    Extract specific frames from video by their indices.
    
    Args:
        video_path: Path to the video file.
        indices: List of frame indices to extract.
    
    Returns:
        List of PIL.Image frames.
    
    Raises:
        FileNotFoundError: If video file does not exist.
        ValueError: If video cannot be read or indices are invalid.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
    except Exception as e:
        raise ValueError(f"Failed to read video: {video_path}. Error: {e}")
    
    total_frames = len(vr)
    
    # Validate and clip indices
    valid_indices = [int(idx) for idx in indices if 0 <= idx < total_frames]
    
    if len(valid_indices) == 0:
        return []
    
    # Extract frames
    frames_array = vr.get_batch(valid_indices).asnumpy()
    frames = [Image.fromarray(frame) for frame in frames_array]
    
    return frames


def extract_clip_frames(
    video_path: str,
    center_time: float,
    window_seconds: float = 2.0,
    clip_fps: float = 1.0
) -> Tuple[List[Image.Image], List[float]]:
    """
    Extract frames from a time window around a center point.
    
    Args:
        video_path: Path to the video file.
        center_time: Center time in seconds.
        window_seconds: Half-window size in seconds (total window = 2 * window_seconds).
        clip_fps: FPS for sampling within the clip.
    
    Returns:
        Tuple of:
            - frames: List of PIL.Image frames
            - timestamps: List of timestamps (in seconds) for each frame
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
    except Exception as e:
        raise ValueError(f"Failed to read video: {video_path}. Error: {e}")
    
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    duration = total_frames / video_fps
    
    # Calculate time window
    start_time = max(0, center_time - window_seconds)
    end_time = min(duration, center_time + window_seconds)
    
    # Convert to frame indices
    start_frame = int(start_time * video_fps)
    end_frame = int(end_time * video_fps)
    
    # Sample within clip
    frame_interval = video_fps / clip_fps
    indices = np.arange(start_frame, end_frame, frame_interval).astype(int)
    indices = np.clip(indices, 0, total_frames - 1)
    indices = np.unique(indices)
    
    if len(indices) == 0:
        return [], []
    
    # Extract frames
    frames_array = vr.get_batch(indices.tolist()).asnumpy()
    frames = [Image.fromarray(frame) for frame in frames_array]
    timestamps = [idx / video_fps for idx in indices]
    
    return frames, timestamps


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.
    
    Args:
        video_path: Path to the video file.
    
    Returns:
        Dictionary containing:
            - duration: Video duration in seconds
            - fps: Average FPS
            - total_frames: Total number of frames
            - width: Frame width
            - height: Frame height
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
    except Exception as e:
        raise ValueError(f"Failed to read video: {video_path}. Error: {e}")
    
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    
    # Get first frame to determine dimensions
    first_frame = vr[0].asnumpy()
    height, width = first_frame.shape[:2]
    
    return {
        "duration": total_frames / fps,
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
    }
