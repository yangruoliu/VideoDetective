"""
Mathematical Operations for VideoDetective.

Provides vectorized NumPy operations for signal processing.
"""

import numpy as np
from typing import Optional


def gaussian_kernel(
    length: int,
    center: float,
    sigma: float,
    height: float = 1.0
) -> np.ndarray:
    """
    Generate a 1D Gaussian kernel.
    
    Formula: G(t) = height * exp(-(t - center)^2 / (2 * sigma^2))
    
    Args:
        length: Length of the output array.
        center: Center position of the Gaussian.
        sigma: Standard deviation of the Gaussian.
        height: Peak height of the Gaussian. Default is 1.0.
    
    Returns:
        1D numpy array of shape [length] containing the Gaussian kernel.
    
    Examples:
        >>> kernel = gaussian_kernel(10, 5.0, 2.0, 1.0)
        >>> kernel.shape
        (10,)
        >>> kernel[5]  # Peak value at center
        1.0
    """
    if length <= 0:
        return np.array([])
    
    if sigma <= 0:
        # Return delta function at center if sigma is zero or negative
        result = np.zeros(length)
        center_idx = int(np.clip(center, 0, length - 1))
        result[center_idx] = height
        return result
    
    t = np.arange(length)
    kernel = height * np.exp(-((t - center) ** 2) / (2 * sigma ** 2))
    
    return kernel


def softmax_sharpen(
    x: np.ndarray,
    temperature: float = 0.1
) -> np.ndarray:
    """
    Apply temperature-scaled softmax for sharpening probability distributions.
    
    Formula: softmax(x / T) where T is the temperature.
    
    Lower temperature -> sharper distribution (more peaked).
    Higher temperature -> smoother distribution.
    
    Args:
        x: Input array of any shape.
        temperature: Temperature parameter for scaling. Default is 0.1.
    
    Returns:
        Array of same shape as input with softmax applied along last axis.
    
    Note:
        Uses numerical stability trick of subtracting max before exp.
    
    Examples:
        >>> x = np.array([0.1, 0.5, 0.3, 0.1])
        >>> result = softmax_sharpen(x, temperature=0.1)
        >>> np.sum(result)  # Sums to 1
        1.0
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    # Scale by temperature
    x_scaled = np.asarray(x) / temperature
    
    # Numerical stability: subtract max
    x_max = np.max(x_scaled, axis=-1, keepdims=True)
    x_exp = np.exp(x_scaled - x_max)
    
    # Normalize
    return x_exp / np.sum(x_exp, axis=-1, keepdims=True)


def cosine_similarity_matrix(
    features: np.ndarray,
    normalize: bool = False
) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.
    
    Args:
        features: Feature matrix of shape [N, D].
        normalize: If True, apply L2 normalization to features first.
    
    Returns:
        Similarity matrix of shape [N, N].
    """
    if normalize:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        features = features / norms
    
    # Compute dot product (cosine similarity for normalized vectors)
    similarity = features @ features.T
    
    return similarity


def temporal_decay_mask(
    length: int,
    gamma: float = 1e-3
) -> np.ndarray:
    """
    Create temporal decay mask for affinity matrix.
    
    Formula: M_ij = exp(-gamma * (i - j)^2)
    
    Args:
        length: Size of the square mask.
        gamma: Decay rate. Default is 1e-3.
    
    Returns:
        Square mask matrix of shape [length, length].
    """
    indices = np.arange(length)
    diff = indices[:, None] - indices[None, :]
    mask = np.exp(-gamma * (diff ** 2))
    
    return mask


def find_peaks(
    signal: np.ndarray,
    k: int,
    min_distance: int = 1
) -> np.ndarray:
    """
    Find top-k peaks in a 1D signal with minimum distance constraint.
    
    Args:
        signal: 1D array of values.
        k: Number of peaks to find.
        min_distance: Minimum distance between peaks. Default is 1.
    
    Returns:
        Array of peak indices, sorted by position.
    """
    if len(signal) == 0:
        return np.array([], dtype=int)
    
    k = min(k, len(signal))
    
    # Get indices sorted by signal value (descending)
    sorted_indices = np.argsort(signal)[::-1]
    
    selected = []
    for idx in sorted_indices:
        if len(selected) >= k:
            break
        
        # Check minimum distance to already selected peaks
        too_close = False
        for selected_idx in selected:
            if abs(idx - selected_idx) < min_distance:
                too_close = True
                break
        
        if not too_close:
            selected.append(idx)
    
    # Sort by position
    return np.sort(selected)


def normalize_to_range(
    x: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> np.ndarray:
    """
    Normalize array to specified range.
    
    Args:
        x: Input array.
        min_val: Target minimum value.
        max_val: Target maximum value.
    
    Returns:
        Normalized array.
    """
    x_min = np.min(x)
    x_max = np.max(x)
    
    if x_max - x_min < 1e-8:
        return np.full_like(x, (min_val + max_val) / 2)
    
    return (x - x_min) / (x_max - x_min) * (max_val - min_val) + min_val
