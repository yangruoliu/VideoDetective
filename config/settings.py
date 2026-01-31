"""
VideoDetective Configuration Settings.

This module handles all configuration loading from environment variables
and provides default values for the system.

Environment Variables:
    VLM API Settings:
        - VIDEODETECTIVE_API_KEY: API key for VLM service (required)
        - VIDEODETECTIVE_BASE_URL: API base URL (default: OpenAI-compatible endpoint)
        - VIDEODETECTIVE_VLM_MODEL: VLM model name (default: qwen3-vl-8b-instruct)
    
    Text LLM Settings (optional, falls back to VLM settings):
        - VIDEODETECTIVE_LLM_MODEL: Text LLM model name
        - VIDEODETECTIVE_LLM_API_KEY: Separate API key for text LLM
        - VIDEODETECTIVE_LLM_BASE_URL: Separate base URL for text LLM
    
    Encoder Settings:
        - SIGLIP_MODEL_ID: HuggingFace model ID for SigLIP (default: google/siglip-so400m-patch14-384)
        - CLIP_LOCAL_PATH: Local path to CLIP model (fallback)
        - HF_CACHE_DIR: HuggingFace cache directory
        - HF_ENDPOINT: HuggingFace mirror endpoint (for users in China)
    
    Pipeline Settings:
        - ENABLE_MULTI_ROUTE_RECALL: Enable multi-route recall (default: true)
        - USE_VLM_RELEVANCE: Use VLM-based relevance scoring (default: false)
        - INCLUDE_ANSWER_EVIDENCE: Include evidence in final answer (default: true)
"""

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class QwenConfig:
    """VLM API configuration (OpenAI-compatible)."""
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3-vl-8b-instruct"
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout: float = 300.0
    max_frames_per_call: int = 64


@dataclass
class TextLLMConfig:
    """Text-only LLM configuration (can use separate API settings from VLM)."""
    model: str = "qwen3-8b"
    max_tokens: int = 2048
    temperature: float = 0.0
    api_key: str = ""  # Falls back to VLM settings if not set
    base_url: str = ""  # Falls back to VLM settings if not set


@dataclass
class EncoderConfig:
    """Encoder model configuration."""
    # SigLIP model ID (will look in local models/ directory first, then download)
    siglip_model_id: str = "google/siglip-so400m-patch14-384"
    # CLIP local path (fallback if SigLIP fails)
    clip_local_path: str = ""
    # HuggingFace cache directory (default: project models/ directory)
    cache_dir: str = ""
    # HuggingFace mirror endpoint (for users in China)
    hf_endpoint: str = "https://hf-mirror.com"
    
    # Runtime state
    use_siglip: bool = True
    actual_model_path: Optional[str] = None


@dataclass
class Settings:
    """Main settings container for VideoDetective."""
    qwen: QwenConfig = field(default_factory=QwenConfig)
    text_llm: TextLLMConfig = field(default_factory=TextLLMConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    
    # Pipeline defaults
    default_fps: float = 1.0
    max_steps: int = 5
    total_frame_budget: int = 64
    default_k_clusters: int = 8

    # V4.3: Multi-route recall settings
    enable_multi_route_recall: bool = True
    multi_route_alpha: float = 0.5  # 50% visual + 50% semantic
    skeleton_frames: int = 32

    # V4.7: Text-only scoring (disable subjective VLM relevance)
    use_vlm_relevance: bool = False

    # V4.7: Answer-time evidence attachment
    include_answer_evidence: bool = True
    
    def __post_init__(self):
        """Load settings from environment after initialization."""
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Set default cache directory to project's models/ directory
        project_root = Path(__file__).parent.parent
        default_cache_dir = str(project_root / "models")
        default_clip_path = str(project_root / "models" / "models--openai--clip-vit-base-patch32")
        default_hf_endpoint = self.encoder.hf_endpoint

        # Load .env file from project root
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        
        # ===== VLM API Settings =====
        self.qwen.api_key = os.getenv(
            "VIDEODETECTIVE_API_KEY",
            os.getenv("OPENAI_API_KEY", os.getenv("QWEN_API_KEY", self.qwen.api_key)),
        )
        self.qwen.base_url = os.getenv(
            "VIDEODETECTIVE_BASE_URL",
            os.getenv("OPENAI_BASE_URL", os.getenv("QWEN_BASE_URL", self.qwen.base_url)),
        )
        self.qwen.model = os.getenv(
            "VIDEODETECTIVE_VLM_MODEL",
            os.getenv("VLM_MODEL", os.getenv("QWEN_VLM_MODEL", os.getenv("QWEN_MODEL", self.qwen.model))),
        )
        
        # ===== Text LLM Settings =====
        self.text_llm.model = os.getenv(
            "VIDEODETECTIVE_LLM_MODEL",
            os.getenv("LLM_MODEL", os.getenv("QWEN_LLM_MODEL", os.getenv("TEXT_LLM_MODEL", self.text_llm.model))),
        )
        self.text_llm.api_key = os.getenv(
            "VIDEODETECTIVE_LLM_API_KEY",
            os.getenv("TEXT_LLM_API_KEY", ""),
        )
        self.text_llm.base_url = os.getenv(
            "VIDEODETECTIVE_LLM_BASE_URL",
            os.getenv("TEXT_LLM_BASE_URL", ""),
        )
        
        # Optional overrides
        self.qwen.max_tokens = int(os.getenv("QWEN_MAX_TOKENS", str(self.qwen.max_tokens)))
        self.text_llm.max_tokens = int(os.getenv("TEXT_LLM_MAX_TOKENS", str(self.text_llm.max_tokens)))
        self.qwen.temperature = float(os.getenv("QWEN_TEMPERATURE", str(self.qwen.temperature)))
        self.text_llm.temperature = float(os.getenv("TEXT_LLM_TEMPERATURE", str(self.text_llm.temperature)))
        
        try:
            self.qwen.max_frames_per_call = int(
                os.getenv(
                    "VIDEODETECTIVE_MAX_FRAMES_PER_CALL",
                    os.getenv("QWEN_MAX_FRAMES_PER_CALL", str(self.qwen.max_frames_per_call)),
                )
            )
        except Exception:
            pass
        self.qwen.max_frames_per_call = max(1, int(self.qwen.max_frames_per_call))

        # ===== Pipeline Settings =====
        def _parse_bool(v: str, default: bool) -> bool:
            if v is None:
                return default
            s = str(v).strip().lower()
            if s in ("1", "true", "yes", "y", "on"):
                return True
            if s in ("0", "false", "no", "n", "off"):
                return False
            return default

        self.enable_multi_route_recall = _parse_bool(
            os.getenv("ENABLE_MULTI_ROUTE_RECALL", os.getenv("VIDEODETECTIVE_ENABLE_MULTI_ROUTE_RECALL", "")),
            self.enable_multi_route_recall,
        )
        self.use_vlm_relevance = _parse_bool(
            os.getenv("USE_VLM_RELEVANCE", os.getenv("VIDEODETECTIVE_USE_VLM_RELEVANCE", "")),
            self.use_vlm_relevance,
        )
        self.include_answer_evidence = _parse_bool(
            os.getenv("INCLUDE_ANSWER_EVIDENCE", os.getenv("VIDEODETECTIVE_INCLUDE_ANSWER_EVIDENCE", "")),
            self.include_answer_evidence,
        )
        
        try:
            self.multi_route_alpha = float(
                os.getenv("MULTI_ROUTE_ALPHA", os.getenv("VIDEODETECTIVE_MULTI_ROUTE_ALPHA", str(self.multi_route_alpha)))
            )
        except Exception:
            pass
        self.multi_route_alpha = max(0.0, min(1.0, float(self.multi_route_alpha)))
        
        try:
            self.skeleton_frames = int(
                os.getenv("SKELETON_FRAMES", os.getenv("VIDEODETECTIVE_SKELETON_FRAMES", str(self.skeleton_frames)))
            )
        except Exception:
            pass
        self.skeleton_frames = max(0, int(self.skeleton_frames))
        
        # ===== Encoder Settings =====
        self.encoder.siglip_model_id = os.getenv("SIGLIP_MODEL_ID", self.encoder.siglip_model_id)
        self.encoder.clip_local_path = os.getenv("CLIP_LOCAL_PATH", default_clip_path)
        self.encoder.cache_dir = os.getenv("HF_CACHE_DIR", default_cache_dir)
        self.encoder.hf_endpoint = os.getenv("HF_ENDPOINT", self.encoder.hf_endpoint)

        # Sanitize HF endpoint
        if self.encoder.hf_endpoint:
            hf = str(self.encoder.hf_endpoint).strip()
            if not (hf.startswith("http://") or hf.startswith("https://")):
                warnings.warn(
                    f"HF_ENDPOINT is not a valid URL ({hf!r}); falling back to default {default_hf_endpoint!r}."
                )
                hf = default_hf_endpoint
            self.encoder.hf_endpoint = hf
        
        # Set HF mirror endpoint
        if self.encoder.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self.encoder.hf_endpoint
    
    def get_encoder_model_info(self) -> tuple[str, str, bool]:
        """
        Get encoder model information.
        
        Returns:
            Tuple of (model_path_or_id, cache_dir, is_siglip)
        """
        return self.encoder.siglip_model_id, self.encoder.cache_dir, True
    
    def get_clip_fallback_path(self) -> str:
        """Get CLIP local path for fallback."""
        clip_path = Path(self.encoder.clip_local_path)
        if self.encoder.clip_local_path and not clip_path.exists():
            warnings.warn(f"CLIP local path does not exist: {clip_path}")
        return str(clip_path)


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
