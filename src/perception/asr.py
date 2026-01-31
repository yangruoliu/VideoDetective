"""
ASR (Automatic Speech Recognition) Module for VideoDetective.

Uses OpenAI Whisper to extract speech transcripts from video audio.
"""

import os
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings


class ASRExtractor:
    """
    ASR extractor using OpenAI Whisper.
    
    Extracts audio from video and transcribes it to text with timestamps.
    """
    
    def __init__(
        self,
        model_name: str = "base",
        model_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize ASR extractor.
        
        Args:
            model_name: Whisper model name ("tiny", "base", "small", "medium", "large")
            model_dir: Directory containing Whisper model files
            device: Device to run on ("cuda" or "cpu"). Auto-detect if None.
        """
        self.model_name = model_name
        self.model_dir = model_dir
        self.device = device
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model."""
        try:
            import whisper
            import torch
            
            # Determine device
            if self.device is None:
                env_dev = os.getenv("ASR_DEVICE", os.getenv("VIDEODETECTIVE_ASR_DEVICE", "")).strip().lower()
                if env_dev in ("cpu", "cuda"):
                    self.device = env_dev
                else:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Try to load from local model directory first
            if self.model_dir:
                model_path = Path(self.model_dir) / f"{self.model_name}.pt"
                if model_path.exists():
                    print(f"[ASR] Loading Whisper model from: {model_path}")
                    self.model = whisper.load_model(str(model_path), device=self.device)
                    print(f"[ASR] Whisper {self.model_name} loaded on {self.device}")
                    return
            
            # Fall back to downloading/cache
            print(f"[ASR] Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            print(f"[ASR] Whisper {self.model_name} loaded on {self.device}")
            
        except ImportError:
            warnings.warn("openai-whisper not installed. ASR will be disabled. Install with: pip install openai-whisper")
            self.model = None
        except Exception as e:
            warnings.warn(f"Failed to load Whisper model: {e}. ASR will be disabled.")
            self.model = None
    
    def extract_audio(self, video_path: str) -> Optional[str]:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to temporary audio file, or None if extraction failed
        """
        try:
            # Create temporary file for audio
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_audio.close()
            
            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", "16000",  # 16kHz sample rate (Whisper requirement)
                "-ac", "1",  # Mono
                temp_audio.name
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                # Check if video has no audio stream
                if "does not contain any stream" in result.stderr or "no audio" in result.stderr.lower():
                    print("[ASR] Video has no audio stream")
                    os.unlink(temp_audio.name)
                    return None
                print(f"[ASR] FFmpeg error: {result.stderr[:200]}")
                os.unlink(temp_audio.name)
                return None
            
            return temp_audio.name
            
        except subprocess.TimeoutExpired:
            print("[ASR] Audio extraction timed out")
            return None
        except FileNotFoundError:
            print("[ASR] FFmpeg not found. Please install ffmpeg.")
            return None
        except Exception as e:
            print(f"[ASR] Audio extraction failed: {e}")
            return None
    
    def transcribe(
        self,
        video_path: str,
        language: Optional[str] = None
    ) -> List[Dict]:
        """
        Transcribe video audio to text with timestamps.
        
        Args:
            video_path: Path to video file
            language: Language code (e.g., "en", "zh"). Auto-detect if None.
            
        Returns:
            List of segments, each with "start", "end", "text" fields
        """
        if self.model is None:
            return []
        
        # Extract audio
        audio_path = self.extract_audio(video_path)
        if audio_path is None:
            return []
        
        try:
            # Transcribe with Whisper
            print(f"[ASR] Transcribing audio...")
            result = self.model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                verbose=False
            )
            
            # Extract segments
            segments = []
            for seg in result.get("segments", []):
                segments.append({
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "text": seg["text"].strip()
                })
            
            print(f"[ASR] Transcribed {len(segments)} segments")
            return segments
            
        except Exception as e:
            print(f"[ASR] Transcription failed: {e}")
            return []
        finally:
            # Clean up temp audio file
            try:
                os.unlink(audio_path)
            except:
                pass
    
    def get_asr_by_frame(
        self,
        video_path: str,
        fps: float = 1.0,
        language: Optional[str] = None
    ) -> Dict[int, str]:
        """
        Get ASR text mapped to frame indices.
        
        Args:
            video_path: Path to video file
            fps: Frames per second for mapping
            language: Language code for transcription
            
        Returns:
            Dict mapping frame index to ASR text
        """
        segments = self.transcribe(video_path, language)
        
        asr_by_frame: Dict[int, str] = {}
        
        for seg in segments:
            start_frame = int(seg["start"] * fps)
            end_frame = int(seg["end"] * fps)
            text = seg["text"]
            
            if not text:
                continue
            
            for frame_idx in range(start_frame, end_frame + 1):
                if frame_idx not in asr_by_frame:
                    asr_by_frame[frame_idx] = text
                else:
                    # Avoid duplicating the same text
                    if text not in asr_by_frame[frame_idx]:
                        asr_by_frame[frame_idx] += " " + text
        
        return asr_by_frame


# Global instance (lazy initialization)
_asr_extractor: Optional[ASRExtractor] = None


def get_asr_extractor(
    model_name: str = "base",
    model_dir: Optional[str] = None
) -> Optional[ASRExtractor]:
    """Get or create the global ASR extractor instance."""
    global _asr_extractor
    
    if _asr_extractor is None:
        _asr_extractor = ASRExtractor(model_name=model_name, model_dir=model_dir)
    
    return _asr_extractor if _asr_extractor.model is not None else None
