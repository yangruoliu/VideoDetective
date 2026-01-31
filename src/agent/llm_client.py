"""
LLM Client for VideoDetective.

OpenAI-compatible API client supporting text and multimodal inputs.
"""

import base64
import json
import re
import time
import random
import os
from io import BytesIO
from typing import List, Optional, Union, Dict, Any

from PIL import Image

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("openai is required. Install with: pip install openai")


class LLMClient:
    """
    OpenAI-compatible LLM client for VideoDetective.
    
    Supports:
        - Text-only queries
        - Multimodal queries with images
        - JSON response parsing
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3-vl-8b-instruct",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout: float = 300.0,
        max_frames_per_call: int = 64,
        enable_thinking: bool = False,
        retry_max_attempts: int = 5,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 20.0,
    ):
        """
        Initialize LLM client.
        
        Args:
            api_key: API key for authentication.
            base_url: API base URL.
            model: Model name/ID.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            timeout: Request timeout in seconds.
            max_frames_per_call: Maximum frames per VLM call.
        """
        # Auth header customization for OpenAI-compatible proxies.
        # Default: Authorization: Bearer <api_key> (OpenAI style).
        #
        # Some "API aggregation" proxies expect different auth headers, e.g.:
        #   - x-api-key: <api_key>
        #   - x-goog-api-key: <api_key>
        #
        # You can control this via env vars (no code changes needed per run):
        #   - VIDEODETECTIVE_AUTH_HEADER_NAME   (default: Authorization)
        #   - VIDEODETECTIVE_AUTH_PREFIX        (default: Bearer)
        auth_header_name = (os.getenv("VIDEODETECTIVE_AUTH_HEADER_NAME") or "Authorization").strip()
        auth_prefix = os.getenv("VIDEODETECTIVE_AUTH_PREFIX")
        if auth_prefix is None:
            auth_prefix = "Bearer"
        auth_prefix = str(auth_prefix).strip()

        default_headers: Dict[str, str] = {}
        if auth_header_name and auth_header_name.lower() != "authorization":
            # Avoid sending a potentially confusing/invalid Authorization header when proxy expects x-api-key.
            # (OpenAI SDK will still set Authorization from api_key; we override it to empty as best-effort.)
            default_headers["Authorization"] = ""
            default_headers[auth_header_name] = api_key if not auth_prefix else f"{auth_prefix} {api_key}"
            # Use a non-empty dummy key to satisfy OpenAI SDK internal checks.
            api_key_for_sdk = "DUMMY"
        else:
            # Standard OpenAI auth
            api_key_for_sdk = api_key
            # Allow overriding Authorization formatting if needed (rare)
            if auth_prefix != "Bearer":
                default_headers["Authorization"] = api_key if not auth_prefix else f"{auth_prefix} {api_key}"

        self.client = OpenAI(
            api_key=api_key_for_sdk,
            base_url=base_url,
            timeout=timeout,
            default_headers=default_headers if default_headers else None,
        )
        self.model = model
        
        # GLM-4V (not 4.5V) has strict limitations
        model_lower = str(model).lower()
        if "glm-4v" in model_lower and "4.5" not in model_lower:
            # GLM-4V (not 4.5V) has lower max_tokens limit (2048) and max 5 images per call
            self.max_tokens = min(max_tokens, 2048)
            self.max_frames_per_call = min(max_frames_per_call, 5)
        else:
            # GLM-4.5V and other models use default limits
            self.max_tokens = max_tokens
            self.max_frames_per_call = max_frames_per_call
            
        self.temperature = temperature
        # Qwen OpenAI-compatible API requirement:
        # For non-streaming calls, parameter.enable_thinking must be false.
        self.enable_thinking = bool(enable_thinking)
        # Retry settings for transient provider/network errors
        self.retry_max_attempts = int(max(1, retry_max_attempts))
        self.retry_base_delay = float(max(0.1, retry_base_delay))
        self.retry_max_delay = float(max(self.retry_base_delay, retry_max_delay))

        # Best-effort usage accounting (per-client instance).
        # Not all OpenAI-compatible providers return granular usage (e.g., image/visual tokens).
        self.reset_usage()

    def reset_usage(self) -> None:
        """Reset accumulated usage stats for this client instance."""
        self._usage: Dict[str, Any] = {
            "calls": 0,
            "text_calls": 0,
            "multimodal_calls": 0,
            "images_sent": 0,
            # Common OpenAI fields
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            # Common alternative fields used by some proxies/providers
            "input_tokens": 0,
            "output_tokens": 0,
            # Best-effort visual tokens (provider-dependent)
            "image_tokens": 0,
            # Keep the last raw usage object (for debugging)
            "last_usage_raw": None,
        }

    def get_usage_summary(self) -> Dict[str, Any]:
        """Return a shallow copy of current usage stats."""
        return dict(self._usage or {})

    def _extract_usage(self, resp: Any) -> Dict[str, Any]:
        """Best-effort extraction of usage fields from OpenAI-compatible responses."""
        if resp is None:
            return {}

        # OpenAI SDK object: resp.usage with prompt_tokens/completion_tokens/total_tokens
        try:
            usage_obj = getattr(resp, "usage", None)
            if usage_obj is not None:
                out: Dict[str, Any] = {}
                for k in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
                    v = getattr(usage_obj, k, None)
                    if isinstance(v, (int, float)):
                        out[k] = int(v)
                # OpenAI-style nested: prompt_tokens_details.image_tokens
                try:
                    ptd = getattr(usage_obj, "prompt_tokens_details", None)
                    img_toks = getattr(ptd, "image_tokens", None) if ptd is not None else None
                    if isinstance(img_toks, (int, float)):
                        out["image_tokens"] = int(img_toks)
                except Exception:
                    pass
                if out:
                    out["raw"] = usage_obj
                    return out
        except Exception:
            pass

        # Dict-style response: resp.get("usage")
        if isinstance(resp, dict):
            u = resp.get("usage", None)
            if isinstance(u, dict):
                out: Dict[str, Any] = {}
                for k in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens", "image_tokens"):
                    v = u.get(k, None)
                    if isinstance(v, (int, float)):
                        out[k] = int(v)
                # Some providers nest details
                ptd = u.get("prompt_tokens_details", None)
                if isinstance(ptd, dict):
                    v = ptd.get("image_tokens", None)
                    if isinstance(v, (int, float)):
                        out["image_tokens"] = int(v)
                if out:
                    out["raw"] = u
                    return out

        # Last resort: try model_dump() / dict() if available, then read ["usage"]
        try:
            if hasattr(resp, "model_dump"):
                d = resp.model_dump()  # type: ignore[attr-defined]
                if isinstance(d, dict):
                    u = d.get("usage", None)
                    if isinstance(u, dict):
                        return self._extract_usage({"usage": u})
        except Exception:
            pass

        return {}

    def _accumulate_usage(self, resp: Any, modality: str = "text", images_sent: int = 0) -> None:
        """Accumulate usage stats from a response."""
        try:
            self._usage["calls"] = int(self._usage.get("calls", 0)) + 1
            if str(modality) == "multimodal":
                self._usage["multimodal_calls"] = int(self._usage.get("multimodal_calls", 0)) + 1
                self._usage["images_sent"] = int(self._usage.get("images_sent", 0)) + int(images_sent or 0)
            else:
                self._usage["text_calls"] = int(self._usage.get("text_calls", 0)) + 1
        except Exception:
            pass

        u = self._extract_usage(resp)
        if not u:
            return
        try:
            self._usage["last_usage_raw"] = u.get("raw")
            for k in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens", "image_tokens"):
                v = u.get(k, None)
                if isinstance(v, (int, float)):
                    self._usage[k] = int(self._usage.get(k, 0)) + int(v)
        except Exception:
            pass

    def _extract_request_id(self, msg: str) -> Optional[str]:
        """Best-effort extract request_id from exception string."""
        if not msg:
            return None
        # Common patterns in provider errors
        m = re.search(r'"request_id"\s*:\s*"([^"]+)"', msg)
        if m:
            return m.group(1)
        m = re.search(r"request[_-]?id[=:\s]+([A-Za-z0-9\\-_.]+)", msg, re.IGNORECASE)
        if m:
            return m.group(1)
        return None

    def _is_transient_provider_error(self, e: Exception) -> bool:
        """Heuristic for transient connectivity/provider availability issues."""
        msg = (str(e) or "").lower()
        transient_markers = [
            "unable to reach the model provider",
            "error_openai",
            "connection error",
            "connect",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "502",
            "503",
            "504",
        ]
        return any(m in msg for m in transient_markers)

    def _sleep_backoff(self, attempt: int) -> None:
        """Exponential backoff with jitter."""
        # attempt starts at 1
        base = self.retry_base_delay * (2 ** (attempt - 1))
        delay = min(self.retry_max_delay, base)
        # jitter in [0.8, 1.2]
        delay = delay * (0.8 + 0.4 * random.random())
        time.sleep(delay)
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[str] = None,
        modality: str = "text",
        images_sent: int = 0,
    ) -> str:
        """
        Send chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            response_format: Optional format hint ('json' for JSON mode).
        
        Returns:
            Response content as string.
        """
        # Provider-specific knobs:
        # Qwen OpenAI-compatible endpoint may require enable_thinking=false for non-streaming.
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        # Only add extra_body for Qwen models (not for GLM or other providers)
        model_lower = str(self.model).lower()
        if "qwen" in model_lower and "glm" not in model_lower:
            kwargs["extra_body"] = {"enable_thinking": False}

        # Optional JSON mode hint (best-effort; not all providers support it)
        if response_format == "json":
            try:
                kwargs["response_format"] = {"type": "json_object"}
            except Exception:
                pass

        def _alias_candidates(model_name: str) -> List[str]:
            cur = (model_name or "").strip()
            if not cur:
                return []
            cands: List[str] = []
            fixed = cur.replace("instrcut", "instruct")
            if fixed != cur:
                cands.append(fixed)
            if "qwen3vl-" in fixed:
                cands.append(fixed.replace("qwen3vl-", "qwen3-vl-"))
            if "qwen3-vl-" in fixed:
                cands.append(fixed.replace("qwen3-vl-", "qwen3vl-"))
            # De-dup while preserving order
            seen = set()
            out: List[str] = []
            for m in cands:
                if m and m not in seen:
                    seen.add(m)
                    out.append(m)
            return out

        def _extract_content(resp: Any) -> str:
            """
            Extract assistant content from various OpenAI-compatible response shapes.
            Supports:
              - OpenAI SDK objects: resp.choices[0].message.content
              - dict: resp["choices"][0]["message"]["content"] or ["text"]
              - plain string: resp
            """
            if resp is None:
                return ""
            if isinstance(resp, str):
                return resp
            if isinstance(resp, dict):
                try:
                    choices = resp.get("choices", None)
                    if isinstance(choices, list) and choices:
                        c0 = choices[0] or {}
                        if isinstance(c0, dict):
                            msg = c0.get("message", None)
                            if isinstance(msg, dict) and "content" in msg:
                                return str(msg.get("content", "") or "")
                            # Some providers use "text" at choice level
                            if "text" in c0:
                                return str(c0.get("text", "") or "")
                    # Some proxies return { "content": "..."} directly
                    if "content" in resp:
                        return str(resp.get("content", "") or "")
                except Exception:
                    pass
                return str(resp)
            # OpenAI SDK object path
            try:
                return resp.choices[0].message.content  # type: ignore[attr-defined]
            except Exception:
                # last resort: stringify
                return str(resp)

        def _try_with_model(model_name: str) -> Optional[str]:
            """Try a specific model name (with transient retry). Return None if fails."""
            try:
                kwargs["model"] = model_name
                for attempt in range(1, self.retry_max_attempts + 1):
                    try:
                        response = self.client.chat.completions.create(**kwargs)
                        self._accumulate_usage(response, modality=str(modality), images_sent=int(images_sent or 0))
                        return _extract_content(response)
                    except Exception as e:
                        if self._is_transient_provider_error(e) and attempt < self.retry_max_attempts:
                            rid = self._extract_request_id(str(e))
                            rid_s = f", request_id={rid}" if rid else ""
                            print(f"[LLMClient] WARNING: transient provider error (attempt {attempt}/{self.retry_max_attempts}){rid_s}: {e}")
                            self._sleep_backoff(attempt)
                            continue
                        raise
            except Exception:
                return None

        # Main call with retry on transient errors
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                self._accumulate_usage(response, modality=str(modality), images_sent=int(images_sent or 0))
                return _extract_content(response)
            except Exception as e:
                error_msg = str(e)

                # Transient provider errors: retry with backoff
                if self._is_transient_provider_error(e) and attempt < self.retry_max_attempts:
                    rid = self._extract_request_id(error_msg)
                    rid_s = f", request_id={rid}" if rid else ""
                    print(f"[LLMClient] WARNING: transient provider error (attempt {attempt}/{self.retry_max_attempts}){rid_s}: {e}")
                    self._sleep_backoff(attempt)
                    continue

                # Content safety triggered; degrade but keep pipeline running.
                if "data_inspection_failed" in error_msg or "inappropriate content" in error_msg:
                    rid = self._extract_request_id(error_msg)
                    rid_s = f", request_id={rid}" if rid else ""
                    print(f"[LLMClient] WARNING: Content safety check failed{rid_s}: {error_msg}")
                    return "[SAFETY_FILTERED]"

                # Provider unreachable after retries; degrade but keep pipeline running.
                if "Unable to reach the model provider" in error_msg or "ERROR_OPENAI" in error_msg:
                    rid = self._extract_request_id(error_msg)
                    rid_s = f", request_id={rid}" if rid else ""
                    print(f"[LLMClient] WARNING: Model provider unreachable after retries{rid_s}: {error_msg}")
                    return "[PROVIDER_UNREACHABLE]"

                # Model alias fallback for model_not_found errors.
                if "model_not_found" in error_msg or "does not exist" in error_msg:
                    current = str(kwargs.get("model") or "")
                    for m in _alias_candidates(current):
                        out = _try_with_model(m)
                        if out is not None:
                            self.model = m  # lock in
                            return out

                # Otherwise: surface the error
                raise
    
    def chat_with_images(
        self,
        prompt: str,
        images: List[Image.Image],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Send multimodal chat request with images.
        
        Args:
            prompt: User prompt text.
            images: List of PIL.Image objects.
            system_prompt: Optional system prompt.
        
        Returns:
            Response content as string.
        """
        # Limit number of images
        if len(images) > self.max_frames_per_call:
            # Uniformly sample
            indices = [int(i * len(images) / self.max_frames_per_call) 
                      for i in range(self.max_frames_per_call)]
            images = [images[i] for i in indices]
        
        # Build content with images
        content = []
        
        # Add images first
        for img in images:
            base64_image = self._image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({
            "role": "user",
            "content": content
        })

        return self.chat(messages, modality="multimodal", images_sent=len(images))
    
    def chat_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        images: Optional[List[Image.Image]] = None,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Send request expecting JSON response with retry on parse failure.
        
        Args:
            prompt: User prompt (should request JSON format).
            system_prompt: Optional system prompt.
            images: Optional list of images for multimodal.
            max_retries: Maximum number of retries on JSON parse failure.
        
        Returns:
            Parsed JSON response as dict.
        
        Raises:
            ValueError: If JSON parsing fails after all retries.
        """
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                if images and len(images) > 0:
                    response = self.chat_with_images(prompt, images, system_prompt)
                else:
                    messages: List[Dict[str, Any]] = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                    response = self.chat(messages, response_format="json")
                
                return self._parse_json_response(response)
                
            except ValueError as e:
                # JSON parse failure - retry
                last_error = e
                if attempt < max_retries:
                    print(f"[LLMClient] JSON parse failed (attempt {attempt + 1}/{max_retries + 1}), retrying...")
                    continue
                break
            except Exception as e:
                error_msg = str(e)
                if "data_inspection_failed" in error_msg or "inappropriate content" in error_msg:
                    rid = self._extract_request_id(error_msg)
                    rid_s = f", request_id={rid}" if rid else ""
                    print(f"[LLMClient] WARNING: Content safety check failed{rid_s}: {error_msg}")
                    return {"error": "safety_check_failed", "lambda": 0.5, "relevance": 0.0, "reasoning": "Content safety triggered"}
                if "Unable to reach the model provider" in error_msg or "ERROR_OPENAI" in error_msg:
                    rid = self._extract_request_id(error_msg)
                    rid_s = f", request_id={rid}" if rid else ""
                    print(f"[LLMClient] WARNING: Model provider unreachable{rid_s}: {error_msg}")
                    return {"error": "provider_unreachable", "lambda": 0.5, "relevance": 0.0, "reasoning": "Model provider unreachable"}
                print(f"[LLMClient] Chat failed: {e}")
                raise

        # All retries exhausted
        print(f"[LLMClient] JSON parse failed after {max_retries + 1} attempts")
        if last_error is not None:
            raise last_error
        raise ValueError("JSON parse failed")
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize if too large (to save bandwidth)
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from response with fallback strategies.
        
        Args:
            response: Raw response string.
        
        Returns:
            Parsed JSON dict.
        """
        # Try direct parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code block
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{[^{}]*\}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    continue
        
        # Return default if all parsing fails
        raise ValueError(f"Failed to parse JSON from response: {response[:200]}...")
