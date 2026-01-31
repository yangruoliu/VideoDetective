"""
Observer for VideoDetective v4.0.

VLM-based observer that inspects video clips, determines relevance,
and identifies logical gaps for further investigation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

from PIL import Image

from .llm_client import LLMClient


@dataclass
class RefinementPlan:
    """Plan for refining the search based on observation."""
    needs_more_info: bool = False
    missing_visual_keyword: Optional[str] = None


@dataclass
class ObservationResult:
    """Result of clip observation (v4.0)."""
    relevance: float  # 0.0-1.0 confidence score
    reasoning: str
    caption: Optional[str] = None
    refinement_plan: RefinementPlan = field(default_factory=RefinementPlan)
    ocr_text: str = ""  # New field for V4.1
    
    # Backward compatibility: offset is deprecated in v4.0
    offset: float = 0.0


class Observer:
    """
    VLM Observer for v4.0 with logical gap analysis.
    
    Key changes from v3.0:
    - No longer predicts time offsets (graph handles spatial reasoning)
    - Outputs refinement_plan with missing_visual_keyword for gap filling
    - Focus on option verification and logical connection
    """
    
    SYSTEM_PROMPT = """Role: Video Logic Analyst.
Task: Analyze the clip to verify options or identify missing logic, and produce a comprehensive caption.

You will see frames from a video chunk. Your job is:
1. **Verification**: Does this clip support or contradict any option?
2. **Caption**: Describe what is visible in this chunk as completely as possible.
3. **Logical Gap**: If the answer is not here, what SPECIFIC visual event is missing?

Return ONLY this JSON (no other text):
{
    "relevance": <0.0-1.0>,
    "caption": "<1-2 short sentences>",
    "refinement_plan": {"needs_more_info": <true/false>, "missing_visual_keyword": "<keyword or null>"}
}

STOP after closing }. Maximum 200 words total.

Scoring guide:
- 0.0: Completely irrelevant (wrong topic/scene).
- 0.1-0.3: Same topic but no specific evidence.
- 0.4-0.6: Contains related elements (relevant objects, text on screen).
- 0.7-0.8: Strong evidence supporting or contradicting an option.
- 0.9-1.0: Definitive answer visible.

IMPORTANT:
- missing_visual_keyword should be CONCRETE VISUAL phrases only (e.g., "collision car crash", "person celebrating").
- Do NOT guess timestamps or time codes.
- Always provide a caption describing what you observe.
- The caption MUST include:
  - Visual details: objects, actions, attributes, colors, counts, spatial relations, on-screen text.
  - Event semantics: what is happening, roles/relationships, cause-and-effect if implied.
- If the user prompt includes **Focus Keywords** / **Focus Semantic Queries**, explicitly address them if visible.
- If there is on-screen text, quote it as accurately as possible.
- If evidence is unclear, say it is unclear and describe what IS visible; do not invent events.
- Output in English only.

CAPTION CHECKLIST (follow step by step):
1) WHO/WHAT is visible (people, objects, scene type)
2) WHAT ACTIONS are happening (verbs, interactions)
3) KEY DETAILS (colors, counts, attributes, positions, tools)
4) TEXT ON SCREEN (titles, labels, numbers, names)
5) EVENT MEANING (what the scene suggests or accomplishes)

Good caption example:
"A man in a gray suit speaks directly to camera while courtroom movie clips play in a small screen behind him. He gestures toward the screen as a uniformed officer appears in the clip. On-screen text says 'SUBSCRIBE' and 'CHECK OUT THIS VIDEO'. The segment explains legal concepts while showing examples from films."

Bad caption example:
A man is talking about a video."""

    # V4.7: When use_vlm_relevance is disabled, we do NOT want subjective scoring.
    # We still keep the same JSON schema for backward compatibility, but force relevance=0.0.
    SYSTEM_PROMPT_TEXT_ONLY = """Role: Video Captioner & Logic Analyst.
Task: Analyze the clip and output a comprehensive, query-oriented caption plus what visual evidence is missing.

You will see frames from a video chunk. Your job is:
1. **Caption**: Describe what is visible in this chunk as completely as possible.
2. **Logical Gap**: If the answer is not here, what SPECIFIC visual event is missing?

IMPORTANT:
- Set "relevance" to 0.0 ALWAYS. Do NOT provide subjective confidence scores.
- The user prompt may include **Focus Keywords** and **Focus Semantic Queries**.
  You MUST describe these parts clearly if they are visible (names, objects, actions, relationships, numbers).
- If there is on-screen text, quote it as accurately as possible (prefer verbatim fragments over paraphrase).
- Avoid vague phrases like "someone" or "something" when a concrete description is possible.
- Caption should cover: WHO/WHAT/WHERE, key actions, key objects, attributes/colors, counts, spatial relations,
  notable text/diagrams, and any changes across frames; also state the event-level semantics when possible.
- If evidence is unclear, state what is visible and what is missing; do not hallucinate.
- Output in English only.
- Output ONLY valid JSON in the exact schema below.

{
  "relevance": 0.0,
  "reasoning": "<brief analysis>",
  "caption": "<2-5 sentences describing this chunk; prioritize details relevant to the focus keywords/semantic queries>",
  "refinement_plan": {
    "needs_more_info": <true/false>,
    "missing_visual_keyword": "<VISUAL noun/verb phrase if needs_more_info is true, otherwise null>"
  }
}"""

    PROMPT_TEMPLATE = """Query: {query}

Analyze these video frames and determine:
1. Relevance to the query (0.0-1.0)
2. What is visible in this clip (caption)
3. If we need more info, what specific visual should we look for?

Output ONLY valid JSON:"""

    def __init__(self, llm_client: LLMClient):
        """Initialize Observer."""
        self.llm_client = llm_client
        
        # OCR support (optional)
        self.ocr_reader = None
        try:
            import easyocr
            import warnings
            warnings.filterwarnings("ignore")
            self.ocr_reader = easyocr.Reader(['en'], gpu=True)
            print("[Observer] EasyOCR initialized for v4.0")
        except Exception:
            pass

    def inspect(
        self,
        frames: List[Image.Image],
        query: str,
        need_relevance: bool = True
    ) -> ObservationResult:
        """
        Inspect a video clip for query relevance (v4.0).
        
        Args:
            frames: List of PIL.Image frames from the clip.
            query: Query to check relevance for.
        
        Returns:
            ObservationResult with relevance, caption, and refinement_plan.
        """
        if len(frames) == 0:
            return ObservationResult(
                relevance=0.0,
                reasoning="No frames provided",
                refinement_plan=RefinementPlan(needs_more_info=True, missing_visual_keyword="any video content")
            )
        
        # OCR context extraction
        ocr_context = self._extract_ocr(frames) if self.ocr_reader else ""
        
        prompt = self.PROMPT_TEMPLATE.format(query=query)
        if ocr_context:
            prompt += f"\n[OCR DETECTED]: {ocr_context}"
        
        try:
            result = self.llm_client.chat_json(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT if need_relevance else self.SYSTEM_PROMPT_TEXT_ONLY,
                images=frames
            )
            
            relevance = float(result.get("relevance", 0.0))
            relevance = max(0.0, min(1.0, relevance))
            if not need_relevance:
                relevance = 0.0
            
            # Note: reasoning field removed from prompt for conciseness
            reasoning = ""  # No longer used
            caption = result.get("caption")
            if caption:
                caption = str(caption)
            
            # Parse refinement plan
            plan_data = result.get("refinement_plan", {})
            if isinstance(plan_data, dict):
                refinement_plan = RefinementPlan(
                    needs_more_info=bool(plan_data.get("needs_more_info", False)),
                    missing_visual_keyword=plan_data.get("missing_visual_keyword")
                )
            else:
                refinement_plan = RefinementPlan()
            
            # Relevance calibration (ONLY when we actually need relevance):
            # if caption matches query keywords but relevance is low, give a minimum score
            # to prevent completely ignoring relevant frames.
            if need_relevance and relevance < 0.3 and caption:
                caption_lower = caption.lower()
                query_words = [w.lower().strip(".,?!") for w in query.split() if len(w) > 3]
                matches = sum(1 for w in query_words if w in caption_lower)
                if matches >= 2:
                    relevance = max(relevance, 0.3)
                    reasoning += " [Calibrated: caption matches query keywords]"
            
            return ObservationResult(
                relevance=relevance,
                reasoning=reasoning,
                caption=caption,
                refinement_plan=refinement_plan,
                ocr_text=ocr_context
            )
        
        except Exception as e:
            print(f"Observation failed: {e}")
            return ObservationResult(
                relevance=0.0,
                reasoning=f"Error: {str(e)[:100]}",
                refinement_plan=RefinementPlan(needs_more_info=True, missing_visual_keyword="retry observation"),
                ocr_text=ocr_context
            )
    
    def _extract_ocr(self, frames: List[Image.Image]) -> str:
        """Extract text from frames using OCR."""
        if not self.ocr_reader:
            return ""
        
        try:
            import numpy as np
            # Sample 3 frames
            indices = sorted(set([0, len(frames)//2, len(frames)-1]))
            detected = []
            
            for idx in indices:
                if idx < len(frames):
                    arr = np.array(frames[idx])
                    res = self.ocr_reader.readtext(arr, detail=0)
                    if res:
                        valid = [t for t in res if len(t.strip()) > 3]
                        detected.extend(valid)
            
            if detected:
                unique = list(dict.fromkeys(detected))[:30]
                return ", ".join(unique)
        except Exception:
            pass
        
        return ""
    
    def batch_inspect(
        self,
        clips: List[List[Image.Image]],
        query: str
    ) -> List[ObservationResult]:
        """Inspect multiple clips."""
        return [self.inspect(frames, query) for frames in clips]
