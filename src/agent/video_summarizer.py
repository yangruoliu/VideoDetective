"""
Video Summarizer for VideoDetective v4.3.

Generates event skeleton (timeline descriptions) from video frames
for semantic recall enhancement.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import re

from PIL import Image

from .llm_client import LLMClient


@dataclass
class EventSegment:
    """Single event segment in the video."""
    start_time: float       # Start time in seconds
    end_time: float         # End time in seconds
    description: str        # Detailed event description
    chunk_indices: List[int] = field(default_factory=list)  # Mapped graph node indices


@dataclass
class EventSkeleton:
    """Video event skeleton containing timeline descriptions."""
    events: List[EventSegment]  # List of events
    raw_text: str              # Raw VLM output
    video_duration: float = 0.0  # Total video duration in seconds


class VideoSummarizer:
    """
    Video event skeleton generator for v4.3 multi-route recall.
    
    Generates detailed timeline descriptions from sampled keyframes,
    which are used for semantic-based retrieval alongside visual features.
    """
    
    SYSTEM_PROMPT = """You are a professional video event analyst. Your task is to observe sampled keyframes and generate **event timeline** descriptions.

## Core Principle: Describe events, not visual features

Correct examples (event-oriented):
- "A scientist begins a controlled experiment and sets up two sample groups"
- "The host introduces the guest and leads into today's topic"
- "A worker finishes assembling the first part and begins a quality check"

Wrong examples (visual-feature oriented - forbidden):
- "A person wearing white stands next to a table"
- "The background is blue with several objects"
- "A person is looking at something"

## Output Requirements

Format: one event per line, formatted as `[MM:SS-MM:SS] event description`

What to emphasize:
1. Actions and behaviors: who does what? for what purpose?
2. Event progress: what stage the process has reached
3. Causality: what the action leads to
4. Key information: names, numbers, places mentioned
5. State changes: from state A to state B
6. Narrative content: if the footage is narration/news/explanation, state clearly what it is talking about

Forbidden:
- Do not describe colors, clothing, or other visual appearance
- Do not describe static scenes like "the background has ..."
- Do not describe vague actions such as "doing something"
- Do not treat purely visual details as event outcomes

Example output:
[00:00-00:15] The speaker begins by introducing the purpose of the experiment: to verify how a catalyst affects reaction speed
[00:15-00:40] Experimental materials are prepared, and two identical samples are taken as control and experimental groups
[00:40-01:20] The catalyst is added to the experimental group while a timer is started to record reaction time
[01:20-02:00] The progress of both samples is observed and recorded; the experimental group is clearly faster
[02:00-02:30] The conclusion summarizes that the catalyst increases the reaction rate by about three times"""

    PROMPT_TEMPLATE = """Please analyze the following video frame sequence and generate an event timeline.

Video information:
- Sampled frames: {sample_count} frames
- Estimated duration: about {duration:.1f} seconds

Please write one line for each identifiable **event**, in the format: [MM:SS-MM:SS] event description

Important reminders:
- Describe "what happened," not "what was seen"
- Focus on actions, behaviors, progress, and results
- Ignore visual appearance (colors, clothing, background layout, etc.)
- If it is news/lecture/explanation, state clearly what viewpoint or conclusion is presented
- Make event descriptions as specific as possible, including people/organizations/places/actions/results

Please output the event timeline directly:"""

    def __init__(self, llm_client: LLMClient):
        """Initialize Video Summarizer."""
        self.llm_client = llm_client
    
    def summarize(
        self,
        frames: List[Image.Image],
        timestamps: List[float],
        chunks: List[Tuple[int, int]],
        fps: float = 1.0
    ) -> EventSkeleton:
        """
        Generate video event skeleton from sampled keyframes.
        
        Args:
            frames: Sampled keyframes (typically 32 frames).
            timestamps: Timestamp in seconds for each frame.
            chunks: Graph node ranges [(start_frame, end_frame), ...].
            fps: Frames per second of the original video.
        
        Returns:
            EventSkeleton containing list of event segments.
        """
        if len(frames) == 0:
            return EventSkeleton(events=[], raw_text="", video_duration=0.0)
        
        # Compute video duration
        video_duration = max(timestamps) if timestamps else 0.0
        
        prompt = self.PROMPT_TEMPLATE.format(
            sample_count=len(frames),
            duration=video_duration
        )
        
        try:
            # Call VLM with images
            raw_text = self.llm_client.chat_with_images(
                prompt=prompt,
                images=frames,
                system_prompt=self.SYSTEM_PROMPT
            )
            
            # Parse output
            events = self._parse_events(raw_text, video_duration)
            
            # Map events to graph chunks
            self._map_events_to_chunks(events, chunks, fps)
            
            return EventSkeleton(
                events=events,
                raw_text=raw_text,
                video_duration=video_duration
            )
        
        except Exception as e:
            print(f"[VideoSummarizer] Error generating skeleton: {e}")
            return EventSkeleton(events=[], raw_text=f"Error: {e}", video_duration=video_duration)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
    
    def _parse_time(self, time_str: str) -> float:
        """Parse MM:SS or HH:MM:SS to seconds."""
        parts = time_str.strip().split(":")
        try:
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                return 0.0
        except ValueError:
            return 0.0
    
    def _parse_events(
        self,
        raw_text: str,
        video_duration: float
    ) -> List[EventSegment]:
        """Parse VLM output into EventSegment list."""
        events = []
        
        # Pattern: [MM:SS-MM:SS] Description
        # Also handle [MM:SS - MM:SS] with spaces
        pattern = r'\[(\d{1,2}:\d{2})\s*[-–—]\s*(\d{1,2}:\d{2})\]\s*(.+?)(?=\n\[|\n*$)'
        
        matches = re.findall(pattern, raw_text, re.DOTALL)
        
        for start_str, end_str, description in matches:
            start_time = self._parse_time(start_str)
            end_time = self._parse_time(end_str)
            
            # Sanity check
            if end_time <= start_time:
                end_time = start_time + 5.0  # Default 5 second segment
            
            if end_time > video_duration + 10:  # Allow some tolerance
                end_time = video_duration
            
            # Clean description
            description = description.strip()
            if description:
                events.append(EventSegment(
                    start_time=start_time,
                    end_time=end_time,
                    description=description
                ))
        
        # If no events parsed, try line-by-line fallback
        if not events:
            events = self._fallback_parse(raw_text, video_duration)
        
        return events
    
    def _fallback_parse(
        self,
        raw_text: str,
        video_duration: float
    ) -> List[EventSegment]:
        """Fallback parsing when regex fails."""
        events = []
        lines = raw_text.strip().split("\n")
        
        # Distribute time evenly
        if len(lines) == 0:
            return events
        
        segment_duration = video_duration / max(len(lines), 1)
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 10:
                continue
            
            # Remove any leading markers like "1.", "-", "*"
            line = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
            
            if line:
                events.append(EventSegment(
                    start_time=i * segment_duration,
                    end_time=(i + 1) * segment_duration,
                    description=line
                ))
        
        return events
    
    def _map_events_to_chunks(
        self,
        events: List[EventSegment],
        chunks: List[Tuple[int, int]],
        fps: float
    ):
        """Map events to graph chunk indices based on time overlap."""
        for event in events:
            event.chunk_indices = []
            
            for chunk_idx, (start_frame, end_frame) in enumerate(chunks):
                # Convert frame range to time range
                chunk_start_time = start_frame / fps if fps > 0 else 0
                chunk_end_time = end_frame / fps if fps > 0 else 0
                
                # Check if there's overlap
                if self._time_overlap(
                    event.start_time, event.end_time,
                    chunk_start_time, chunk_end_time
                ):
                    event.chunk_indices.append(chunk_idx)
    
    def _time_overlap(
        self,
        start1: float, end1: float,
        start2: float, end2: float
    ) -> bool:
        """Check if two time ranges overlap."""
        return start1 < end2 and start2 < end1
    
    def get_caption_for_chunk(
        self,
        skeleton: EventSkeleton,
        chunk_idx: int
    ) -> Optional[str]:
        """Get the event description that covers a specific chunk."""
        for event in skeleton.events:
            if chunk_idx in event.chunk_indices:
                return event.description
        return None
    
    def build_chunk_caption_map(
        self,
        skeleton: EventSkeleton
    ) -> Dict[int, str]:
        """Build a mapping from chunk index to event description."""
        caption_map: Dict[int, str] = {}
        
        for event in skeleton.events:
            for chunk_idx in event.chunk_indices:
                # If multiple events cover the same chunk, concatenate
                if chunk_idx in caption_map:
                    caption_map[chunk_idx] += " | " + event.description
                else:
                    caption_map[chunk_idx] = event.description
        
        return caption_map
