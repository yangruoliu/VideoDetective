"""
VideoDetective v4.0 Pipeline (Graph-Constellation).

Implements semantic graph-based video understanding with:
- Semantic chunking for discrete node representation
- Multi-channel forced patrol for option verification
- Matrix belief propagation for evidence aggregation
- Graph-NMS for final frame selection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np
import re
import json
import os
from PIL import Image

from config.settings import get_settings, Settings
from .perception.encoder import VideoEncoder
from .perception.asr import get_asr_extractor
from .engine.graph_engine import GraphBeliefEngine
from .agent.llm_client import LLMClient
from .agent.query_processor import QueryProcessor, QueryResultV5
from .agent.observer import Observer
from .agent.video_summarizer import VideoSummarizer, EventSkeleton
from .utils.video_io import load_video


@dataclass
class SolveResult:
    """Result of VideoDetective.solve()."""
    answer: str
    debug_info: Dict[str, Any] = field(default_factory=dict)


class VideoDetective:
    """
    VideoDetective v4.0: Graph-Constellation Search.
    
    Architecture:
        1. Semantic Chunking: Divide video into coherent segments
        2. Graph Construction: Build similarity + temporal affinity matrix
        3. Forced Patrol: Visit anchors for each option and temporal stage
        4. Matrix Propagation: Update beliefs across graph
        5. Graph-NMS: Select diverse, high-evidence frames
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        verbose: bool = True
    ):
        """Initialize VideoDetective v4.0 pipeline."""
        self.settings = settings or get_settings()
        self.verbose = verbose
        
        # Initialize encoder
        self._log("Initializing VideoEncoder...")
        self.encoder = VideoEncoder(
            siglip_model_id=self.settings.encoder.siglip_model_id,
            clip_local_path=self.settings.encoder.clip_local_path,
            cache_dir=self.settings.encoder.cache_dir
        )
        
        # Initialize VLM client
        self._log("Initializing VLM client...")
        self.llm_client = LLMClient(
            api_key=self.settings.qwen.api_key,
            base_url=self.settings.qwen.base_url,
            model=self.settings.qwen.model,
            max_tokens=self.settings.qwen.max_tokens,
            temperature=self.settings.qwen.temperature,
            timeout=self.settings.qwen.timeout,
            max_frames_per_call=self.settings.qwen.max_frames_per_call
        )
        
        # Initialize text LLM client (use separate API settings if configured)
        self._log("Initializing Text LLM client...")
        text_llm_api_key = self.settings.text_llm.api_key or self.settings.qwen.api_key
        text_llm_base_url = self.settings.text_llm.base_url or self.settings.qwen.base_url
        self.text_llm_client = LLMClient(
            api_key=text_llm_api_key,
            base_url=text_llm_base_url,
            model=self.settings.text_llm.model,
            max_tokens=self.settings.text_llm.max_tokens,
            temperature=self.settings.text_llm.temperature,
            timeout=self.settings.qwen.timeout,
            max_frames_per_call=1
        )
        
        # Initialize agents
        self.query_processor = QueryProcessor(self.text_llm_client)
        self.observer = Observer(self.llm_client)
        
        # V4.3: Video Summarizer for event skeleton generation
        self.video_summarizer = VideoSummarizer(self.llm_client)
    
    def _log(self, message: str):
        """Print log message if verbose."""
        if self.verbose:
            print(f"[VideoDetective v4.0] {message}")
    
    def solve(
        self,
        video_path: str,
        query: str,
        max_steps: int = 10,
        total_budget: int = 32
    ) -> SolveResult:
        """
        Solve a video question using Graph-Constellation Search.
        
        Args:
            video_path: Path to video file.
            query: Question with options.
            max_steps: Maximum VLM observation steps.
            total_budget: Total frames for final answer.
        
        Returns:
            SolveResult with answer and debug_info.
        """
        debug_info: Dict[str, Any] = {
            "video_path": video_path,
            "query": query,
            "version": "v4.0-graph-constellation"
        }
        
        # V4.1.2: Track global word frequencies for IDF weighting
        global_word_freq: Dict[str, int] = {}
        # V4.2: Pre-seed IDF with common video stop words
        common_visual_stopwords = {
            "video", "clip", "scene", "frame", "screen", "visual", "show", "showing",
            "image", "picture", "footage", "camera", "shot", "view", "angle",
            "person", "people", "man", "woman", "guy", "girl", "someone",
            "background", "foreground", "left", "right", "center", "top", "bottom"
        }
        for sw in common_visual_stopwords:
            global_word_freq[sw] = 100  # Seed with high count to suppress

        
        # ===== 1. PREPROCESSING =====
        self._log("Step 1: Preprocessing...")
        
        # 1.1 Encode video
        self._log("  Encoding video frames...")
        feats, frames = self.encoder.encode(video_path, fps=self.settings.default_fps)
        T = len(frames)
        debug_info["num_frames"] = T
        self._log(f"  Extracted {T} frames")
        
        if T == 0:
            return SolveResult(answer="Error", debug_info={"error": "No frames extracted"})
        
        # 1.2 Semantic chunking
        self._log("  Chunking video into semantic segments...")
        # min_len configurable via env var (default 10 for long videos, 4 for short/medium)
        chunk_min_len = int(os.getenv("VIDEODETECTIVE_CHUNK_MIN_LEN", "10"))
        chunks, node_feats = self.encoder.chunk_video(feats, sim_threshold=0.82, min_len=chunk_min_len)
        self._log(f"  Using chunk min_len={chunk_min_len}")
        K = len(chunks)
        debug_info["num_chunks"] = K
        debug_info["chunks"] = [(int(s), int(e)) for s, e in chunks]
        self._log(f"  Created {K} semantic chunks")
        
        if K == 0:
            return SolveResult(answer="Error", debug_info={"error": "No chunks created"})
        
        # 1.2.5 V4.4: Load ASR data
        asr_by_frame = self._load_asr_data(video_path, self.settings.default_fps)
        debug_info["asr_available"] = len(asr_by_frame) > 0
        
        # 1.3 Process query with multi-channel decomposition
        self._log("  Processing query...")
        # V4.3: Use process_v5 for multi-route recall if enabled
        if self.settings.enable_multi_route_recall:
            plan = self.query_processor.process_v5(query)
            debug_info["semantic_queries"] = plan.semantic_queries
            debug_info["general_semantic_query"] = plan.general_semantic_query
            self._log(f"  Semantic queries: {list(plan.semantic_queries.keys())}")
        else:
            plan = self.query_processor.process(query)
        debug_info["option_keywords"] = plan.option_keywords
        debug_info["vlm_query"] = plan.vlm_query
        self._log(f"  Option keywords: {plan.option_keywords}")

        # V4.3.2: Support options beyond A-D (A/E/...).
        # Infer the option id set from query processing outputs.
        option_ids: List[str] = self._infer_option_ids(plan, query=query)
        debug_info["option_ids"] = option_ids

        # Auto-scale max_steps by option count (4 options -> 10 steps; each extra option +1)
        auto_max_steps = self._compute_auto_max_steps(option_ids)
        debug_info["option_count"] = len(option_ids)
        debug_info["max_steps_requested"] = max_steps
        debug_info["max_steps_auto"] = auto_max_steps
        if max_steps is None or int(max_steps) <= 0 or int(max_steps) < auto_max_steps:
            max_steps = auto_max_steps
        debug_info["max_steps_used"] = max_steps
        if debug_info["max_steps_used"] != debug_info["max_steps_requested"]:
            self._log(
                f"  Auto max_steps: {debug_info['max_steps_requested']} -> "
                f"{debug_info['max_steps_used']} for {debug_info['option_count']} options"
            )

        # V4.4: Log ALL rewritten keywords and semantic queries for debugging
        def _dump_json_block(tag: str, obj: Any) -> None:
            try:
                import json as _json
                self._log(f"  {tag}:")
                print(f"----- {tag}_BEGIN -----")
                print(_json.dumps(obj, ensure_ascii=False, indent=2, default=str))
                print(f"----- {tag}_END -----")
            except Exception:
                pass

        # Normalize/clean for logging (do not mutate plan)
        cleaned_query_keywords: List[str] = self._dedup_keywords(getattr(plan, "query_keywords", []) or [])
        cleaned_option_keywords: Dict[str, List[str]] = {}
        for opt in option_ids:
            kws = []
            try:
                kws = (plan.option_keywords or {}).get(opt, []) or []
            except Exception:
                kws = []
            cleaned_option_keywords[opt] = self._dedup_keywords(kws)
        cleaned_semantic_queries: Dict[str, str] = {}
        if self.settings.enable_multi_route_recall:
            sq_map = getattr(plan, "semantic_queries", {}) or {}
            for opt in option_ids:
                cleaned_semantic_queries[opt] = str(sq_map.get(opt, "") or "").strip()
        general_semantic_query = str(getattr(plan, "general_semantic_query", "") or "").strip()

        def _build_observer_query(opt_id: Optional[str] = None) -> str:
            """
            Build an Observer query that includes focus keywords / semantic queries to guide caption generation.
            Keep this compact to avoid blowing up prompt length.
            """
            lines = [str(plan.vlm_query or "").strip() or str(query)]
            # Query-level entities
            if cleaned_query_keywords:
                lines.append("Focus Keywords (query): " + ", ".join(cleaned_query_keywords[:20]))
            # Option-specific hints (if any)
            if opt_id:
                ok = cleaned_option_keywords.get(opt_id, []) or []
                if ok:
                    lines.append(f"Focus Keywords (option {opt_id}): " + ", ".join(ok[:20]))
                if self.settings.enable_multi_route_recall:
                    sq = cleaned_semantic_queries.get(opt_id, "") or ""
                    if sq:
                        lines.append(f"Focus Semantic Query (option {opt_id}): {sq}")
            else:
                if self.settings.enable_multi_route_recall and general_semantic_query:
                    lines.append(f"Focus Semantic Query (general): {general_semantic_query}")
            return "\n".join(lines).strip()

        debug_info["query_keywords"] = cleaned_query_keywords
        debug_info["option_keywords_rewritten"] = cleaned_option_keywords
        if self.settings.enable_multi_route_recall:
            debug_info["semantic_queries_rewritten"] = cleaned_semantic_queries
            debug_info["general_semantic_query_rewritten"] = general_semantic_query

        _dump_json_block("QUERY_KEYWORDS_REWRITTEN", cleaned_query_keywords)
        _dump_json_block("OPTION_KEYWORDS_REWRITTEN", cleaned_option_keywords)
        if self.settings.enable_multi_route_recall:
            _dump_json_block("SEMANTIC_QUERIES_REWRITTEN", cleaned_semantic_queries)
            _dump_json_block("GENERAL_SEMANTIC_QUERY_REWRITTEN", general_semantic_query)
        
        # 1.4 Question type flags (used for coverage bias)
        q_flags = self._classify_question(query)
        debug_info["question_types"] = q_flags

        # Extra uniform coverage frames for certain question types
        coverage_bonus = 0
        if q_flags["count"]:
            coverage_bonus = 12
        elif q_flags["temporal"] or q_flags["text"] or q_flags["audio"]:
            coverage_bonus = 8
        debug_info["coverage_bonus"] = coverage_bonus

        # Precompute frame relevance scores for chunk-level frame selection
        frame_scores = self._build_frame_scores(feats, plan, asr_by_frame=asr_by_frame)
        if frame_scores is not None:
            debug_info["frame_scores"] = {
                "min": float(np.min(frame_scores)),
                "max": float(np.max(frame_scores)),
                "mean": float(np.mean(frame_scores)),
            }
        
        # ===== 2. BUILD GRAPH =====
        self._log("Step 2: Building semantic graph...")
        graph = GraphBeliefEngine(node_feats, chunks, alpha=0.6, tau=30.0, top_k=8)

        # Filter text-only keywords for visual retrieval
        visual_option_keywords = self._filter_visual_option_keywords(plan.option_keywords or {})
        visual_query_keywords = self._filter_visual_keywords(getattr(plan, "query_keywords", []) or [])
        debug_info["visual_option_keywords"] = visual_option_keywords
        debug_info["visual_query_keywords"] = visual_query_keywords

        # Initialize multi-channel priors (V4.3: include query keywords in global prior)
        graph.init_multi_channel_prior(
            visual_option_keywords,
            self.encoder,
            query_keywords=visual_query_keywords
        )
        debug_info["initial_F"] = graph.F.tolist() if len(graph.F) > 0 else []
        debug_info["channel_scores"] = {k: v.tolist() for k, v in graph.channel_scores.items()}
        self._log(f"  Graph initialized with {K} nodes")
        
        # ===== 2.5 V4.3: EVENT SKELETON GENERATION =====
        skeleton = None
        if self.settings.enable_multi_route_recall:
            self._log("Step 2.5: Building event skeleton (V4.3)...")
            skeleton = self._build_event_skeleton(frames, chunks, graph)
            if skeleton and skeleton.events:
                debug_info["event_skeleton"] = [
                    {
                        "start_time": e.start_time,
                        "end_time": e.end_time,
                        "description": e.description[:100] + "..." if len(e.description) > 100 else e.description,
                        "chunks": e.chunk_indices
                    }
                    for e in skeleton.events
                ]
                self._log(f"  Generated {len(skeleton.events)} event segments")
                
                # Initialize semantic channel priors
                if hasattr(plan, "semantic_queries") and plan.semantic_queries:
                    graph.init_semantic_channel_prior(plan.semantic_queries, self.encoder)
                    debug_info["semantic_channel_scores"] = {
                        k: v.tolist() for k, v in graph.semantic_channel_scores.items()
                    }
                    self._log(f"  Semantic channel priors initialized")
            else:
                self._log("  Event skeleton generation skipped or failed")
        
        # ===== 3. FORCED PATROL =====
        self._log("Step 3: Forced patrol (anchor verification)...")
        visit_queue: List[int] = []
        
        # 3.0 Query-specific anchor (from question text)
        # V4.3: Support both keyword search and semantic query search
        query_anchor = None
        if visual_query_keywords:
            # Use keyword search (visual-only keywords)
            query_anchor = graph.search_by_text(visual_query_keywords, self.encoder, exclude_visited=True)
            self._log(f"  Query anchor (keyword): node {query_anchor}")
        elif self.settings.enable_multi_route_recall and general_semantic_query:
            # Fallback to semantic search using general_semantic_query
            semantic_scores = graph.compute_semantic_scores(general_semantic_query, self.encoder)
            if np.max(semantic_scores) > 0:
                query_anchor = int(np.argmax(semantic_scores))
                self._log(f"  Query anchor (semantic): node {query_anchor}")
        
        if query_anchor is not None and query_anchor not in visit_queue:
            visit_queue.append(query_anchor)
        if query_anchor is not None:
            debug_info["query_anchor"] = query_anchor

        # 3.1 Option anchors
        # V4.3: Use fused visual + semantic scores when multi-route recall is enabled
        if self.settings.enable_multi_route_recall and graph.semantic_channel_scores:
            alpha = self.settings.multi_route_alpha
            option_anchors = graph.get_fused_option_anchors(alpha=alpha, exclude_visited=True)
            self._log(f"  Using fused anchors (alpha={alpha})")
        else:
            option_anchors = graph.get_option_anchors(exclude_visited=True)
        for opt_id, node_idx in option_anchors.items():
            if node_idx not in visit_queue:
                visit_queue.append(node_idx)
                self._log(f"  Option anchor [{opt_id}]: node {node_idx}")
            else:
                self._log(f"  Option anchor [{opt_id}]: node {node_idx} (already queued)")
        debug_info["option_anchors"] = option_anchors
        
        # ===== 4. OBSERVE & PROPAGATE =====
        self._log("Step 4: Observing anchors...")
        observations: List[Dict] = []
        steps_used = 0
        last_refinement_keyword = None
        
        # V4.2: Track which node belongs to which option for dynamic retry
        node_to_option_map: Dict[int, str] = {}
        for opt_id, node_idx in option_anchors.items():
            node_to_option_map[node_idx] = opt_id

        # V4.2.2: Track BEST node per option for final guarantee
        # format: opt_id -> (node_idx, score)
        best_option_nodes: Dict[str, Tuple[int, float]] = {}
        for opt_id, node_idx in option_anchors.items():
            best_option_nodes[opt_id] = (node_idx, 0.0)

        # Track max score per option to trigger retry
        option_max_scores: Dict[str, float] = {opt: 0.0 for opt in option_ids}
        
        for node_idx in visit_queue:
            if steps_used >= max_steps:
                break
            
            # Get frames for this chunk
            chunk_frames = self._get_chunk_frames(node_idx, chunks, frames)
            if not chunk_frames:
                continue
                
            # Observe
            # Provide keyword/semantic-query hints so the caption is explicit and complete.
            opt_hint = node_to_option_map.get(node_idx)
            obs_query = _build_observer_query(opt_hint)
            obs = self.observer.inspect(chunk_frames, obs_query, need_relevance=getattr(self.settings, "use_vlm_relevance", False))
            
            # --- STRONG EVIDENCE LOOP (V4.1) ---
            # V4.4: Build combined text for IDF stats (Caption + OCR + ASR)
            # V4.7: Scoring can be text-only (caption/OCR/ASR) to avoid subjective VLM relevance.
            caption_text = (obs.caption or "")
            combined_text = caption_text
            if hasattr(obs, "ocr_text") and obs.ocr_text:
                # OCR is highly reliable for text-based questions
                combined_text += " [OCR] " + obs.ocr_text
            
            # V4.4: Add ASR text for this chunk
            chunk = chunks[node_idx] if node_idx < len(chunks) else None
            if chunk:
                asr_text = self._get_asr_for_chunk(chunk, asr_by_frame)
                if asr_text:
                    combined_text += " [ASR] " + asr_text
            
            # Update global word frequencies for IDF (V4.1.2)
            for tok in self._tokenize_for_tfidf(combined_text):
                global_word_freq[tok] = global_word_freq.get(tok, 0) + 1
            
            # V4.7: Compute per-channel text similarity (all normalized to [0, 1])
            ocr_text = (getattr(obs, "ocr_text", "") or "")
            asr_text_for_score = (asr_text or "") if "asr_text" in locals() else ""

            caption_score = self._calculate_strong_evidence_score(
                caption_text,
                plan.option_keywords,
                getattr(plan, "query_keywords", []),
                getattr(plan, "semantic_queries", None),
                global_word_freq,
                source="caption"
            )
            ocr_score = self._calculate_strong_evidence_score(
                ocr_text,
                plan.option_keywords,
                getattr(plan, "query_keywords", []),
                getattr(plan, "semantic_queries", None),
                global_word_freq,
                source="ocr"
            )
            asr_score = self._calculate_strong_evidence_score(
                asr_text_for_score,
                plan.option_keywords,
                getattr(plan, "query_keywords", []),
                getattr(plan, "semantic_queries", None),
                global_word_freq,
                source="asr"
            )

            text_score = float(max(caption_score, ocr_score, asr_score))

            # Final score
            if getattr(self.settings, "use_vlm_relevance", False):
                final_relevance = float(max(obs.relevance, text_score))
            else:
                final_relevance = float(text_score)

            # Detailed logging for score calculation
            self._log(
                f"  Node {node_idx}: VLM={obs.relevance:.2f}, cap={caption_score:.2f}, OCR={ocr_score:.2f}, ASR={asr_score:.2f} "
                f"-> TextMax={text_score:.2f} -> Final={final_relevance:.2f} (use_vlm_relevance={getattr(self.settings,'use_vlm_relevance', False)})"
            )
            if obs.caption:
                self._log(f"    Caption: {obs.caption[:80]}...")
            # V4.4: Log OCR/ASR evidence snippets (truncated) to verify combined_text integration
            try:
                ocr_snip = (getattr(obs, "ocr_text", "") or "").strip()
                asr_snip = (asr_text or "").strip() if "asr_text" in locals() else ""
                if ocr_snip:
                    self._log(f"    OCR: {ocr_snip[:80]}...")
                if asr_snip:
                    self._log(f"    ASR: {asr_snip[:80]}...")
            except Exception:
                pass
            
            # Update graph
            graph.update(node_idx, final_relevance)
            
            # Update option tracking
            if node_idx in node_to_option_map:
                opt_id = node_to_option_map[node_idx]
                option_max_scores[opt_id] = max(option_max_scores[opt_id], final_relevance)
                
                # V4.2.2: Update best node if this one is better
                current_best_node, current_best_score = best_option_nodes.get(opt_id, (None, -1.0))
                if final_relevance > current_best_score:
                    best_option_nodes[opt_id] = (node_idx, final_relevance)
                
                # --- V4.2 DYNAMIC RETRY ---
                # If this was the primary anchor for an option and it failed (relevance < 0.2),
                # try to fetch the next best candidate immediately.
                if final_relevance < 0.2:
                    self._log(f"    [Dynamic Retry] Option {opt_id} primary anchor failed (score {final_relevance:.2f}). Finding replacement...")
                    # V4.3: Use fused candidates when multi-route recall is enabled
                    if self.settings.enable_multi_route_recall and graph.semantic_channel_scores:
                        alpha = self.settings.multi_route_alpha
                        candidates = graph.get_fused_candidates_for_option(opt_id, alpha=alpha, exclude_visited=True)
                    else:
                        candidates = graph.get_candidates_for_option(opt_id, exclude_visited=True)
                    if candidates:
                        next_node = candidates[0]
                        if next_node not in visit_queue:
                            # Insert next_node into queue immediately after current
                            current_q_idx = visit_queue.index(node_idx)
                            visit_queue.insert(current_q_idx + 1, next_node)
                            node_to_option_map[next_node] = opt_id # Track it too
                            self._log(f"    [Dynamic Retry] Added backup node {next_node} for Option {opt_id}")

            # Record observation
            observations.append({
                "node": node_idx,
                # IMPORTANT: downstream fallback uses observations[*]["relevance"].
                # It must reflect the fused score (final_relevance), not raw VLM.
                "relevance": float(final_relevance),
                "final_relevance": float(final_relevance),
                "vlm_relevance": float(obs.relevance),
                "text_score": float(text_score),
                "caption_score": float(caption_score),
                "ocr_score": float(ocr_score),
                "asr_score": float(asr_score),
                "reasoning": obs.reasoning,
                "caption": obs.caption,
                "ocr_text": getattr(obs, "ocr_text", ""),
                "asr_text": (asr_text or "") if "asr_text" in locals() else "",
                "refinement_plan": {
                    "needs_more_info": obs.refinement_plan.needs_more_info,
                    "missing_visual_keyword": obs.refinement_plan.missing_visual_keyword
                }
            })
            
            # Track refinement keyword
            if obs.refinement_plan.needs_more_info and obs.refinement_plan.missing_visual_keyword:
                last_refinement_keyword = obs.refinement_plan.missing_visual_keyword
            
            steps_used += 1
        
        # ===== 5. GAP FILLING =====
        self._log("Step 5: Gap filling (remaining budget)...")
        remaining_budget = max_steps - steps_used
        
        while remaining_budget > 0:
            # Find next target
            if last_refinement_keyword:
                if str(last_refinement_keyword).lower().startswith("text:"):
                    self._log("  Gap fill keyword is text-only; skipping visual search")
                    last_refinement_keyword = None
                    remaining_budget -= 1
                    continue
                # Use missing keyword to search
                target = graph.search_by_text([last_refinement_keyword], self.encoder, exclude_visited=True)
                self._log(f"  Gap fill (keyword={last_refinement_keyword}): node {target}")
                last_refinement_keyword = None  # Reset
            else:
                # Select highest F * (1 - visited)
                F_masked = graph.F * (1 - graph.visited.astype(float))
                if np.max(F_masked) <= 0:
                    break
                target = int(np.argmax(F_masked))
                self._log(f"  Gap fill (highest belief): node {target}")
            
            # Observe target
            chunk_frames = self._get_chunk_frames(target, chunks, frames)
            if not chunk_frames:
                remaining_budget -= 1
                continue
            
            opt_hint = node_to_option_map.get(target)
            obs_query = _build_observer_query(opt_hint)
            obs = self.observer.inspect(chunk_frames, obs_query, need_relevance=getattr(self.settings, "use_vlm_relevance", False))
            
            # V4.4: Strong Evidence Logic for Gap Filling (Caption + OCR + ASR)
            caption_text = (obs.caption or "")
            combined_text = caption_text
            if hasattr(obs, "ocr_text") and obs.ocr_text:
                combined_text += " [OCR] " + obs.ocr_text
            
            # Add ASR for this chunk
            chunk = chunks[target] if target < len(chunks) else None
            if chunk:
                asr_text = self._get_asr_for_chunk(chunk, asr_by_frame)
                if asr_text:
                    combined_text += " [ASR] " + asr_text

            # Update global word frequencies for IDF (V4.1.2)
            for tok in self._tokenize_for_tfidf(combined_text):
                global_word_freq[tok] = global_word_freq.get(tok, 0) + 1

            # V4.7: Per-channel text similarity scores
            ocr_text = (getattr(obs, "ocr_text", "") or "")
            asr_text_for_score = (asr_text or "") if "asr_text" in locals() else ""

            caption_score = self._calculate_strong_evidence_score(
                caption_text,
                plan.option_keywords,
                getattr(plan, "query_keywords", []),
                getattr(plan, "semantic_queries", None),
                global_word_freq,
                source="caption"
            )
            ocr_score = self._calculate_strong_evidence_score(
                ocr_text,
                plan.option_keywords,
                getattr(plan, "query_keywords", []),
                getattr(plan, "semantic_queries", None),
                global_word_freq,
                source="ocr"
            )
            asr_score = self._calculate_strong_evidence_score(
                asr_text_for_score,
                plan.option_keywords,
                getattr(plan, "query_keywords", []),
                getattr(plan, "semantic_queries", None),
                global_word_freq,
                source="asr"
            )

            text_score = float(max(caption_score, ocr_score, asr_score))
            if getattr(self.settings, "use_vlm_relevance", False):
                final_relevance = float(max(obs.relevance, text_score))
            else:
                final_relevance = float(text_score)

            # Detailed logging
            self._log(
                f"  Node {target}: VLM={obs.relevance:.2f}, cap={caption_score:.2f}, OCR={ocr_score:.2f}, ASR={asr_score:.2f} "
                f"-> TextMax={text_score:.2f} -> Final={final_relevance:.2f} (use_vlm_relevance={getattr(self.settings,'use_vlm_relevance', False)})"
            )
            # Also log caption snippet for gap-fill nodes (to match anchor logging)
            if obs.caption:
                self._log(f"    Caption: {obs.caption[:80]}...")
            # V4.4: Log OCR/ASR evidence snippets (truncated) for gap-fill nodes too
            try:
                ocr_snip = (getattr(obs, "ocr_text", "") or "").strip()
                asr_snip = (asr_text or "").strip() if "asr_text" in locals() else ""
                if ocr_snip:
                    self._log(f"    OCR: {ocr_snip[:80]}...")
                if asr_snip:
                    self._log(f"    ASR: {asr_snip[:80]}...")
            except Exception:
                pass
            
            graph.update(target, final_relevance)
            
            # V4.2.2: Also update best option tracker if this gap-fill node is better
            # Note: gap fill nodes might not be explicitly mapped to an option, 
            # so we only update if it WAS mapped (e.g. from a prior run or if we map it now).
            # For strict correctness, we should only update if it is an option node.
            
            if target in node_to_option_map:
                 opt_id = node_to_option_map[target]
                 option_max_scores[opt_id] = max(option_max_scores[opt_id], float(final_relevance))
                 
                 # Debug potential key error
                 if opt_id not in best_option_nodes:
                     self._log(f"WARNING: opt_id {opt_id} not in best_option_nodes! Keys: {list(best_option_nodes.keys())}")
                     best_option_nodes[opt_id] = (target, float(final_relevance))
                 else:
                     current_best_node, current_best_score = best_option_nodes[opt_id]
                     if float(final_relevance) > current_best_score:
                         best_option_nodes[opt_id] = (target, float(final_relevance))
            
            observations.append({
                "node": target,
                # Keep consistent with Step 4: fused score drives fallback decision
                "relevance": float(final_relevance),
                "final_relevance": float(final_relevance),
                "vlm_relevance": float(obs.relevance),
                "text_score": float(text_score),
                "caption_score": float(caption_score),
                "ocr_score": float(ocr_score),
                "asr_score": float(asr_score),
                "reasoning": obs.reasoning,
                "caption": obs.caption,
                "ocr_text": getattr(obs, "ocr_text", ""),
                "asr_text": (asr_text or "") if "asr_text" in locals() else "",
                "refinement_plan": {
                    "needs_more_info": obs.refinement_plan.needs_more_info,
                    "missing_visual_keyword": obs.refinement_plan.missing_visual_keyword
                }
            })
            
            if obs.refinement_plan.needs_more_info and obs.refinement_plan.missing_visual_keyword:
                last_refinement_keyword = obs.refinement_plan.missing_visual_keyword
            
            remaining_budget -= 1
        
        debug_info["observations"] = observations
        debug_info["final_F"] = graph.F.tolist()
        debug_info["visited"] = graph.visited.tolist()

        # ===== 5.5 FALLBACK CHECK =====
        fallback, fallback_reason = self._should_fallback_to_uniform(observations)
        if fallback:
            self._log(f"Step 5.5: Fallback to uniform sampling ({fallback_reason})")
            debug_info["fallback_reason"] = fallback_reason
            frame_indices = self._uniform_frame_indices_with_focus(T, total_budget, query, q_flags)
            debug_info["final_frame_indices"] = frame_indices
            final_frames = [frames[i] for i in frame_indices if i < len(frames)]
            timestamps = [f"{idx / self.settings.default_fps:.1f}s" for idx in frame_indices]
            if len(final_frames) == 0:
                return SolveResult(answer="Error", debug_info={"error": "No frames selected"})
            evidence_map = {}
            if getattr(self.settings, "include_answer_evidence", True):
                evidence_map = self._build_evidence_map(observations, chunks, frame_indices)
                evidence_map = self._fill_fallback_evidence_map(
                    evidence_map,
                    observations,
                    chunks,
                    frame_indices,
                    plan,
                    asr_by_frame,
                    graph,
                    global_word_freq
                )
                self._log_evidence_map(evidence_map, timestamps, tag="EVIDENCE_MAP_FALLBACK")
            else:
                self._log("  Answer evidence attachment disabled (include_answer_evidence=False)")
            answer, reasoning = self._generate_answer(final_frames, query, timestamps, captions=(evidence_map or None))
            debug_info["answer"] = answer
            debug_info["reasoning"] = reasoning
            debug_info["answer_parse_ok"] = answer in option_ids
            # Always keep full raw output (even if parse succeeds)
            debug_info["vlm_raw_output"] = reasoning
            self._dump_full_model_output_to_log(reasoning, tag="VLM_RAW_OUTPUT_FALLBACK")
            self._log(f"Answer: {answer}")
            return SolveResult(answer=answer, debug_info=debug_info)
        
        # ===== 6. GRAPH-NMS SELECTION =====
        self._log("Step 6: Graph-NMS final selection...")
        
        # V4.2: Option Frame Guarantee - Force at least 1 chunk per option
        # V4.2.2 Fix: Use the BEST nodes found during patrol/retry, not just the initial anchors
        guaranteed_option_chunks = []
        
        # Log any updates from the original plan
        for opt_id, (node_idx, score) in best_option_nodes.items():
            original_node = option_anchors.get(opt_id)
            if original_node is not None and node_idx != original_node:
                self._log(f"  [Option Update] {opt_id} upgraded: node {original_node} -> node {node_idx} (score {score:.2f})")
        
        for opt_id, (node_idx, score) in best_option_nodes.items():
            if node_idx is not None and 0 <= node_idx < len(chunks):
                guaranteed_option_chunks.append(node_idx)
                self._log(f"  [Option Guarantee] {opt_id} -> chunk {node_idx} (score {score:.2f})")

        # V4.3.1: Query anchor guarantee (if query has content)
        query_anchor = debug_info.get("query_anchor")
        if query_anchor is not None and 0 <= query_anchor < len(chunks):
            if query_anchor not in guaranteed_option_chunks:
                guaranteed_option_chunks.append(query_anchor)
                self._log(f"  [Query Guarantee] query -> chunk {query_anchor}")

        # Select additional chunks using Graph-NMS (excluding already guaranteed ones)
        num_chunks_to_select = max(8, total_budget // 4)
        remaining_slots = num_chunks_to_select - len(guaranteed_option_chunks)
        
        if remaining_slots > 0:
            nms_selected = graph.graph_nms(num_select=remaining_slots + 4, suppression_factor=0.2)
            # Filter out already guaranteed chunks
            nms_selected = [c for c in nms_selected if c not in guaranteed_option_chunks][:remaining_slots]
        else:
            nms_selected = []
        
        # Combine: guaranteed option chunks first, then NMS-selected
        selected_chunks = guaranteed_option_chunks + nms_selected
        self._log(f"  Selected {len(selected_chunks)} chunks ({len(guaranteed_option_chunks)} guaranteed + {len(nms_selected)} NMS)")
        debug_info["selected_chunks"] = selected_chunks
        
        # Option-pack path for NOT-correct questions
        if q_flags["not_correct"]:
            option_pack = self._build_option_pack_frames(
                graph=graph,
                frames=frames,
                chunks=chunks,
                frame_scores=frame_scores,
                option_anchors=option_anchors,
                best_option_nodes=best_option_nodes,
                total_budget=total_budget
            )
            frame_indices = option_pack["frame_indices"]
            debug_info["option_pack_ranges"] = option_pack["option_ranges"]
            if option_pack.get("extra_range"):
                debug_info["option_pack_extra_range"] = option_pack["extra_range"]
            debug_info["final_frame_indices"] = frame_indices
            self._log(f"  Option-pack frames ({len(frame_indices)}): {frame_indices[:10]}...")

            # ===== 7. GENERATE ANSWER (OPTION PACK) =====
            self._log("Step 7: Generating final answer (option-pack)...")
            final_frames = [frames[i] for i in frame_indices if i < len(frames)]
            timestamps = [f"{idx / self.settings.default_fps:.1f}s" for idx in frame_indices]
            if len(final_frames) == 0:
                return SolveResult(answer="Error", debug_info={"error": "No frames selected"})
            evidence_map = {}
            if getattr(self.settings, "include_answer_evidence", True):
                evidence_map = self._build_evidence_map(observations, chunks, frame_indices)
                self._log_evidence_map(evidence_map, timestamps, tag="EVIDENCE_MAP_OPTION_PACK")
            else:
                self._log("  Answer evidence attachment disabled (include_answer_evidence=False)")
            answer, reasoning = self._generate_answer_option_pack(
                final_frames,
                query,
                timestamps,
                option_pack["option_ranges"],
                option_pack.get("extra_range"),
                captions=(evidence_map or None)
            )
            debug_info["answer"] = answer
            debug_info["reasoning"] = reasoning
            debug_info["answer_parse_ok"] = answer in option_ids
            # Always keep full raw output (even if parse succeeds)
            debug_info["vlm_raw_output"] = reasoning
            self._dump_full_model_output_to_log(reasoning, tag="VLM_RAW_OUTPUT_OPTION_PACK")
            self._log(f"Answer: {answer}")
            return SolveResult(answer=answer, debug_info=debug_info)

        # Sample frames from selected chunks
        frame_indices = graph.get_frames_from_chunks(
            selected_chunks,
            frames_per_chunk=4,
            frame_scores=frame_scores
        )
        
        # V4.3: Always mix at least 4 uniform frames for global context (more for temporal/count)
        # Support env override: MIN_UNIFORM_FRAMES or VIDEODETECTIVE_MIN_UNIFORM_FRAMES
        MIN_UNIFORM_FRAMES = int(os.getenv("MIN_UNIFORM_FRAMES", os.getenv("VIDEODETECTIVE_MIN_UNIFORM_FRAMES", "4")))
        uniform_count = MIN_UNIFORM_FRAMES + coverage_bonus
        uniform_count = min(uniform_count, max(MIN_UNIFORM_FRAMES, total_budget - 8))
        max_recall_frames = max(0, total_budget - uniform_count)
        
        # Get uniform frames for global context
        uniform_indices = self._uniform_frame_indices_with_focus(T, uniform_count, query, q_flags)
        
        # V4.3.1: Deduplicate frames by similarity to maximize information density
        # If a recalled frame is too similar to already selected frames, skip it
        SIM_THRESHOLD = 0.92  # Frames with cosine sim > this are considered duplicates
        
        final_frame_indices = []
        
        # Step 1: Add uniform frames first (they define the global context)
        for idx in uniform_indices:
            if idx < len(feats):
                final_frame_indices.append(idx)
        
        # Limit recalled frames budget (e.g., 24 if total_budget=32 and MIN_UNIFORM_FRAMES=8)
        if max_recall_frames > 0:
            frame_indices = frame_indices[:max_recall_frames]

        # Step 2: Add recalled frames, deduplicating by similarity
        dedup_skipped = 0
        for idx in frame_indices:
            if len(final_frame_indices) >= total_budget:
                    break
            if idx in final_frame_indices:
                continue
            if idx >= len(feats):
                continue
            
            # Check similarity against already selected frames
            is_duplicate = False
            for existing_idx in final_frame_indices:
                if existing_idx < len(feats):
                    sim = float(np.dot(feats[idx], feats[existing_idx]))
                    if sim > SIM_THRESHOLD:
                        is_duplicate = True
                        dedup_skipped += 1
                    break
            
            if not is_duplicate:
                final_frame_indices.append(idx)

        # Step 2.5: If not enough frames, relax dedup threshold and retry remaining recalled frames
        if len(final_frame_indices) < total_budget:
            RELAXED_SIM_THRESHOLD = 0.95
            for idx in frame_indices:
                if len(final_frame_indices) >= total_budget:
                    break
                if idx in final_frame_indices:
                    continue
                if idx >= len(feats):
                    continue
                is_duplicate = False
                for existing_idx in final_frame_indices:
                    if existing_idx < len(feats):
                        sim = float(np.dot(feats[idx], feats[existing_idx]))
                        if sim > RELAXED_SIM_THRESHOLD:
                            is_duplicate = True
                            break
                if not is_duplicate:
                    final_frame_indices.append(idx)
        
        # Step 3: If we still need more frames, use intelligent fallback selection
        # Select frames that are: 1) diverse from existing frames, 2) have high relevance scores
        if len(final_frame_indices) < total_budget:
            fallback_count = 0
            
            # Build candidate list: all frames except already selected
            candidates = []
            for i in range(len(feats)):
                if i in final_frame_indices:
                    continue
                # Avoid first and last 5% of frames (often low-info)
                if i < len(feats) * 0.05 or i > len(feats) * 0.95:
                    continue
                candidates.append(i)
            
            # Score each candidate by: diversity (low sim to existing) + relevance (frame_scores)
            candidate_scores = []
            for idx in candidates:
                # Diversity: minimum similarity to already selected frames
                min_sim = 1.0
                for existing_idx in final_frame_indices:
                    if existing_idx < len(feats):
                        sim = float(np.dot(feats[idx], feats[existing_idx]))
                        min_sim = min(min_sim, sim)
                
                # Clip min_sim to [0, 1] to ensure diversity_score is also in [0, 1]
                min_sim = max(0.0, min(1.0, min_sim))
                
                # Relevance: from frame_scores if available (already normalized to [0, 1])
                relevance = frame_scores[idx] if frame_scores is not None and idx < len(frame_scores) else 0.5
                
                # Combined score: prioritize diversity, then relevance
                # Lower similarity = higher diversity score (now guaranteed in [0, 1])
                diversity_score = 1.0 - min_sim
                combined_score = 0.7 * diversity_score + 0.3 * relevance
                
                candidate_scores.append((idx, combined_score, min_sim))
            
            # Sort by combined score (descending)
            candidate_scores.sort(key=lambda x: -x[1])
            
            # Add top candidates until budget is filled
            FALLBACK_SIM_THRESHOLD = 0.90
            for idx, score, min_sim in candidate_scores:
                if len(final_frame_indices) >= total_budget:
                    break
                # Still apply some similarity threshold
                if min_sim < FALLBACK_SIM_THRESHOLD:
                    final_frame_indices.append(idx)
                    fallback_count += 1
            
            if fallback_count > 0:
                self._log(f"    Added {fallback_count} fallback frames (diverse + high-score)")
        
        frame_indices = sorted(final_frame_indices)[:total_budget]
        
        debug_info["recall_frames_count"] = len([i for i in frame_indices if i not in uniform_indices])
        debug_info["uniform_frames_count"] = len([i for i in frame_indices if i in uniform_indices])
        debug_info["dedup_skipped"] = dedup_skipped

        debug_info["final_frame_indices"] = frame_indices
        self._log(f"  Final frames ({len(frame_indices)}): {debug_info['recall_frames_count']} recalled + {debug_info['uniform_frames_count']} uniform, {dedup_skipped} skipped by dedup")
        
        # ===== 7. GENERATE ANSWER =====
        self._log("Step 7: Generating final answer...")
        
        final_frames = [frames[i] for i in frame_indices if i < len(frames)]
        timestamps = [f"{idx / self.settings.default_fps:.1f}s" for idx in frame_indices]
        
        # Captions disabled per user request
        # caption_map = self._build_caption_map(observations, chunks, frame_indices)
        
        if len(final_frames) == 0:
            return SolveResult(answer="Error", debug_info={"error": "No frames selected"})
        
        evidence_map = {}
        if getattr(self.settings, "include_answer_evidence", True):
            evidence_map = self._build_evidence_map(observations, chunks, frame_indices)
            self._log_evidence_map(evidence_map, timestamps, tag="EVIDENCE_MAP_FINAL")
        else:
            self._log("  Answer evidence attachment disabled (include_answer_evidence=False)")
        answer, reasoning = self._generate_answer(final_frames, query, timestamps, captions=(evidence_map or None))
        debug_info["answer"] = answer
        debug_info["reasoning"] = reasoning
        debug_info["answer_parse_ok"] = answer in option_ids
        # Always keep full raw output (even if parse succeeds)
        debug_info["vlm_raw_output"] = reasoning
        self._dump_full_model_output_to_log(reasoning, tag="VLM_RAW_OUTPUT_FINAL")
        
        self._log(f"Answer: {answer}")
        
        return SolveResult(answer=answer, debug_info=debug_info)
    
    def _get_chunk_frames(
        self,
        node_idx: int,
        chunks: List[Tuple[int, int]],
        frames: List[Image.Image],
        max_frames: int = 9
    ) -> List[Image.Image]:
        """Get frames for a chunk (up to max_frames, uniformly sampled)."""
        if node_idx < 0 or node_idx >= len(chunks):
            return []
        
        start, end = chunks[node_idx]
        chunk_len = end - start + 1
        
        if chunk_len <= max_frames:
            indices = list(range(start, end + 1))
        else:
            step = chunk_len / max_frames
            indices = [int(start + i * step) for i in range(max_frames)]
        
        # If frames is a lazy container with batch decode support, use it for efficiency.
        try:
            if hasattr(frames, "get_batch"):
                pos = [i for i in indices if 0 <= i < len(frames)]
                return frames.get_batch(pos)  # type: ignore[attr-defined]
        except Exception:
            pass
        return [frames[i] for i in indices if 0 <= i < len(frames)]

    def _classify_question(self, query: str) -> Dict[str, bool]:
        """Lightweight question type detection for routing and coverage decisions."""
        q = (query or "").lower()
        return {
            "not_correct": bool(re.search(r"\b(not correct|incorrect|not true|false|is not)\b", q)),
            "temporal": bool(re.search(r"\b(first|last|before|after|then|earlier|later|beginning|end|start|finally|next|previous)\b", q)),
            "count": bool(re.search(r"\b(how many|number of|count|times)\b", q)),
            "text": bool(re.search(r"\b(text|title|word|name|label|subtitle|sign|caption|written|letter|logo|year|date)\b", q)),
            "audio": bool(re.search(r"\b(hear|said|says|narrator|voice|sound|music|audio|spoken|speak)\b", q)),
        }

    def _uniform_frame_indices(self, num_frames: int, count: int) -> List[int]:
        """Uniformly sample frame indices across the full video."""
        if num_frames <= 0 or count <= 0:
            return []
        if num_frames <= count:
            return list(range(num_frames))
        indices = [int(i) for i in np.linspace(0, num_frames - 1, count)]
        return sorted(set(indices))

    def _temporal_focus_indices(
        self,
        num_frames: int,
        query: str,
        count: int
    ) -> List[int]:
        """Pick extra indices for temporal/order questions to emphasize start/end/order cues."""
        if num_frames <= 0 or count <= 0:
            return []
        q = (query or "").lower()
        fps = float(self.settings.default_fps or 1.0)
        focus: List[int] = []

        def add_window(start_idx: int, end_idx: int, n: int) -> None:
            if n <= 0:
                return
            start_idx = max(0, min(num_frames - 1, start_idx))
            end_idx = max(0, min(num_frames - 1, end_idx))
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx
            window_len = end_idx - start_idx + 1
            if window_len <= 1:
                focus.append(start_idx)
                return
            window_indices = self._uniform_frame_indices(window_len, n)
            focus.extend([start_idx + i for i in window_indices])

        # Explicit time windows like "first 6 minutes" or "last 30 seconds"
        m_first = re.search(r"\bfirst\s+(\d+)\s*(seconds?|secs?|minutes?|mins?)\b", q)
        if m_first:
            val = int(m_first.group(1))
            unit = m_first.group(2)
            seconds = val * (60 if "min" in unit else 1)
            end_idx = int(seconds * fps)
            add_window(0, end_idx, min(4, count))

        m_last = re.search(r"\blast\s+(\d+)\s*(seconds?|secs?|minutes?|mins?)\b", q)
        if m_last:
            val = int(m_last.group(1))
            unit = m_last.group(2)
            seconds = val * (60 if "min" in unit else 1)
            start_idx = max(0, num_frames - int(seconds * fps) - 1)
            add_window(start_idx, num_frames - 1, min(4, count))

        # General start/end emphasis
        if re.search(r"\b(beginning|start|first|early)\b", q):
            end_idx = int(num_frames * 0.1)
            add_window(0, end_idx, min(3, count))
        if re.search(r"\b(end|last|final|finally|later)\b", q):
            start_idx = max(0, num_frames - max(1, int(num_frames * 0.1)))
            add_window(start_idx, num_frames - 1, min(3, count))

        # Order/sequence anchors
        if re.search(r"\b(before|after|then|next|previous|order|sequence)\b", q):
            anchors = [0, int(num_frames * 0.25), int(num_frames * 0.5), int(num_frames * 0.75), num_frames - 1]
            focus.extend(anchors)

        # Dedup and bounds
        focus = [i for i in sorted(set(focus)) if 0 <= i < num_frames]
        if len(focus) > count:
            keep = self._uniform_frame_indices(len(focus), count)
            focus = [focus[i] for i in keep]
        return focus

    def _uniform_frame_indices_with_focus(
        self,
        num_frames: int,
        count: int,
        query: str,
        q_flags: Dict[str, bool]
    ) -> List[int]:
        """Uniform sampling with temporal-focused bias when needed."""
        base = self._uniform_frame_indices(num_frames, count)
        if not q_flags.get("temporal"):
            return base
        focus = self._temporal_focus_indices(num_frames, query, count)
        if not focus:
            return base
        merged: List[int] = []
        for idx in focus:
            if idx not in merged:
                merged.append(idx)
            if len(merged) >= count:
                return sorted(merged)
        for idx in base:
            if idx not in merged:
                merged.append(idx)
            if len(merged) >= count:
                break
        return sorted(merged)

    def _dump_full_model_output_to_log(self, output: str, tag: str) -> None:
        """
        Dump the FULL model output into log_test_*.txt for manual inspection.
        """
        try:
            self._log(f"  [MODEL_OUTPUT] Dumping full model output below ({tag})")
            print(f"----- {tag}_BEGIN -----")
            print(output or "")
            print(f"----- {tag}_END -----")
        except Exception:
            # Never let logging itself break the pipeline.
            pass

    # ===== V4.3.2: Dynamic option support (A/E/...) =====

    def _opt_sort_key(self, opt_id: str) -> Tuple[int, str]:
        """Sort options in a stable, human-friendly way (A, B, C, ..., Z, then others)."""
        s = (opt_id or "").strip().upper()
        if len(s) == 1 and "A" <= s <= "Z":
            return (0, s)
        # Numeric options (rare)
        try:
            return (1, f"{int(s):06d}")
        except Exception:
            return (2, s)

    def _infer_option_ids_from_query(self, query: str) -> List[str]:
        """
        Best-effort parse option ids from the raw query string.
        Supports common formats:
        - Lines like "A. xxx" / "B) xxx" / "(C) xxx"
        - Inline python-list string like: Options: ['A. xxx', 'B. yyy', ...]
        """
        import re
        import ast

        q = query or ""
        found: List[str] = []

        # 1) Parse python-list style: Options: ['A. ...', 'B. ...']
        try:
            if "Options:" in q and "[" in q and "]" in q:
                start = q.find("[")
                end = q.rfind("]")
                if 0 <= start < end:
                    sub = q[start : end + 1]
                    maybe_list = ast.literal_eval(sub)
                    if isinstance(maybe_list, (list, tuple)):
                        for item in maybe_list:
                            s = str(item)
                            m = re.match(r"^\s*([A-Z])\s*[\.\):]\s*", s)
                            if m:
                                found.append(m.group(1).upper())
        except Exception:
            pass

        # 2) Parse line-prefix style
        for line in q.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^\s*\(?\s*([A-Z])\s*\)?\s*[\.\):]\s+", line)
            if m:
                found.append(m.group(1).upper())

        # Dedupe preserve order
        out: List[str] = []
        seen = set()
        for x in found:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return sorted(out, key=self._opt_sort_key)

    def _infer_option_ids(self, plan, query: str = "") -> List[str]:
        """Infer option ids from plan outputs and/or raw query."""
        keys = set()
        try:
            if getattr(plan, "option_keywords", None):
                keys |= set(getattr(plan, "option_keywords").keys())
        except Exception:
            pass
        try:
            if getattr(plan, "semantic_queries", None):
                keys |= set(getattr(plan, "semantic_queries").keys())
        except Exception:
            pass
        for x in self._infer_option_ids_from_query(query):
            keys.add(x)

        out = [k.strip().upper() for k in keys if isinstance(k, str) and k.strip()]
        out = [k for k in out if len(k) == 1 and "A" <= k <= "Z"]
        if not out:
            out = ["A", "B", "C", "D"]
        return sorted(out, key=self._opt_sort_key)

    def _compute_auto_max_steps(self, option_ids: List[str]) -> int:
        """
        Auto-scale max_steps based on option count.

        Default rule: 4 options -> 10 steps, each extra option adds 1 step.
        Tunable via env:
          - VIDEODETECTIVE_BASE_MAX_STEPS (default: 10)
          - VIDEODETECTIVE_BASE_OPTION_COUNT (default: 4)
          - VIDEODETECTIVE_STEPS_PER_EXTRA_OPTION (default: 1)
        """
        try:
            base_steps = int(os.getenv("VIDEODETECTIVE_BASE_MAX_STEPS", "10"))
        except Exception:
            base_steps = 10
        try:
            base_options = int(os.getenv("VIDEODETECTIVE_BASE_OPTION_COUNT", "4"))
        except Exception:
            base_options = 4
        try:
            steps_per_extra = int(os.getenv("VIDEODETECTIVE_STEPS_PER_EXTRA_OPTION", "1"))
        except Exception:
            steps_per_extra = 1

        opt_count = len(option_ids) if option_ids else base_options
        opt_count = max(1, int(opt_count))
        base_steps = max(1, int(base_steps))
        steps_per_extra = max(0, int(steps_per_extra))

        extra = max(0, opt_count - base_options)
        return max(1, base_steps + extra * steps_per_extra)

    def _merge_with_uniform(
        self,
        selected: List[int],
        num_frames: int,
        total_budget: int,
        coverage_count: int
    ) -> List[int]:
        """Merge selected frames with uniform coverage frames, preserving budget."""
        if total_budget <= 0:
            return []
        merged = []
        seen = set()
        for idx in selected:
            if idx not in seen:
                merged.append(idx)
                seen.add(idx)
        for idx in self._uniform_frame_indices(num_frames, coverage_count):
            if len(merged) >= total_budget:
                break
            if idx not in seen:
                merged.append(idx)
                seen.add(idx)
        merged = merged[:total_budget]
        return sorted(merged)

    def _load_asr_data(self, video_path: str, fps: float) -> Dict[int, str]:
        """
        Extract ASR (Automatic Speech Recognition) data from video using Whisper.
        
        Runs Whisper in real-time to transcribe the video audio.
        
        Args:
            video_path: Path to video file.
            fps: Frame rate for mapping ASR to frame indices.
        
        Returns:
            Dict mapping frame indices to ASR text for that frame's time range.
        """
        # Check if ASR is enabled
        enable_asr = os.getenv("VIDEODETECTIVE_ENABLE_ASR", "true").lower() in ("true", "1", "yes", "on")
        if not enable_asr:
            self._log("  ASR disabled via VIDEODETECTIVE_ENABLE_ASR=false")
            return {}
        
        # Get ASR extractor (uses Whisper)
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        whisper_model_dir = str(project_root / "models" / "whisper")
        whisper_model = os.getenv("VIDEODETECTIVE_WHISPER_MODEL", "base")
        
        self._log(f"  Initializing Whisper ASR (model={whisper_model})...")
        asr_extractor = get_asr_extractor(
            model_name=whisper_model,
            model_dir=whisper_model_dir
        )
        
        if asr_extractor is None:
            self._log("  ASR extractor not available (Whisper not installed or failed to load)")
            return {}
        
        # Extract ASR from video
        self._log(f"  Extracting ASR from video...")
        asr_by_frame = asr_extractor.get_asr_by_frame(video_path, fps=fps)
        
        if asr_by_frame:
            self._log(f"  ASR extracted: {len(asr_by_frame)} frames with text")
        else:
            self._log(f"  No ASR text extracted (video may have no audio or speech)")
        
        return asr_by_frame

    def _get_asr_for_chunk(self, chunk: Tuple[int, int], asr_by_frame: Dict[int, str]) -> str:
        """Get ASR text for a specific chunk (frame range)."""
        start_frame, end_frame = chunk
        texts = []
        seen = set()
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx in asr_by_frame:
                text = asr_by_frame[frame_idx]
                if text not in seen:
                    texts.append(text)
                    seen.add(text)
        return " ".join(texts)

    def _dedup_keywords(self, keywords: List[str]) -> List[str]:
        """Deduplicate keywords while preserving order."""
        seen = set()
        out = []
        for kw in keywords:
            kw = (kw or "").strip()
            if not kw or kw in seen:
                continue
            seen.add(kw)
            out.append(kw)
        return out

    def _filter_visual_keywords(self, keywords: List[str]) -> List[str]:
        """Filter out text-only keywords for visual retrieval."""
        if not keywords:
            return []
        filtered = []
        for kw in keywords:
            s = (kw or "").strip()
            if not s:
                continue
            if s.lower().startswith("text:"):
                continue
            filtered.append(s)
        return self._dedup_keywords(filtered)

    def _filter_visual_option_keywords(
        self, option_keywords: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Filter option keyword map for visual retrieval."""
        out: Dict[str, List[str]] = {}
        for opt_id, kws in (option_keywords or {}).items():
            out[opt_id] = self._filter_visual_keywords(kws or [])
        return out

    def _build_frame_scores(
        self,
        feats: np.ndarray,
        plan,
        asr_by_frame: Optional[Dict[int, str]] = None
    ) -> Optional[np.ndarray]:
        """Compute per-frame relevance scores using keyword-visual similarity."""
        try:
            keywords = []
            if getattr(plan, "query_keywords", None):
                keywords.extend(plan.query_keywords)
            if getattr(plan, "option_keywords", None):
                for kws in plan.option_keywords.values():
                    keywords.extend(kws)
            keywords = self._filter_visual_keywords(keywords)
            if not keywords:
                return None
            text_feats = self.encoder.encode_text(keywords)
            sims = text_feats @ feats.T  # [K, T]
            scores = np.max(sims, axis=0)

            # Normalize to [0, 1]
            scores = np.clip(scores, 0.0, 1.0)

            return scores
        except Exception:
            return None

    def _should_fallback_to_uniform(self, observations: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Decide whether to fall back to uniform sampling."""
        if not observations:
            return True, "no_observations"
        rels = [o.get("relevance", 0.0) for o in observations if isinstance(o, dict)]
        if not rels:
            return True, "no_relevance_scores"
        max_rel = max(rels)
        mean_rel = sum(rels) / len(rels)
        gap = max_rel - mean_rel
        low_ratio = sum(1 for r in rels if r < 0.2) / len(rels)
        if max_rel < 0.4:
            return True, f"low_max_rel={max_rel:.2f}"
        if mean_rel < 0.2:
            return True, f"low_mean_rel={mean_rel:.2f}"
        # Flatness fallback: scores are too uniform to be reliable
        # Tunable threshold:
        # - Lower => fewer fallbacks (more trust in retrieval)
        # - Higher => more fallbacks (more coverage-first)
        #
        # Tunable threshold via env var. Default = 0.15 (original behavior).
        try:
            import os as _os
            flat_gap_th = float(_os.getenv("VIDEODETECTIVE_FLAT_GAP_THRESHOLD", "0.15"))
        except Exception:
            flat_gap_th = 0.15
        if gap < flat_gap_th:
            return True, f"flat_gap={gap:.2f}"
        if low_ratio > 0.7:
            return True, f"low_ratio={low_ratio:.2f}"
        return False, ""

    def _build_option_pack_frames(
        self,
        graph: GraphBeliefEngine,
        frames: List[Image.Image],
        chunks: List[Tuple[int, int]],
        frame_scores: Optional[np.ndarray],
        option_anchors: Dict[str, int],
        best_option_nodes: Dict[str, Tuple[int, float]],
        total_budget: int = 32
    ) -> Dict[str, Any]:
        """Build per-option frame packs for NOT-correct questions."""
        # V4.3.2: Support dynamic options (A/E/...) for option-pack mode.
        option_order = sorted(option_anchors.keys(), key=self._opt_sort_key)
        if not option_order:
            option_order = sorted(best_option_nodes.keys(), key=self._opt_sort_key)
        frames_per_chunk = 4
        per_option_budget = max(1, total_budget // len(option_order))
        chunks_per_option = max(1, per_option_budget // frames_per_chunk)

        option_ranges: Dict[str, Tuple[int, int]] = {}
        frame_indices: List[int] = []

        for opt_id in option_order:
            chunk_list: List[int] = []
            best_node = best_option_nodes.get(opt_id, (None, 0.0))[0]
            if best_node is not None:
                chunk_list.append(best_node)
            anchor = option_anchors.get(opt_id)
            if anchor is not None and anchor not in chunk_list:
                chunk_list.append(anchor)
            candidates = graph.get_candidates_for_option(opt_id, exclude_visited=False)
            for cand in candidates:
                if cand not in chunk_list:
                    chunk_list.append(cand)
                if len(chunk_list) >= chunks_per_option:
                    break

            opt_frames: List[int] = []
            seen = set()
            for chunk_idx in chunk_list[:chunks_per_option]:
                chunk_frames = graph.get_frames_from_chunks(
                    [chunk_idx],
                    frames_per_chunk=frames_per_chunk,
                    frame_scores=frame_scores
                )
                for fi in chunk_frames:
                    if fi not in seen:
                        opt_frames.append(fi)
                        seen.add(fi)
                if len(opt_frames) >= per_option_budget:
                    break
            opt_frames = opt_frames[:per_option_budget]

            if opt_frames:
                start_idx = len(frame_indices) + 1
                frame_indices.extend(opt_frames)
                end_idx = len(frame_indices)
                option_ranges[opt_id] = (start_idx, end_idx)

        # Fill remaining budget with uniform frames (extra context)
        extra_range = None
        if len(frame_indices) < total_budget and len(frames) > 0:
            needed = total_budget - len(frame_indices)
            uniform = self._uniform_frame_indices(len(frames), needed)
            start_idx = len(frame_indices) + 1
            added = 0
            seen = set(frame_indices)
            for idx in uniform:
                if idx in seen:
                    continue
                frame_indices.append(idx)
                seen.add(idx)
                added += 1
                if len(frame_indices) >= total_budget:
                    break
            if added > 0:
                extra_range = (start_idx, start_idx + added - 1)

        # Trim to budget if needed
        frame_indices = frame_indices[:total_budget]

        return {
            "frame_indices": frame_indices,
            "option_ranges": option_ranges,
            "extra_range": extra_range,
        }
    
    def _build_caption_map(
        self,
        observations: List[Dict],
        chunks: List[Tuple[int, int]],
        frame_indices: List[int]
    ) -> Dict[int, List[str]]:
        """Build caption map for final frames based on observations."""
        caption_map: Dict[int, List[str]] = {}
        fps = self.settings.default_fps
        
        for obs in observations:
            if not obs.get("caption"):
                continue
            
            node_idx = obs["node"]
            if node_idx < 0 or node_idx >= len(chunks):
                continue
            
            start, end = chunks[node_idx]
            caption = obs["caption"]
            
            # Find which final frames fall in this chunk
            for i, frame_idx in enumerate(frame_indices):
                if start <= frame_idx <= end:
                    time_range = f"[{start/fps:.1f}s-{end/fps:.1f}s]"
                    caption_str = f"{time_range}: {caption}"
                    if i not in caption_map:
                        caption_map[i] = []
                    if caption_str not in caption_map[i]:
                        caption_map[i].append(caption_str)
        
        return caption_map

    def _build_evidence_map(
        self,
        observations: List[Dict[str, Any]],
        chunks: List[Tuple[int, int]],
        frame_indices: List[int]
    ) -> Dict[int, List[str]]:
        """
        Build evidence text map for inspected chunks.

        Only frames whose chunks were inspected get evidence text.
        """
        if not observations or not frame_indices:
            return {}

        fps = self.settings.default_fps
        evidence_map: Dict[int, List[str]] = {}

        # Keep best observation per node (highest final_relevance)
        obs_by_node: Dict[int, Dict[str, Any]] = {}
        for obs in observations:
            node_idx = obs.get("node")
            if node_idx is None:
                continue
            prev = obs_by_node.get(node_idx)
            if prev is None or float(obs.get("final_relevance", 0.0)) > float(prev.get("final_relevance", 0.0)):
                obs_by_node[node_idx] = obs

        for node_idx, obs in obs_by_node.items():
            if node_idx < 0 or node_idx >= len(chunks):
                continue
            start, end = chunks[node_idx]

            sources = [
                ("ASR", float(obs.get("asr_score", 0.0)), obs.get("asr_text", "")),
                ("OCR", float(obs.get("ocr_score", 0.0)), obs.get("ocr_text", "")),
                ("CAPTION", float(obs.get("caption_score", 0.0)), obs.get("caption", "")),
            ]
            best_src, best_score, best_text = max(sources, key=lambda x: x[1])
            best_text = (best_text or "").strip()
            if best_score <= 0.0 or not best_text:
                continue

            best_text = re.sub(r"\s+", " ", best_text).strip()
            if len(best_text) > 180:
                best_text = best_text[:180] + "..."

            time_range = f"{start/fps:.1f}-{end/fps:.1f}s"
            # Include the TF-IDF-derived score for traceability.
            evidence_str = f"[{time_range}] {best_src}({best_score:.2f}): {best_text}"

            # Attach evidence only to the first frame from this chunk
            for i, frame_idx in enumerate(frame_indices):
                if start <= frame_idx <= end:
                    evidence_map.setdefault(i, [])
                    if evidence_str not in evidence_map[i]:
                        evidence_map[i].append(evidence_str)
                    break

        return evidence_map

    def _fill_fallback_evidence_map(
        self,
        evidence_map: Dict[int, List[str]],
        observations: List[Dict[str, Any]],
        chunks: List[Tuple[int, int]],
        frame_indices: List[int],
        plan: QueryResultV5,
        asr_by_frame: Dict[int, str],
        graph: GraphBeliefEngine,
        global_word_freq: Dict[str, int]
    ) -> Dict[int, List[str]]:
        """Fill missing evidence entries for fallback frames using ASR or event captions."""
        if not frame_indices:
            return evidence_map or {}
        evidence_map = evidence_map or {}

        obs_by_node: Dict[int, Dict[str, Any]] = {}
        for obs in observations or []:
            node_idx = obs.get("node")
            if node_idx is None:
                continue
            prev = obs_by_node.get(node_idx)
            if prev is None or float(obs.get("final_relevance", 0.0)) > float(prev.get("final_relevance", 0.0)):
                obs_by_node[node_idx] = obs

        opt_keywords = getattr(plan, "option_keywords", {}) or {}
        query_keywords = getattr(plan, "query_keywords", []) or []
        semantic_queries = getattr(plan, "semantic_queries", None)

        for i, frame_idx in enumerate(frame_indices):
            if i in evidence_map:
                continue
            node_idx = self._find_chunk_index(frame_idx, chunks)
            if node_idx is None:
                continue
            if node_idx < 0 or node_idx >= len(chunks):
                continue
            start, end = chunks[node_idx]

            best_src = None
            best_score = 0.0
            best_text = ""

            # Prefer inspected evidence if available
            obs = obs_by_node.get(node_idx)
            if obs:
                sources = [
                    ("ASR", float(obs.get("asr_score", 0.0)), obs.get("asr_text", "")),
                    ("OCR", float(obs.get("ocr_score", 0.0)), obs.get("ocr_text", "")),
                    ("CAPTION", float(obs.get("caption_score", 0.0)), obs.get("caption", "")),
                ]
                best_src, best_score, best_text = max(sources, key=lambda x: x[1])

            # If no inspected evidence, fall back to ASR or event skeleton captions
            if not best_text:
                if asr_by_frame:
                    asr_text = self._get_asr_for_chunk((start, end), asr_by_frame)
                    if asr_text:
                        score = self._calculate_strong_evidence_score(
                            asr_text,
                            opt_keywords,
                            query_keywords,
                            semantic_queries,
                            global_word_freq,
                            source="asr"
                        )
                        if score > best_score:
                            best_src, best_score, best_text = "ASR", score, asr_text
                if getattr(graph, "semantic_captions", None):
                    sem_caption = graph.semantic_captions.get(node_idx, "")
                    if sem_caption:
                        score = self._calculate_strong_evidence_score(
                            sem_caption,
                            opt_keywords,
                            query_keywords,
                            semantic_queries,
                            global_word_freq,
                            source="caption"
                        )
                        if score > best_score:
                            best_src, best_score, best_text = "CAPTION", score, sem_caption

            best_text = (best_text or "").strip()
            if not best_text:
                continue
            if best_score <= 0.0:
                best_score = 0.01

            best_text = re.sub(r"\s+", " ", best_text).strip()
            if len(best_text) > 180:
                best_text = best_text[:180] + "..."

            time_range = f"{start/self.settings.default_fps:.1f}-{end/self.settings.default_fps:.1f}s"
            evidence_str = f"[{time_range}] {best_src}({best_score:.2f}): {best_text}"
            evidence_map.setdefault(i, [])
            if evidence_str not in evidence_map[i]:
                evidence_map[i].append(evidence_str)

        return evidence_map

    def _find_chunk_index(
        self,
        frame_idx: int,
        chunks: List[Tuple[int, int]]
    ) -> Optional[int]:
        """Locate the chunk index containing a specific frame index."""
        for idx, (start, end) in enumerate(chunks):
            if start <= frame_idx <= end:
                return idx
        return None

    def _log_evidence_map(
        self,
        evidence_map: Dict[int, List[str]],
        timestamps: List[str],
        tag: str = "EVIDENCE_MAP"
    ) -> None:
        """Log evidence_map in a compact but explicit way for debugging."""
        try:
            self._log(f"  {tag}: {len(evidence_map)} images have attached evidence")
            print(f"----- {tag}_BEGIN -----")
            for img_idx in sorted(evidence_map.keys()):
                ts = ""
                try:
                    if 0 <= int(img_idx) < len(timestamps):
                        ts = timestamps[int(img_idx)]
                except Exception:
                    ts = ""
                lines = evidence_map.get(img_idx) or []
                if ts:
                    print(f"Image {int(img_idx)+1} @ {ts}:")
                else:
                    print(f"Image {int(img_idx)+1}:")
                for l in lines:
                    print(f"  - {l}")
            print(f"----- {tag}_END -----")
        except Exception:
            pass

    def _parse_answer_from_response(
        self,
        response: str,
        opt_ids: Optional[List[str]] = None
    ) -> str:
        """Parse option letter from LLM response."""
        if not response:
            return "UNPARSEABLE"
        match = re.search(r"Final Answer:\s*([A-Z])", response, re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
        else:
            letters = re.findall(r"\b([A-Z])\b", response)
            answer = letters[-1].upper() if letters else "UNPARSEABLE"
        if opt_ids and answer not in opt_ids:
            return "UNPARSEABLE"
        return answer

    def _retry_answer_with_options(
        self,
        frames: List[Image.Image],
        prompt: str,
        opt_ids: List[str]
    ) -> Tuple[str, str]:
        """Second-pass call to force a valid option letter output."""
        if not opt_ids:
            return "UNPARSEABLE", ""
        system_prompt = (
            "You must output ONLY one option letter from the allowed list. "
            f"Allowed options: {', '.join(opt_ids)}. "
            "Output a single letter only."
        )
        try:
            response = self.llm_client.chat_with_images(
                prompt=prompt,
                images=frames,
                system_prompt=system_prompt
            )
            answer = self._parse_answer_from_response(response, opt_ids)
            return answer, response
        except Exception as e:
            return "UNPARSEABLE", f"VLM error: {str(e)}"

    def _generate_answer_option_pack(
        self,
        frames: List[Image.Image],
        query: str,
        timestamps: List[str],
        option_ranges: Dict[str, Tuple[int, int]],
        extra_range: Optional[Tuple[int, int]] = None,
        captions: Dict[int, List[str]] = None
    ) -> Tuple[str, str]:
        """Generate final answer using option-packed frames for NOT-correct questions."""
        system_prompt = """You are a video analysis assistant.
Your task is to identify the ONE INCORRECT statement among the options.

The images are grouped by option. Use ONLY the images in each option's group to judge that option.

DECISION STEPS (follow these in order):
1) Read the question and all options.
2) For each option, inspect ONLY its image group and the attached evidence lines.
3) Mark each option as supported, contradicted, or unclear.
4) Choose the single INCORRECT option based on the strongest contradiction or weakest support.
5) If evidence is weak, choose the best guess and state low confidence.

CRITICAL RULES:
- You MUST output an option LETTER as the final answer (e.g., A/B/C/D).
- Even if evidence is weak, you must still choose the best answer letter and state low confidence in the reason.
- DO NOT output placeholders like "NO EVIDENCE" as the final answer.
- You MUST think step-by-step and consider each option internally before deciding.
- Write the analysis and reason in English.

RESPONSE FORMAT:
Analysis: <your reasoning>
Final Answer: <ONE LETTER>
Reason: <one short sentence>"""

        group_lines = []
        opt_ids = sorted(option_ranges.keys(), key=self._opt_sort_key)
        if not opt_ids:
            opt_ids = self._infer_option_ids_from_query(query)
        for opt_id in opt_ids:
            if opt_id in option_ranges:
                s, e = option_ranges[opt_id]
                group_lines.append(f"Option {opt_id}: Images {s}-{e}")
            else:
                group_lines.append(f"Option {opt_id}: NO IMAGES")
        if extra_range:
            group_lines.append(f"Extra context: Images {extra_range[0]}-{extra_range[1]}")

        frame_info_lines = []
        for i, ts in enumerate(timestamps):
            line = f"Image {i+1}: {ts}"
            if captions and i in captions:
                for cap in captions[i]:
                    line += f"\n    (During {cap})"
            frame_info_lines.append(line)
        frame_info_str = "\n".join(frame_info_lines)

        prompt = f"""Based on these video frames, answer the following question:

Frame Groups:
{chr(10).join(group_lines)}

Frame Information:
{frame_info_str}

Question: {query}

Provide your analysis and final answer:"""

        try:
            response = self.llm_client.chat_with_images(
                prompt=prompt,
                images=frames,
                system_prompt=system_prompt
            )
            answer = self._parse_answer_from_response(response, opt_ids)
            if answer == "UNPARSEABLE" and opt_ids:
                retry_prompt = (
                    "Answer with ONLY one option letter.\n"
                    f"Allowed options: {', '.join(opt_ids)}\n\n"
                    f"Frame Groups:\n{chr(10).join(group_lines)}\n\n"
                    f"Frame Information:\n{frame_info_str}\n\n"
                    f"Question: {query}\n"
                )
                retry_answer, retry_response = self._retry_answer_with_options(
                    frames,
                    retry_prompt,
                    opt_ids
                )
                if retry_answer != "UNPARSEABLE":
                    response = response + "\n\n[RETRY]\n" + retry_response
                    answer = retry_answer
            return answer, response
        except Exception as e:
            return "Error", f"VLM error: {str(e)}"
    
    def _generate_answer(
        self,
        frames: List[Image.Image],
        query: str,
        timestamps: List[str] = None,
        captions: Dict[int, List[str]] = None
    ) -> Tuple[str, str]:
        """Generate final answer using VLM."""
        # Detect "NOT correct" question type
        is_not_correct = any(kw in query.lower() for kw in ["not correct", "incorrect", "not true", "false", "is not"])
        
        target_verdict = "FALSE" if is_not_correct else "TRUE"
        criteria = "INCORRECT" if is_not_correct else "CORRECT"
        
        # V4.4: Minimal, parse-friendly output (avoid UNPARSEABLE like "NO EVIDENCE")
        system_prompt = f"""You are a video analysis assistant.
Your task is to identify the ONE {criteria} statement among the options.

DECISION STEPS (follow these in order):
1) Read the question and all options.
2) For each option, check the frames and the attached evidence lines (the "During ..." lines).
3) Prefer explicit evidence over vague impressions.
4) If the question is about order/time, compare early vs late frames; if about text, use OCR evidence.
5) If evidence is weak, choose the most plausible option and say low confidence.

CRITICAL RULES:
- You MUST output an option LETTER as the final answer (e.g., A/B/C/D).
- Even if evidence is weak or unclear, you must still choose the best answer letter and state low confidence in the reason.
- DO NOT output placeholders like "NO EVIDENCE" as the final answer.
- You MUST think step-by-step before deciding.
- Write the analysis and reason in English.

RESPONSE FORMAT:
Analysis: <your reasoning>
Final Answer: <ONE LETTER>
Reason: <one short sentence>"""
        
        # Build frame info
        frame_info_lines = []
        if timestamps:
            for i, ts in enumerate(timestamps):
                line = f"Image {i+1}: {ts}"
                if captions and i in captions:
                    for cap in captions[i]:
                        line += f"\n    (During {cap})"
                frame_info_lines.append(line)
        
        frame_info_str = "\n".join(frame_info_lines) if frame_info_lines else ""
        
        prompt = f"""Based on these video frames, answer the following question:

Frame Information:
{frame_info_str}

Question: {query}

Remember: include a clear 'Final Answer: <LETTER>' line so it can be parsed."""
        
        try:
            response = self.llm_client.chat_with_images(
                prompt=prompt,
                images=frames,
                system_prompt=system_prompt
            )
            
            opt_ids = self._infer_option_ids_from_query(query)
            answer = self._parse_answer_from_response(response, opt_ids)
            if answer == "UNPARSEABLE" and opt_ids:
                retry_prompt = (
                    "Answer with ONLY one option letter.\n"
                    f"Allowed options: {', '.join(opt_ids)}\n\n"
                    f"Frame Information:\n{frame_info_str}\n\n"
                    f"Question: {query}\n"
                )
                retry_answer, retry_response = self._retry_answer_with_options(
                    frames,
                    retry_prompt,
                    opt_ids
                )
                if retry_answer != "UNPARSEABLE":
                    response = response + "\n\n[RETRY]\n" + retry_response
                    answer = retry_answer
            return answer, response
        
        except Exception as e:
            return "Error", f"VLM error: {str(e)}"

    def _calculate_strong_evidence_score(
        self,
        text: str,
        option_keywords: Dict[str, List[str]],
        query_keywords: List[str],
        semantic_queries: Dict[str, str] = None,
        global_word_freq: Dict[str, int] = None,
        source: str = None  # V4.7: "ocr", "asr", "caption" for weighted fusion
    ) -> float:
        """
        Calculate semantic match score between text and multi-route queries.
        
        V4.8: Hybrid retrieval combining:
        - Sparse route (TF-IDF) ONLY for keyword/entity matching
        - Dense route (text embedding) ONLY for semantic event queries
        
        Final score = weighted blend of (tfidf_normalized, dense_semantic_score)
        """
        if not text or len(text.strip()) < 5:
            return 0.0

        # Collect all candidates
        candidates = []
        candidates.extend(query_keywords or [])
        for kws in option_keywords.values():
            candidates.extend(kws or [])
        
        # Collect semantic queries
        sem_queries = []
        if semantic_queries:
            sem_queries = [q for q in semantic_queries.values() if q]

        # --- Route 1: TF-IDF Sparse Matching ---
        # TF-IDF scores range ~0.0 to ~0.5 typically, need normalization
        # Keywords/entities only (caption/OCR/ASR are matched sparsely to avoid semantic drift)
        tfidf_score = self._tfidf_similarity(text, candidates, global_word_freq) if candidates else 0.0
        # Normalize TF-IDF: mild scaling to align with embedding range
        tfidf_normalized = min(1.0, tfidf_score * 1.3)

        # --- Route 2: Embedding Dense Matching ---
        # Semantic queries only (dense)
        # CLIP/SigLIP has 77 token limit, so chunk long text
        embedding_score = 0.0
        try:
            if sem_queries and hasattr(self, "encoder"):
                # Split long text into chunks (~60 words each to stay under 77 tokens)
                text_chunks = self._split_text_for_embedding(text, max_words=50)
                if text_chunks:
                    # Encode all text chunks
                    text_feats = self.encoder.encode_text(text_chunks)  # [C, D]
                    query_feats = self.encoder.encode_text(sem_queries)  # [N, D]
                    # Compute all pairwise similarities
                    sims = np.dot(text_feats, query_feats.T)  # [C, N]
                    # Take max across all chunks and queries
                    embedding_score = float(np.max(sims)) if sims.size > 0 else 0.0
                    embedding_score = float(np.clip(embedding_score, 0.0, 1.0))
        except Exception:
            embedding_score = 0.0

        # --- Fusion: weighted blend based on source type ---
        # OCR: tfidf is more reliable (0.7/0.3)
        # ASR: balanced (0.5/0.5)
        # Caption: embedding is more reliable (0.3/0.7)
        source = (source or "").lower()
        if source == "ocr":
            tfidf_weight, embed_weight = 0.7, 0.3
        elif source == "asr":
            tfidf_weight, embed_weight = 0.5, 0.5
        elif source == "caption":
            tfidf_weight, embed_weight = 0.3, 0.7
        else:
            # Default: balanced
            tfidf_weight, embed_weight = 0.5, 0.5
        
        final_score = tfidf_weight * tfidf_normalized + embed_weight * embedding_score
        
        return float(np.clip(final_score, 0.0, 1.0))

    def _tokenize_for_tfidf(self, text: str) -> List[str]:
        """Tokenize text for TF-IDF scoring (ASCII-safe)."""
        if not text:
            return []
        text = text.lower().replace("text:", " ")
        text = re.sub(r"[^a-z0-9]+", " ", text)
        tokens = []
        for tok in text.split():
            tok = tok.strip()
            if not tok:
                continue
            if len(tok) > 1 or tok.isdigit():
                tokens.append(tok)
        return tokens

    def _split_text_for_embedding(self, text: str, max_words: int = 50) -> List[str]:
        """
        Split long text into chunks for CLIP/SigLIP embedding (77 token limit).
        
        Uses ~50 words per chunk to stay safely under the token limit.
        Returns at least one chunk even if text is short.
        """
        if not text or not text.strip():
            return []
        
        words = text.split()
        if len(words) <= max_words:
            return [text.strip()]
        
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks if chunks else [text.strip()]

    def _tfidf_similarity(
        self,
        text: str,
        candidates: List[str],
        global_word_freq: Dict[str, int] = None
    ) -> float:
        """Compute max TF-IDF cosine similarity between text and candidate phrases."""
        if not text or not candidates:
            return 0.0

        text_tokens = self._tokenize_for_tfidf(text)
        if not text_tokens:
            return 0.0

        total_words = sum(global_word_freq.values()) if global_word_freq else 0
        if total_words <= 0:
            total_words = max(1, len(text_tokens))

        def idf(tok: str) -> float:
            if not global_word_freq:
                return 1.0
            freq = global_word_freq.get(tok, 0)
            return float(np.log((1.0 + total_words) / (1.0 + freq)) + 1.0)

        def tfidf_vec(tokens: List[str]) -> Dict[str, float]:
            if not tokens:
                return {}
            counts = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            denom = float(len(tokens))
            vec = {}
            for t, c in counts.items():
                tf = c / denom
                vec[t] = tf * idf(t)
            return vec

        text_vec = tfidf_vec(text_tokens)
        if not text_vec:
            return 0.0
        text_norm = float(np.sqrt(sum(v * v for v in text_vec.values())))
        if text_norm <= 0:
            return 0.0

        best = 0.0
        for cand in candidates:
            cand_tokens = self._tokenize_for_tfidf(cand or "")
            if not cand_tokens:
                continue
            cand_vec = tfidf_vec(cand_tokens)
            if not cand_vec:
                continue
            cand_norm = float(np.sqrt(sum(v * v for v in cand_vec.values())))
            if cand_norm <= 0:
                continue
            dot = 0.0
            # Compute dot product on overlap
            for tok, v in cand_vec.items():
                dot += v * text_vec.get(tok, 0.0)
            sim = dot / (text_norm * cand_norm) if dot > 0 else 0.0
            if sim > best:
                best = sim

        return float(best)
    
    # ===== V4.3: Event Skeleton Building =====
    
    def _build_event_skeleton(
        self,
        frames: List[Image.Image],
        chunks: List[Tuple[int, int]],
        graph: GraphBeliefEngine
    ) -> Optional[EventSkeleton]:
        """
        Build event skeleton from video frames using VideoSummarizer.
        
        Args:
            frames: All video frames.
            chunks: Graph node ranges.
            graph: GraphBeliefEngine instance.
        
        Returns:
            EventSkeleton or None if generation fails.
        """
        try:
            T = len(frames)
            if T == 0:
                return None
            
            # Sample skeleton frames (uniformly)
            num_skeleton_frames = min(self.settings.skeleton_frames, T)
            if num_skeleton_frames <= 0:
                return None
            
            # Uniform sampling
            step = T / num_skeleton_frames
            sample_indices = [int(i * step) for i in range(num_skeleton_frames)]
            sample_indices = [min(idx, T - 1) for idx in sample_indices]
            
            sample_frames = [frames[i] for i in sample_indices]
            timestamps = [i / self.settings.default_fps for i in sample_indices]
            
            self._log(f"    Sampling {len(sample_frames)} frames for skeleton")
            
            # Generate skeleton
            skeleton = self.video_summarizer.summarize(
                sample_frames,
                timestamps,
                chunks,
                fps=self.settings.default_fps
            )
            
            # Build caption map for graph
            if skeleton and skeleton.events:
                caption_map = self.video_summarizer.build_chunk_caption_map(skeleton)
                graph.set_semantic_captions(caption_map)
                self._log(f"    Mapped captions to {len(caption_map)} chunks")
            
            return skeleton

        except Exception as e:
            self._log(f"    Event skeleton generation failed: {e}")
            return None
