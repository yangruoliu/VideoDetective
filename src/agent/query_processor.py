"""
Query Processor for VideoDetective v4.0.

Extracts visual keywords with multi-channel decomposition:
- Temporal plan: start/action/result keywords
- Option keywords: unique entities per option (supports A-Z)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .llm_client import LLMClient


@dataclass
class QueryResultV4:
    """Result of v4.0 query processing with multi-channel decomposition."""
    temporal_plan: Dict[str, List[str]]  # {"start": [], "action": [], "result": []}
    option_keywords: Dict[str, List[str]]  # {"A": [], "B": [], "C": [], "D": []}
    query_keywords: List[str]  # Keywords specific to the question itself
    vlm_query: str
    original_query: str
    
    # Backward compatibility
    @property
    def visual_keywords(self) -> List[str]:
        """Get all keywords combined for backward compatibility."""
        all_kw = []
        for kws in self.temporal_plan.values():
            all_kw.extend(kws)
        all_kw.extend(self.query_keywords)
        return all_kw


@dataclass
class QueryResultV5(QueryResultV4):
    """
    Result of v4.3 query processing with multi-route recall support.
    
    Extends V4 with semantic queries for text-based retrieval.
    """
    # V4.3: Semantic queries for matching against event descriptions
    semantic_queries: Dict[str, str] = None  # {"A": "...", "B": "...", ...}
    general_semantic_query: str = ""  # General query for overall video understanding
    
    def __post_init__(self):
        if self.semantic_queries is None:
            self.semantic_queries = {}


# Alias for backward compatibility
QueryResult = QueryResultV4


class QueryProcessor:
    """
    Query processor with multi-channel decomposition for v4.0/v4.3.
    
    Extracts:
    - Temporal keywords for start/action/result stages
    - Option-specific keywords for each choice (supports A-Z)
    - Query-specific keywords (from the question text)
    - De-abstracted visual keywords (abstract -> concrete)
    - V4.3: Semantic queries for text-based retrieval
    """
    
    SYSTEM_PROMPT = """Role: Search Keyword Extractor for Video Understanding.
Task: Translate User Query and Options into concrete, SEARCHABLE VISUAL KEYWORDS.

## CRITICAL: Abstract-to-Concrete Translation
If a concept is HIGH-LEVEL or ABSTRACT (cannot be directly seen in video), you MUST translate it to CONCRETE VISUAL objects/actions.

### Example 1: Social/Historical Concept
Query: "How does the video illustrate Civil Disobedience?"
Options: "A. Violent protests. B. Peaceful sit-ins."
Translation:
- "Civil Disobedience" (abstract) -> ["Protest signs", "Police blocking road", "People sitting on ground", "Handcuffs"]
- "Peaceful sit-ins" -> ["Cross-legged sitting", "Holding hands", "Singing", "No weapons visible"]

### Example 2: Scientific/Technical Process
Query: "What method was used to trace the origin of the methane?"
Options: "A. Isotope analysis. B. Satellite imaging."
Translation:
- "Trace origin" (abstract) -> ["Scientist holding sample tube", "Drilling ice core", "Collecting gas bubbles"]
- "Isotope analysis" -> ["Lab equipment", "Mass spectrometer", "Computer screen with graphs", "Blue flame test"]

### Example 3: Emotional/Thematic Concept
Query: "What is the terrifying reality shown in the video?"
Options: "A. War casualties. B. Environmental destruction."
Translation:
- "Terrifying reality" (abstract) -> ["Crying people", "Destroyed buildings", "Smoke", "Hospital beds"]
- "War casualties" -> ["Bodies on ground", "Injured soldiers", "text: death toll"]

## Rules:
1. **De-Abstraction**: ALWAYS translate abstract concepts into concrete visual objects.
2. **Question Analysis**: Extract `query_keywords` from the QUESTION itself (ignore options).
3. **Temporal Decomposition**:
   - Start_Keywords: Visuals expected at intro.
   - Action_Keywords: Visuals of main event/conflict.
   - Result_Keywords: Visuals of outcome.
4. **Option Entity Extraction (Crucial)**:
   - For EACH option, extract unique Nouns/Verbs.
   - If an option mentions a specific Name, Number, or Year, add "text:" prefix (e.g., "text: 1998").
   - Focus on what makes each option VISUALLY DISTINCT.
5. **Text Keyword Hygiene**:
   - "text:" is ONLY for specific on-screen text likely to appear verbatim (names, titles, numbers).
   - DO NOT output option letters (A/B/C/D) or generic words like "help", "each other", "correct".
   - Avoid pronouns/relations; prefer concrete entities (e.g., "orange", "bird", "John Smith", "text:2020").

## In-Context Examples (do the same):

Example A:
Question: "What does Jon Snow use to fight with Ramsay Bolton?"
Options: "A. A shield. B. A sword. C. An Axe. D. A spear."
Good keywords:
- query_keywords: ["Jon Snow", "Ramsay Bolton"]
- option_keywords: {"A": ["shield"], "B": ["sword"], "C": ["axe"], "D": ["spear"]}

Example B:
Question: "What is the relationship between the black stickman and the red stickman?"
Options: "A. Friends. B. Initially friends, but then become enemies. C. Enemies. D. They don't know each other."
Good keywords:
- query_keywords: ["black stickman", "red stickman"]
- option_keywords:
  A: ["friendly gesture", "walking together", "smiling"]
  B: ["team up", "then fight", "betrayal", "attack"]
  C: ["fight", "attack", "weapon clash"]
  D: ["ignore", "pass by", "no interaction"]

Output ONLY valid JSON in English, no other text."""

    # V4.3: Extended system prompt for multi-route recall
    SYSTEM_PROMPT_V5 = """Role: Entity & Event Extractor for Video Understanding.
Task: Extract ENTITIES (for keyword matching) and EVENTS (for semantic matching) from user query.

## Output Requirements

### 1. Query Keywords (ENTITIES from question)
- Extract SPECIFIC ENTITIES: person names, place names, object names, numbers, years
- Examples: "John Brown", "Smithsonian Museum", "1901", "biplane", "red car"
- MUST be empty [] if the question has no specific entities.
  - Example (no entities): "Based on the content of the video, whose viewpoint is relatively the most radical?"
    -> query_keywords MUST be []
- **FORBIDDEN WORDS** (never output these):
  - Generic question words: "based", "content", "whose", "viewpoint", "relatively", "according", "following", "statement", "correct", "video", "mainly", "about", "which", "what", "how", "who", "where", "when", "why"
  - Abstract concepts: "person", "thing", "object", "item", "something", "someone"

### 2. General Semantic Query (EVENT description)
- Describe WHAT EVENT must happen in the video to answer the question
- Focus on ACTIONS and PROCESSES, not objects
- Example: "A person explains the history of early aviation"
- CAN be empty if focus is entirely on options

### 3. Option Keywords (ENTITIES per option) - REQUIRED
- Extract 2-5 SPECIFIC ENTITIES per option
- Types: person names, object names, visible text, numbers
- Use "text:" prefix ONLY for meaningful on-screen text (e.g., "text:1901", "text:John Brown")
- NEVER output option letters (A/B/C/D) or generic words like "person", "thing", "object", "help", "each other"

### 4. Option Semantic Queries (EVENT per option)
- Describe the SPECIFIC EVENT or ACTION that would indicate this option is correct
- Use action verbs: "shows", "demonstrates", "explains", "performs"
- Example: "The narrator credits Whitehead with inventing the first airplane"
- Leave EMPTY if option is just an entity name

## CRITICAL RULES:
1. Keywords = ENTITIES (WHO, WHAT, WHERE, WHEN)
2. Semantic Queries = EVENTS (WHAT HAPPENS)
3. Be SPECIFIC - avoid generic descriptions
4. For each option, at least ONE of (keywords, semantic_query) MUST be non-empty

## Step-by-step guidance (follow closely):
1) Read the question and options.
2) Extract only VISUALIZABLE entities for keywords (names, objects, places, numbers).
3) If a concept is abstract, convert it into visible proxies (actions, objects, interactions).
4) Write semantic queries as short event sentences with clear actions and roles.
5) If you are unsure, keep keywords minimal rather than adding noisy terms.

## Good vs Bad Examples

### BAD Keywords (too generic):
❌ ["author", "person", "text:author"]
❌ ["motorized flight", "first try", "success"]
❌ ["text:A", "text:help", "text:each other"]

### GOOD Keywords (specific entities):
✓ ["John Brown", "text:John Brown"]
✓ ["Wright Brothers", "Kitty Hawk", "1903", "text:Wright Brothers"]
✓ ["Gustave Whitehead", "Connecticut", "1901"]
✓ ["orange", "bird", "Eiffel Tower", "text:HELP WANTED"]

### BAD Semantic Queries (too vague):
❌ "Person or institution named 'author' claims something"
❌ "Identification of the first plane inventor"

### GOOD Semantic Queries (specific events):
✓ "John Brown states in an interview that Whitehead flew first"
✓ "The video shows Wright Brothers' flight at Kitty Hawk with dated footage"
✓ "A museum exhibit or sign credits Whitehead with the first powered flight"

Output ONLY valid JSON in English, no other text."""

    PROMPT_TEMPLATE = """Input Query: "{query}"

Extract visual keywords with temporal and option decomposition.

Output JSON:
{{
  "query_keywords": ["keywords from the question itself..."],
  "temporal_plan": {{
    "start": ["intro visuals..."],
    "action": ["main event visuals..."],
    "result": ["outcome visuals..."]
  }},
  "option_keywords": {{
    "A": ["unique visual keywords for option A"],
    "B": ["unique visual keywords for option B"],
    "C": ["unique visual keywords for option C"],
    "D": ["unique visual keywords for option D"]
    "F": ["unique visual keywords for option F"]
  }},
  "vlm_query": "Clear question for VLM"
}}

JSON:"""

    # V4.3: Extended prompt template
    PROMPT_TEMPLATE_V5 = """Input Query: "{query}"

Extract ENTITIES (keywords) and EVENTS (semantic queries) for video retrieval.

REMEMBER:
- Keywords = SPECIFIC ENTITIES (names, places, objects, numbers)
- Semantic Queries = SPECIFIC EVENTS (what happens, who does what)
- NO generic words like "person", "statement", "video"

Output JSON:
{{
  "query_keywords": ["specific entities from question..."],
  "temporal_plan": {{
    "start": [],
    "action": [],
    "result": []
  }},
  "option_keywords": {{
    "A": ["specific entities for A"],
    "B": ["specific entities for B"],
    "C": ["specific entities for C"],
    "D": ["specific entities for D"]
    "F": ["specific entities for F"]
  }},
  "semantic_queries": {{
    "A": "specific event for A",
    "B": "specific event for B",
    "C": "specific event for C",
    "D": "specific event for D",
    "E": "specific event for E",
    "F": "specific event for F"
  }},
  "general_semantic_query": "what event the video must show",
  "vlm_query": "Clear question for VLM"
}}

JSON:"""

    def __init__(self, llm_client: LLMClient):
        """Initialize Query Processor."""
        self.llm_client = llm_client

    def _clean_query_entities(self, kws: List[str]) -> List[str]:
        """
        Post-filter LLM outputs for query_keywords so they are truly "entities".
        If the question contains no concrete entities, returning [] is preferred.
        """
        import re

        if not kws:
            return []

        # Strong stoplist for question-function / abstract words
        stop = {
            "based", "content", "whose", "viewpoint", "relatively", "according", "following",
            "statement", "correct", "incorrect", "true", "false", "options", "option", "answer",
            "video", "clip", "frame", "scene", "mainly", "about", "which", "what", "how", "who",
            "where", "when", "why", "of", "the", "a", "an", "in", "on", "at", "to", "for", "and",
            "or", "but", "not", "this", "that", "these", "those",
            "person", "someone", "something", "thing", "object", "item",
            "based on", "based-on",
        }

        def has_cjk(s: str) -> bool:
            return bool(re.search(r"[\u4e00-\u9fff]", s))

        out: List[str] = []
        seen = set()
        for kw in kws:
            s = (str(kw) if kw is not None else "").strip()
            if not s:
                continue
            sl = s.lower().strip()
            sl = re.sub(r"\s+", " ", sl)
            if sl in stop:
                continue

            # Keep explicit text tags / numbers / CJK proper nouns
            if sl.startswith("text:"):
                keep = True
            elif any(ch.isdigit() for ch in s):
                keep = True
            elif has_cjk(s) and len(s) >= 2:
                keep = True
            # Heuristic for English proper nouns / named entities: contains uppercase
            elif any(ch.isupper() for ch in s):
                keep = True
            else:
                # Otherwise treat as likely non-entity and drop (e.g., "viewpoint", "relatively")
                keep = False

            if keep and sl not in seen:
                seen.add(sl)
                out.append(s)

        return out
    
    def process(self, query: str) -> QueryResultV4:
        """
        Process a query with v4.0 multi-channel decomposition.
        
        Args:
            query: Original query string (question + options).
        
        Returns:
            QueryResultV4 with temporal_plan, option_keywords, query_keywords, vlm_query.
        """
        prompt = self.PROMPT_TEMPLATE.format(query=query)
        
        try:
            result = self.llm_client.chat_json(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT
            )
            
            temporal_plan = result.get("temporal_plan", {})
            option_keywords = result.get("option_keywords", {})
            query_keywords = result.get("query_keywords", [])
            vlm_query = result.get("vlm_query", query)
            
            # Ensure all temporal stages exist
            for stage in ["start", "action", "result"]:
                if stage not in temporal_plan:
                    temporal_plan[stage] = []
            
            # Ensure all options exist (best-effort parse from query; fallback A-D)
            all_opts = set(option_keywords.keys()) | set(self._extract_options_from_query(query).keys())
            if not all_opts:
                all_opts = {"A", "B", "C", "D"}
            for opt in all_opts:
                if opt not in option_keywords:
                    option_keywords[opt] = []
            
            # Fallback if empty
            all_keywords = []
            for kws in temporal_plan.values():
                all_keywords.extend(kws)
            if not all_keywords and not query_keywords:
                fallback_kw = self._fallback_extract_keywords(query)
                temporal_plan["action"] = fallback_kw
                # query_keywords should be entities; do not re-introduce generic words via fallback
                query_keywords = self._clean_query_entities(fallback_kw)
            
            return QueryResultV4(
                temporal_plan=temporal_plan,
                option_keywords=option_keywords,
                query_keywords=query_keywords,
                vlm_query=vlm_query,
                original_query=query
            )
        
        except Exception as e:
            print(f"Query processing failed: {e}. Using fallback.")
            fallback_kw = self._fallback_extract_keywords(query)
            return QueryResultV4(
                temporal_plan={"start": [], "action": fallback_kw, "result": []},
                option_keywords={"A": [], "B": [], "C": [], "D": []},
                query_keywords=fallback_kw,
                vlm_query=query,
                original_query=query
            )
    
    def process_v5(self, query: str) -> QueryResultV5:
        """
        Process a query with v4.3 multi-route decomposition.
        
        Extends v4.0 processing with semantic queries for text-based retrieval.
        
        Args:
            query: Original query string (question + options).
        
        Returns:
            QueryResultV5 with all v4 fields plus semantic_queries and general_semantic_query.
        """
        prompt = self.PROMPT_TEMPLATE_V5.format(query=query)
        
        try:
            result = self.llm_client.chat_json(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT_V5
            )
            
            temporal_plan = result.get("temporal_plan", {})
            option_keywords = result.get("option_keywords", {})
            query_keywords = self._clean_query_entities(result.get("query_keywords", []) or [])
            vlm_query = result.get("vlm_query", query)
            
            # V4.3: Semantic queries
            semantic_queries = result.get("semantic_queries", {})
            general_semantic_query = result.get("general_semantic_query", "")
            
            # Ensure all temporal stages exist
            for stage in ["start", "action", "result"]:
                if stage not in temporal_plan:
                    temporal_plan[stage] = []
            
            # V4.3.1: Support dynamic options (A-Z, not just A-D)
            # Get all option keys from option_keywords and semantic_queries
            parsed_opts = set(self._extract_options_from_query(query).keys())
            all_option_keys = set(option_keywords.keys()) | set(semantic_queries.keys()) | parsed_opts
            
            # If no options found, add default A-D
            if not all_option_keys:
                all_option_keys = {"A", "B", "C", "D"}
            
            # Ensure each option has both keywords and semantic query
            for opt in all_option_keys:
                if opt not in option_keywords:
                    option_keywords[opt] = []
                if opt not in semantic_queries:
                    semantic_queries[opt] = ""

            # V4.3.2: Semantic query format cleanup & non-empty guarantee per option.
            # - Remove redundant prefix like "If answer is A, the video should show:"
            # - If BOTH option_keywords and semantic_query are empty, backfill keywords from option text.
            opt_texts = self._extract_options_from_query(query)
            for opt in list(all_option_keys):
                sq = (semantic_queries.get(opt) or "").strip()
                # Strip common leading patterns
                for pref in (
                    f"If answer is {opt},",
                    f"If the answer is {opt},",
                    f"If the correct answer is {opt},",
                    f"If answer is {opt} ",
                    f"If the answer is {opt} ",
                    f"If the correct answer is {opt} ",
                ):
                    if sq.lower().startswith(pref.lower()):
                        sq = sq[len(pref):].strip()
                # Also strip a generic "the video should show:" prefix
                if sq.lower().startswith("the video should show"):
                    sq = sq.split(":", 1)[-1].strip() if ":" in sq else ""
                semantic_queries[opt] = sq

                kws = option_keywords.get(opt) or []
                if (not kws) and (not sq):
                    txt = (opt_texts.get(opt) or "").strip()
                    # Backfill at least something searchable
                    backfill = self._fallback_extract_keywords(txt) if txt else [f"option_{opt}"]
                    option_keywords[opt] = backfill[:5]
            
            # Fallback if empty
            all_keywords = []
            for kws in temporal_plan.values():
                all_keywords.extend(kws)
            if not all_keywords and not query_keywords:
                fallback_kw = self._fallback_extract_keywords(query)
                temporal_plan["action"] = fallback_kw
                query_keywords = self._clean_query_entities(fallback_kw)
            
            return QueryResultV5(
                temporal_plan=temporal_plan,
                option_keywords=option_keywords,
                query_keywords=query_keywords,
                vlm_query=vlm_query,
                original_query=query,
                semantic_queries=semantic_queries,
                general_semantic_query=general_semantic_query
            )
        
        except Exception as e:
            print(f"Query processing v5 failed: {e}. Falling back to v4.")
            # Fallback to v4 processing and wrap in v5
            v4_result = self.process(query)
            return QueryResultV5(
                temporal_plan=v4_result.temporal_plan,
                option_keywords=v4_result.option_keywords,
                query_keywords=v4_result.query_keywords,
                vlm_query=v4_result.vlm_query,
                original_query=v4_result.original_query,
                semantic_queries={},
                general_semantic_query=""
            )
    
    def _fallback_extract_keywords(self, query: str) -> List[str]:
        """Fallback keyword extraction using simple heuristics."""
        stop_words = {
            "what", "which", "who", "where", "when", "why", "how",
            "does", "do", "did", "is", "are", "was", "were",
            "the", "a", "an", "of", "in", "on", "at", "to", "for",
            "and", "or", "but", "not", "this", "that", "these", "those",
            "option", "answer", "question", "following", "correct",
            "video", "clip", "frame", "scene",
            # common generic scaffolding words seen in rewritten query keywords
            "based", "content", "whose", "viewpoint", "relatively", "according", "mainly", "about"
        }
        
        words = query.lower().split()
        keywords = []
        
        for word in words:
            word = word.strip(".,?!;:'\"()[]")
            if len(word) > 2 and word not in stop_words:
                keywords.append(word)
        
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords[:5]
    
    def _extract_options_from_query(self, query: str) -> Dict[str, str]:
        """Extract option text from query string."""
        import re
        import ast

        options: Dict[str, str] = {}
        q = query or ""

        # 1) Parse python-list style: Options: ['A. xxx', 'B. yyy', ...]
        try:
            if "Options:" in q and "[" in q and "]" in q:
                start = q.find("[")
                end = q.rfind("]")
                if 0 <= start < end:
                    sub = q[start : end + 1]
                    maybe_list = ast.literal_eval(sub)
                    if isinstance(maybe_list, (list, tuple)):
                        for item in maybe_list:
                            s = str(item).strip()
                            m = re.match(r"^\s*([A-Z])\s*[\.\):]\s*(.*)$", s)
                            if m:
                                options[m.group(1).upper()] = m.group(2).strip()
        except Exception:
            pass

        # 2) Parse line-prefix style: "A. xxx" / "B) xxx" / "(C): xxx"
        for line in q.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^\s*\(?\s*([A-Z])\s*\)?\s*[\.\):]\s*(.+)$", line)
            if m:
                options[m.group(1).upper()] = m.group(2).strip()

        return options


def extract_keywords_simple(query: str) -> List[str]:
    """Simple keyword extraction without LLM."""
    processor = QueryProcessor.__new__(QueryProcessor)
    return processor._fallback_extract_keywords(query)
