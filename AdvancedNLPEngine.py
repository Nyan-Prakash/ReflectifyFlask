import re
import torch
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set, Union
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util
import spacy
import os
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from collections import Counter, defaultdict
import networkx as nx
from itertools import combinations
import json
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedNLPEngine:
    """Enhanced NLP engine with deep narrative understanding for personal journals."""
    
    def __init__(self, user_id=None, memory_path=None):
        """
        Initialize NLP models and resources.
        
        Args:
            user_id (str, optional): Unique identifier for the user to enable personalization
            memory_path (str, optional): Path to store persistent memory data
        """
        # User-specific data for personalization
        self.user_id = user_id
        self.memory_path = memory_path
        self.user_entities = defaultdict(lambda: {"count": 0, "metadata": {}, "last_mentioned": None})
        self.user_concepts = defaultdict(lambda: {"count": 0, "related_to": set(), "first_mentioned": None})
        self.relationship_graph = nx.Graph()
        self.narrative_memory = []
        self.emotion_history = []
        
        # Load memory if available
        if user_id and memory_path:
            self._load_memory()
        
        # Load spaCy model - prioritize larger models for better accuracy
        self._load_spacy()
        
        # Initialize sentence transformer for semantic embeddings
        try:
            # Use a more powerful model for better semantic understanding
            self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        except Exception as e:
            logger.warning(f"Error loading mpnet model, falling back to MiniLM: {e}")
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e2:
                logger.error(f"Error loading embedding model: {e2}")
                self.embedding_model = None
        
        # Initialize emotion analysis
        self._init_emotion_analysis()
        
        # Initialize narrative understanding components
        self._init_narrative_components()
        
        # Temporal expression recognizer
        self._init_temporal_understanding()
        
        # Context tracker for multi-entry analysis
        self.context_tracker = {
            "ongoing_narratives": [],
            "recent_entities": [],
            "recent_topics": [],
            "recent_locations": [],
            "implicit_references": {},
            "emotional_state": {"dominant": None, "intensity": 0, "trend": None}
        }
        
        # Define enhanced event indicators
        self._define_event_indicators()
    
    def _load_spacy(self):
        """Load appropriate spaCy model with prioritized fallbacks."""
        # Prioritized list of models to try, from most to least sophisticated
        spacy_models = ["en_core_web_trf", "en_core_web_lg", "en_core_web_md", "en_core_web_sm"]
        
        for model in spacy_models:
            try:
                self.nlp = spacy.load(model)
                logger.info(f"Successfully loaded spaCy model: {model}")
                
                # Add special pipes for advanced analysis if using lg or trf model
                if model in ["en_core_web_trf", "en_core_web_lg"]:
                    if "entity_ruler" not in self.nlp.pipe_names:
                        self.nlp.add_pipe("entity_ruler", before="ner")
                        # Could add custom entity patterns here
                
                return
            except OSError:
                try:
                    # Try to download the model
                    import subprocess
                    import sys
                    logger.info(f"Downloading spaCy model: {model}")
                    subprocess.run([sys.executable, "-m", "spacy", "download", model], check=True)
                    self.nlp = spacy.load(model)
                    logger.info(f"Successfully downloaded and loaded: {model}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load or download {model}: {e}")
                    continue
        
        # If all attempts fail, raise an error
        raise RuntimeError("Could not load any spaCy model. Please install spaCy and download a model.")
    
    def _init_emotion_analysis(self):
        """Initialize more sophisticated emotion analysis tools."""
        try:
            # Primary sentiment analyzer - fine-tuned for nuance
            os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            # More detailed emotion analyzer
            try:
                # Using a model fine-tuned for emotion detection with 7 emotions
                self.emotion_analyzer = pipeline(
                    "text-classification", 
                    model="bhadresh-savani/distilbert-base-uncased-emotion"
                )
            except Exception as e:
                logger.warning(f"Error loading emotion model, using simplified emotion detection: {e}")
                self.emotion_analyzer = None
                
            # Emotional intensity analyzer
            self.intensity_analyzer = None  # Will use lexicon-based approach
            
            # Load emotion lexicon
            self.emotion_lexicon = self._load_emotion_lexicon()
            
        except Exception as e:
            logger.error(f"Error initializing emotion analysis: {e}")
            self.sentiment_analyzer = None
            self.emotion_analyzer = None
            self.intensity_analyzer = None
    
    def _load_emotion_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Load or create an emotion lexicon with intensity scores."""
        # Basic emotion lexicon - in production, this would be loaded from a file
        lexicon = {
            "joy": {
                "ecstatic": 0.9, "thrilled": 0.85, "delighted": 0.8, "happy": 0.7,
                "pleased": 0.6, "glad": 0.5, "content": 0.4, "satisfied": 0.35
            },
            "sadness": {
                "devastated": 0.9, "heartbroken": 0.85, "depressed": 0.8, "miserable": 0.75,
                "sad": 0.7, "unhappy": 0.6, "disappointed": 0.5, "downhearted": 0.4
            },
            "anger": {
                "furious": 0.9, "enraged": 0.85, "outraged": 0.8, "angry": 0.7,
                "annoyed": 0.5, "irritated": 0.4, "bothered": 0.3, "displeased": 0.2
            },
            "fear": {
                "terrified": 0.9, "horrified": 0.85, "panicked": 0.8, "scared": 0.7,
                "afraid": 0.6, "nervous": 0.5, "anxious": 0.4, "uneasy": 0.3
            },
            "surprise": {
                "astonished": 0.9, "astounded": 0.85, "amazed": 0.8, "shocked": 0.75,
                "surprised": 0.6, "startled": 0.5, "taken aback": 0.4
            },
            "disgust": {
                "repulsed": 0.9, "revolted": 0.85, "disgusted": 0.8, "appalled": 0.7,
                "offended": 0.5, "put off": 0.4, "displeased": 0.3
            },
            "trust": {
                "absolute trust": 0.9, "complete faith": 0.8, "confident": 0.7,
                "trust": 0.6, "believe in": 0.5, "rely on": 0.4
            },
            "anticipation": {
                "excited": 0.8, "eager": 0.7, "looking forward": 0.6, "anticipating": 0.5,
                "expecting": 0.4, "awaiting": 0.3
            }
        }
        
        # Add negations and intensifiers
        self.negations = {"not", "never", "no", "isn't", "aren't", "wasn't", "weren't", 
                         "don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't", 
                         "shouldn't", "can't", "cannot", "hardly", "barely", "rarely"}
        
        self.intensifiers = {"very", "extremely", "really", "absolutely", "completely", 
                           "totally", "utterly", "so", "quite", "pretty", "rather", 
                           "particularly", "especially", "incredibly", "unbelievably"}
        
        return lexicon
    
    def _init_narrative_components(self):
        """Initialize components for narrative understanding."""
        # Narrative arc detection
        self.narrative_arcs = {
            "BEGINNING": ["started", "began", "initiated", "launched", "commenced", "embarked", 
                         "first time", "new", "initially", "at first", "starting"],
            "MIDDLE": ["continuing", "ongoing", "developing", "progressing", "in the middle of", 
                      "working on", "still", "during", "throughout", "while"],
            "CLIMAX": ["finally", "eventually", "at last", "critical moment", "turning point", 
                      "culminated", "reached", "breakthrough", "pivotal", "decisive"],
            "RESOLUTION": ["resolved", "concluded", "ended", "finished", "completed", "solved", 
                          "settled", "fixed", "wrapped up", "finalized", "accomplished"]
        }
        
        # Narrative roles (characters/entities in the story)
        self.narrative_roles = {
            "PROTAGONIST": ["I", "me", "my", "myself"],
            "SUPPORTING": ["helped", "supported", "assisted", "collaborated", "joined"],
            "ANTAGONIST": ["against", "opposed", "blocked", "prevented", "stopped", 
                          "challenged", "difficult", "problem with"]
        }
        
        # Causal signals
        self.causal_signals = {
            "CAUSE": ["because", "since", "as", "due to", "thanks to", "caused by", 
                     "resulted from", "stemmed from", "led to", "created", "generated"],
            "EFFECT": ["therefore", "thus", "consequently", "as a result", "so", "hence", 
                      "accordingly", "that's why", "which meant that", "resulting in"]
        }
        
        # Important narrative elements to track
        self.narrative_elements = {
            "GOAL": ["goal", "aim", "objective", "purpose", "target", "aspiration", "intention", 
                    "hope", "plan", "try", "trying", "attempted", "wanted to", "hoping to", "wished to"],
            "CONFLICTS": ["conflicts", "problem", "issue", "challenge", "difficulty", "obstacle", 
                        "struggle", "barrier", "setback", "hurdle", "tension", "disagreement", "argued"],
            "LEARNING": ["learned", "realized", "understood", "discovered", "found out", 
                        "recognized", "came to understand", "insight", "revelation", "epiphany"],
            "EMOTION": ["felt", "feeling", "emotion", "emotional", "mood", "attitude", 
                       "reaction", "response", "affected", "moved", "touched"]
        }
    
    def _init_temporal_understanding(self):
        """Initialize components for temporal understanding."""
        # Temporal expressions and patterns
        self.date_patterns = [
            # Explicit dates (yyyy-mm-dd, mm/dd/yyyy, etc.)
            r'\d{4}-\d{1,2}-\d{1,2}',
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{1,2}-\d{1,2}-\d{2,4}',
            
            # Month day, year
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}',
            
            # Month day
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?',
            
            # Day of week
            r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Tues|Wed|Thu|Thurs|Fri|Sat|Sun)\.?'
        ]
        
        # Temporal references (relative)
        self.temporal_references = {
            "PAST": ["yesterday", "last week", "last month", "last year", "ago", "previously", 
                    "earlier", "before", "used to", "had been", "once", "formerly"],
            "PRESENT": ["today", "now", "currently", "presently", "at the moment", 
                       "this week", "this month", "this year", "these days", "lately"],
            "FUTURE": ["tomorrow", "next week", "next month", "next year", "soon", 
                      "in the future", "upcoming", "going to", "will", "planning to", 
                      "looking forward to", "expecting", "anticipating"]
        }
        
        # Time expressions
        self.time_patterns = [
            # HH:MM 12-hour
            r'\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)',
            
            # HH:MM 24-hour
            r'\d{1,2}:\d{2}',
            
            # Approximate times
            r'(?:morning|afternoon|evening|night|dawn|dusk|noon|midnight)'
        ]
        
        # Duration expressions
        self.duration_patterns = [
            r'(?:for|during)?\s*(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s*(?:second|minute|hour|day|week|month|year)s?',
            r'(?:all day|all night|all morning|all evening|all afternoon)',
            r'(?:a few|several|many)\s*(?:second|minute|hour|day|week|month|year)s?'
        ]
        
        # Frequency expressions
        self.frequency_patterns = [
            r'(?:every|each)\s*(?:day|week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)',
            r'(?:once|twice|three times|four times|five times|daily|weekly|monthly|yearly)',
            r'(?:occasionally|sometimes|often|frequently|rarely|seldom|never|always)'
        ]
    
    def _define_event_indicators(self):
        """Define enhanced event indicators for detection."""
        # Base event indicators from original code
        base_indicators = [
            # Work events
            "meeting", "project", "presentation", "call with", "interviewed", 
            "hired", "promoted", "deadline", "completed", "launched", "started on",
            "working on", "collaborated", "finished", "submitted", "published",
            
            # Social events
            "went to", "attended", "party", "dinner", "lunch", "breakfast", 
            "met with", "visited", "celebrated", "date with", "drinks with",
            "saw", "talked with", "spoke to", "hung out", "spent time with",
            
            # Travel events
            "trip to", "flew to", "drove to", "traveled to", "arrived in",
            "returned from", "stayed at", "visited", "exploring", "vacation in",
            
            # Health events
            "appointment with", "doctor's appointment", "therapy session",
            "workout", "exercise", "diagnosed with", "recovered from",
            "started taking", "stopped taking", "feeling sick", "feeling better",
            
            # Milestone events
            "anniversary", "birthday", "graduation", "wedding", "engagement",
            "moved to", "bought a", "sold my", "signed up for", "enrolled in",
            "started dating", "broke up", "got married", "got divorced",
            
            # Academic/Test events
            "test", "exam", "quiz", "midterm", "final", "homework", "assignment",
            "studied for", "prepared for", "took a", "wrote a", "passed", "failed",
            "got a grade", "received results", "scored", "got an A", "got a B",
            "got a C", "class", "lecture", "course", "professor", "teacher",
            
            # Follow-up indicators
            "results", "feedback", "review", "evaluation", "assessment",
            "response", "reply", "answered", "heard back", "received",
            "got back", "followed up", "update on", "news about", "decision on"
        ]
        
        # Enhanced indicators for personal narratives
        narrative_indicators = [
            # Personal reflections
            "realized", "thought about", "reflected on", "considered", "pondered",
            "contemplated", "meditated on", "looked back on", "reconsidered",
            
            # Emotional moments
            "felt", "experienced", "overcame", "struggled with", "coped with",
            "processed", "dealt with", "handled", "managed", "worked through",
            
            # Relationship events
            "reconnected with", "reached out to", "caught up with", "made up with",
            "apologized to", "forgave", "confided in", "trusted", "opened up to",
            "bonded with", "connected with", "got closer to", "drifted apart from",
            
            # Personal growth
            "improved", "grew", "developed", "evolved", "advanced", "progressed",
            "mastered", "practiced", "trained", "invested in", "committed to",
            
            # Creative activities
            "created", "made", "built", "designed", "wrote", "composed", "painted",
            "drew", "crafted", "produced", "directed", "performed", "played",
            
            # Discoveries
            "found", "discovered", "uncovered", "came across", "stumbled upon",
            "realized", "noticed", "observed", "spotted", "identified", "recognized",
            
            # Decisions
            "decided", "chose", "selected", "opted", "determined", "resolved",
            "committed to", "settled on", "made up my mind", "figured out",
            
            # Achievements
            "accomplished", "achieved", "succeeded", "won", "earned", "gained",
            "obtained", "secured", "locked in", "clinched", "nailed", "aced",
            
            # Life maintenance
            "cleaned", "organized", "fixed", "repaired", "maintained", "updated",
            "replaced", "upgraded", "installed", "set up", "arranged", "sorted out"
        ]
        
        # Combine all indicators
        self.event_indicators = base_indicators + narrative_indicators
        
        # Group indicators by category for better classification
        self.categorized_indicators = {
            "WORK": ["meeting", "project", "presentation", "deadline", "client", "colleague", 
                    "interview", "report", "email", "boss", "manager", "supervisor", "team", 
                    "worked", "job", "office", "workplace", "career", "professional", "business"],
            
            "SOCIAL": ["friend", "party", "dinner", "lunch", "coffee", "drinks", "date", 
                      "celebration", "gathering", "wedding", "met with", "hung out", "talked with", 
                      "visited", "spent time with", "social", "chat", "conversation", "meet up"],
            
            "HEALTH": ["doctor", "appointment", "therapy", "workout", "exercise", "gym", 
                      "run", "fitness", "health", "sick", "illness", "medicine", "symptoms", 
                      "treatment", "recovery", "mental health", "physical", "diet", "nutrition"],
            
            "EDUCATION": ["class", "course", "lecture", "study", "learn", "homework", 
                         "assignment", "exam", "test", "quiz", "grade", "school", "college", 
                         "university", "professor", "teacher", "student", "education", "academic"],
            
            "CREATIVE": ["write", "wrote", "create", "design", "draw", "paint", "compose", 
                        "play", "perform", "practice", "art", "music", "photography", "creative", 
                        "hobby", "project", "craft", "make", "build", "develop"],
            
            "TRAVEL": ["trip", "travel", "visit", "flight", "drive", "journey", 
                      "vacation", "holiday", "tour", "explore", "sight", "destination", 
                      "place", "country", "city", "hotel", "airport", "abroad"],
            
            "CHORES": ["clean", "organize", "tidy", "wash", "laundry", "dishes", 
                      "groceries", "shopping", "cook", "prepare", "meal", "household", 
                      "chore", "task", "errand", "maintenance", "fix", "repair"],
            
            "TECH": ["computer", "phone", "app", "website", "online", "software", 
                    "hardware", "device", "digital", "technology", "internet", "web", 
                    "program", "code", "develop", "tech", "IT", "system", "platform"],
            
            "FINANCIAL": ["money", "pay", "purchase", "buy", "sell", "budget", 
                         "finance", "financial", "invest", "bank", "account", "bill", 
                         "expense", "income", "salary", "save", "spend", "cost", "fund"],
            
            "FAMILY": ["family", "parent", "child", "kid", "mother", "father", "mom", 
                      "dad", "brother", "sister", "sibling", "grandparent", "relative", 
                      "aunt", "uncle", "cousin", "nephew", "niece", "husband", "wife", "spouse"],
            
            "PERSONAL": ["reflection", "thought", "feeling", "emotion", "personal", 
                        "growth", "development", "self", "goal", "plan", "decision", 
                        "choice", "change", "challenge", "struggle", "achievement", "success"]
        }
    
    def _load_memory(self):
        """Load user-specific memory if available."""
        if not self.user_id or not self.memory_path:
            return
            
        memory_file = os.path.join(self.memory_path, f"{self.user_id}_memory.json")
        try:
            if os.path.exists(memory_file):
                with open(memory_file, 'r') as f:
                    memory_data = json.load(f)
                    
                    # Load user entities
                    if "entities" in memory_data:
                        self.user_entities = defaultdict(lambda: {"count": 0, "metadata": {}, "last_mentioned": None})
                        for entity, data in memory_data["entities"].items():
                            self.user_entities[entity] = data
                    
                    # Load user concepts
                    if "concepts" in memory_data:
                        self.user_concepts = defaultdict(lambda: {"count": 0, "related_to": set(), "first_mentioned": None})
                        for concept, data in memory_data["concepts"].items():
                            data["related_to"] = set(data["related_to"]) if "related_to" in data else set()
                            self.user_concepts[concept] = data
                    
                    # Load relationship graph
                    if "relationships" in memory_data:
                        self.relationship_graph = nx.node_link_graph(memory_data["relationships"])
                    
                    # Load narrative memory
                    if "narrative_memory" in memory_data:
                        self.narrative_memory = memory_data["narrative_memory"]
                    
                    # Load emotion history
                    if "emotion_history" in memory_data:
                        self.emotion_history = memory_data["emotion_history"]
                    
                    logger.info(f"Loaded memory for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error loading memory for user {self.user_id}: {e}")
    
    def _save_memory(self):
        """Save user-specific memory."""
        if not self.user_id or not self.memory_path:
            return
            
        # Ensure directory exists
        os.makedirs(self.memory_path, exist_ok=True)
        
        memory_file = os.path.join(self.memory_path, f"{self.user_id}_memory.json")
        try:
            # Prepare data for serialization
            memory_data = {
                "entities": dict(self.user_entities),
                "concepts": {k: {"count": v["count"], 
                                "related_to": list(v["related_to"]), 
                                "first_mentioned": v["first_mentioned"]} 
                             for k, v in self.user_concepts.items()},
                "relationships": nx.node_link_data(self.relationship_graph),
                "narrative_memory": self.narrative_memory[-100:],  # Keep only recent memories
                "emotion_history": self.emotion_history[-50:]  # Keep only recent emotions
            }
            
            with open(memory_file, 'w') as f:
                json.dump(memory_data, f)
                
            logger.info(f"Saved memory for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error saving memory for user {self.user_id}: {e}")
    
    def process_entry(self, text: str, entry_date: Optional[str] = None, entry_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a journal entry and extract comprehensive structured information.
        
        Args:
            text (str): The journal entry text
            entry_date (str, optional): Date of the journal entry (ISO format)
            entry_id (str, optional): Unique identifier for the entry
            
        Returns:
            Dict[str, Any]: Structured data extracted from the entry
        """
        # Process with spaCy for basic NLP analysis
        doc = self.nlp(text)
        
        # Extract core components (from original process_entry)
        entities = self._extract_entities(doc)
        events = self._extract_events(doc)
        embedding = self._generate_embedding(text)
        sentiment = self._analyze_sentiment(text)
        topics = self._detect_topics(doc)
        
        # Enhanced extraction (new components)
        emotions = self._analyze_emotions(text, doc)
        temporal_info = self._extract_temporal_information(text, doc)
        narrative_elements = self._extract_narrative_elements(text, doc)
        relationships = self._extract_relationships(doc, entities)
        reflections = self._detect_reflections(text, doc)
        actions = self._extract_key_actions(doc)
        questions = self._extract_questions(doc)
        
        # Process places and locations
        locations = self._extract_locations(doc, entities)
        
        # Update user-specific memory and context
        self._update_memory(text, entities, events, emotions, topics, entry_date)
        
        # Detect recurring themes
        recurring_themes = self._detect_recurring_themes(text, topics, entities)
        
        # Detect implicit references and context
        implicit_references = self._detect_implicit_references(text, doc)
        
        # Analyze causal relationships
        causal_relationships = self._extract_causal_relationships(text, doc)
        
        # Summarize the entry
        summary = self._generate_summary(text, entities, events, emotions, topics)
        
        # Structure the complete response
        structured_data = {
            "meta": {
                "entry_id": entry_id,
                "entry_date": entry_date,
                "processing_date": datetime.now().isoformat(),
                "word_count": len(text.split()),
                "processing_version": "2.0"
            },
            "core": {
                "entities": entities,
                "events": events,
                "sentiment": sentiment,
                "topics": topics,
                "embedding": embedding
            },
            "enhanced": {
                "emotions": emotions,
                "temporal_information": temporal_info,
                "narrative_elements": narrative_elements,
                "relationships": relationships,
                "locations": locations,
                "reflections": reflections,
                "actions": actions,
                "questions": questions
            },
            "contextual": {
                "recurring_themes": recurring_themes,
                "implicit_references": implicit_references,
                "causal_relationships": causal_relationships,
            },
            "summary": summary
        }
        
        return structured_data
    
    def _extract_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract named entities with enhanced categorization and metadata."""
        entities = []
        seen = set()  # To avoid duplicates
        
        for ent in doc.ents:
            # Skip very short entities or non-informative entities
            if len(ent.text) < 2 or ent.text.lower() in ["i", "me", "you", "we", "us", "they", "them"]:
                continue
                
            # Skip duplicate entities
            if ent.text.lower() in seen:
                continue
                
            seen.add(ent.text.lower())
            
            # Get more detailed entity subtype and metadata
            entity_type, entity_metadata = self._enhance_entity_type(ent)
            
            entities.append({
                "text": ent.text,
                "type": entity_type,
                "subtype": self._get_entity_subtype(ent),
                "metadata": entity_metadata,
                "start": ent.start_char,
                "end": ent.end_char,
                "sentiment": self._get_entity_sentiment(ent, doc),
                "mentions": self._count_entity_mentions(ent.text.lower(), doc)
            })
        
        # Look for potential entities missed by spaCy
        additional_entities = self._extract_additional_entities(doc, seen)
        entities.extend(additional_entities)
        
        # Update user-specific entity memory
        self._update_entity_memory(entities)
        
        return entities
    
    def _enhance_entity_type(self, ent) -> Tuple[str, Dict[str, Any]]:
        """Enhance entity type with more specific categorization."""
        entity_type = ent.label_
        metadata = {}
        
        # Further categorize person entities
        if entity_type == "PERSON":
            # Check for family relationships
            family_terms = ["mother", "father", "mom", "dad", "brother", "sister", "son", 
                           "daughter", "aunt", "uncle", "cousin", "grandmother", "grandfather",
                           "grandma", "grandpa", "wife", "husband", "spouse", "partner"]
            
            # Check if any family terms appear near the entity
            sent = ent.sent
            sent_text = sent.text.lower()
            
            is_family = any(term in sent_text for term in family_terms)
            if is_family:
                entity_type = "PERSON_FAMILY"
                # Try to identify specific family relationship
                for term in family_terms:
                    if term in sent_text:
                        # Find closest occurrence of term to entity
                        metadata["relationship"] = term
                        break
            
            # Check for work relationships
            work_terms = ["colleague", "coworker", "boss", "manager", "employee", "supervisor",
                         "client", "customer", "vendor", "partner", "teammate", "leader"]
            
            is_work = any(term in sent_text for term in work_terms)
            if is_work:
                entity_type = "PERSON_PROFESSIONAL"
                for term in work_terms:
                    if term in sent_text:
                        metadata["relationship"] = term
                        break
            
            # Check if this is a friend
            friend_terms = ["friend", "buddy", "pal", "mate", "bestie", "best friend"]
            is_friend = any(term in sent_text for term in friend_terms)
            if is_friend:
                entity_type = "PERSON_FRIEND"
        
        # Further categorize location entities
        elif entity_type == "GPE" or entity_type == "LOC":
            # Check if it's a home, work, or frequented place
            place_indicators = {
                "home": ["home", "house", "apartment", "flat", "condo", "my place", "where I live"],
                "work": ["office", "workplace", "work", "job", "company", "where I work", "headquarters"],
                "school": ["school", "university", "college", "campus", "classroom", "lecture hall"],
                "frequent": ["usual", "favorite", "regular", "often", "always", "typically", "normally"]
            }
            
            sent_text = ent.sent.text.lower()
            
            for place_type, indicators in place_indicators.items():
                if any(indicator in sent_text for indicator in indicators):
                    metadata["place_type"] = place_type
                    break
        
        # Enhance organization entities
        elif entity_type == "ORG":
            org_types = {
                "employer": ["work for", "my job", "my company", "employed at", "my employer"],
                "education": ["study at", "attend", "my school", "my university", "my college"],
                "healthcare": ["hospital", "clinic", "medical center", "doctor's office"],
                "business": ["store", "shop", "restaurant", "cafe", "business", "vendor"]
            }
            
            sent_text = ent.sent.text.lower()
            
            for org_type, indicators in org_types.items():
                if any(indicator in sent_text for indicator in indicators):
                    metadata["organization_type"] = org_type
                    break
        
        # Check if entity is mentioned in user's memory
        if self.user_id and ent.text in self.user_entities:
            metadata["frequency"] = self.user_entities[ent.text]["count"]
            metadata["last_mentioned"] = self.user_entities[ent.text]["last_mentioned"]
            
            # Add stored metadata from user memory
            for key, value in self.user_entities[ent.text]["metadata"].items():
                if key not in metadata:
                    metadata[key] = value
        
        return entity_type, metadata
    
    def _get_entity_subtype(self, ent) -> Optional[str]:
        """Get more specific entity subtype when possible."""
        if ent.label_ == "DATE":
            # Distinguish between absolute and relative dates
            if any(word in ent.text.lower() for word in ["yesterday", "today", "tomorrow", "last", "next"]):
                return "RELATIVE_DATE"
            else:
                return "ABSOLUTE_DATE"
        
        elif ent.label_ == "TIME":
            # Distinguish between specific and general times
            if any(char.isdigit() for char in ent.text):
                return "SPECIFIC_TIME"
            else:
                return "GENERAL_TIME"
        
        elif ent.label_ == "GPE":
            # Distinguish between different GPE types
            city_indicators = ["city", "town", "village", "suburb"]
            country_indicators = ["country", "nation", "state"]
            
            sent = ent.sent.text.lower()
            
            if any(indicator in sent for indicator in city_indicators):
                return "CITY"
            elif any(indicator in sent for indicator in country_indicators):
                return "COUNTRY"
        
        elif ent.label_ == "ORG":
            # Distinguish between company, educational institution, etc.
            if any(edu_term in ent.text.lower() for edu_term in ["university", "college", "school", "academy"]):
                return "EDUCATIONAL"
            elif any(corp_term in ent.text.lower() for corp_term in ["inc", "corp", "ltd", "llc", "company"]):
                return "COMPANY"
            elif any(gov_term in ent.text.lower() for gov_term in ["department", "ministry", "agency", "government"]):
                return "GOVERNMENT"
        
        # Default to None if no subtype identified
        return None
    
    def _get_entity_sentiment(self, ent, doc) -> Dict[str, Any]:
        """Analyze sentiment associated with this specific entity in context."""
        sent = ent.sent
        
        # Look for sentiment modifiers near the entity
        entity_range = range(max(0, ent.start - 5), min(len(doc), ent.end + 5))
        nearby_tokens = [doc[i] for i in entity_range]
        
        # Check for sentiment-bearing words near the entity
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", 
                         "positive", "happy", "joy", "love", "like", "enjoy", "beautiful"]
        
        negative_words = ["bad", "terrible", "awful", "horrible", "poor", "negative", 
                         "sad", "angry", "hate", "dislike", "annoying", "frustrating"]
        
        # Get sentiment word counts
        pos_count = sum(1 for token in nearby_tokens if token.text.lower() in positive_words)
        neg_count = sum(1 for token in nearby_tokens if token.text.lower() in negative_words)
        
        # Check for negations that could flip sentiment
        negation_count = sum(1 for token in nearby_tokens if token.text.lower() in ["not", "never", "no"])
        
        # Calculate base sentiment
        total = pos_count + neg_count
        if total == 0:
            sentiment = "NEUTRAL"
            score = 0.5
        else:
            pos_ratio = pos_count / total
            # Apply negation effects
            if negation_count > 0:
                pos_ratio = 1 - pos_ratio  # Flip the sentiment
            
            if pos_ratio > 0.6:
                sentiment = "POSITIVE"
                score = min(0.5 + pos_ratio * 0.5, 1.0)
            elif pos_ratio < 0.4:
                sentiment = "NEGATIVE"
                score = max(0.5 - (1 - pos_ratio) * 0.5, 0.0)
            else:
                sentiment = "NEUTRAL"
                score = 0.5
        
        return {
            "label": sentiment,
            "score": score,
            "context": sent.text
        }
    
    def _count_entity_mentions(self, entity_text: str, doc) -> int:
        """Count how many times an entity (or its references) are mentioned."""
        # Direct matches
        direct_matches = len([token for token in doc if token.text.lower() == entity_text])
        
        # TODO: In a more advanced implementation, we could also look for pronouns
        # and other references to the entity throughout the document
        
        return direct_matches
    
    def _extract_additional_entities(self, doc, seen: Set[str]) -> List[Dict[str, Any]]:
        """
        Extract potential entities that might have been missed by the NER model.
        Looks for patterns like proper nouns, capitalized phrases, etc.
        """
        additional_entities = []
        
        # Look for consecutive proper nouns that weren't already captured as entities
        i = 0
        while i < len(doc):
            # If we find a proper noun that's not part of an existing entity
            if doc[i].pos_ == "PROPN" and not any(doc[i].text.lower() == entity.lower() for entity in seen):
                # Check if it's the start of a multi-token proper noun
                start_idx = i
                while i + 1 < len(doc) and doc[i + 1].pos_ == "PROPN":
                    i += 1
                
                # If it's a reasonable length and not just a single letter
                if i - start_idx > 0 or len(doc[start_idx].text) > 1:
                    entity_text = doc[start_idx:i+1].text
                    
                    # Skip if already seen or if it's a single letter
                    if entity_text.lower() in seen or len(entity_text) <= 1:
                        i += 1
                        continue
                    
                    seen.add(entity_text.lower())
                    
                    # Try to determine entity type
                    entity_type = self._guess_entity_type(entity_text, doc[start_idx:i+1])
                    
                    additional_entities.append({
                        "text": entity_text,
                        "type": entity_type,
                        "subtype": None,
                        "metadata": {"detected": "pattern"},
                        "start": doc[start_idx].idx,
                        "end": doc[i].idx + len(doc[i].text),
                        "sentiment": self._get_entity_sentiment(doc[start_idx:i+1], doc),
                        "mentions": self._count_entity_mentions(entity_text.lower(), doc)
                    })
            i += 1
        
        return additional_entities
    
    def _guess_entity_type(self, entity_text: str, span) -> str:
        """Make an educated guess about entity type for entities not recognized by spaCy."""
        # Check context for clues
        context = span.sent.text.lower()
        
        # Indicators for different entity types
        person_indicators = ["met", "talked to", "spoke with", "friend", "colleague", "family", 
                            "told", "said", "asked", "called", "texted", "messaged", "emailed"]
        
        place_indicators = ["went to", "visited", "at", "in", "lived in", "stayed at", 
                           "traveled to", "arrived at", "left", "near", "far from"]
        
        org_indicators = ["worked at", "company", "organization", "firm", "corporation", 
                         "agency", "department", "team", "group", "committee", "association"]
        
        # Check for indicators in the context
        if any(indicator in context for indicator in person_indicators):
            return "PERSON"
        elif any(indicator in context for indicator in place_indicators):
            return "LOC"
        elif any(indicator in context for indicator in org_indicators):
            return "ORG"
        
        # If already in user memory, use that type
        if entity_text in self.user_entities and "type" in self.user_entities[entity_text]["metadata"]:
            return self.user_entities[entity_text]["metadata"]["type"]
        
        # Default to miscellaneous
        return "MISC"
    
    def _update_entity_memory(self, entities: List[Dict[str, Any]]):
        """Update user-specific entity memory."""
        if not self.user_id:
            return
            
        current_date = datetime.now().isoformat()
        
        for entity in entities:
            entity_text = entity["text"]
            
            # Update entity count and last mentioned date
            self.user_entities[entity_text]["count"] += 1
            self.user_entities[entity_text]["last_mentioned"] = current_date
            
            # Update or add metadata
            for key, value in entity.get("metadata", {}).items():
                self.user_entities[entity_text]["metadata"][key] = value
            
            # Store entity type in metadata if not already present
            if "type" not in self.user_entities[entity_text]["metadata"]:
                self.user_entities[entity_text]["metadata"]["type"] = entity["type"]
    
    def _extract_events(self, doc) -> List[Dict[str, Any]]:
        """Extract events with enhanced context and understanding across sentence boundaries."""
        events = []
    
        # Identify thematic clusters of sentences
        thematic_clusters = self._identify_thematic_cluster(doc)
    
        for cluster_indices, cluster_text in thematic_clusters:
        # Check if this cluster contains an event
            has_indicator = any(indicator in cluster_text.lower() for indicator in self.event_indicators)
        
        # Check for action verbs
            has_action_verb = False
            main_verb = None
        
        # Look for key verbs in the sentences of this cluster
            for idx in cluster_indices:
                sent = list(doc.sents)[idx]
                for token in sent:
                    if token.pos_ == "VERB" and not token.is_stop:
                        if token.lemma_ in ["go", "meet", "talk", "visit", "work", "start", "finish", 
                                      "attend", "celebrate", "plan", "organize", "participate",
                                      "complete", "create", "make", "develop", "discuss", "learn"]:
                            has_action_verb = True
                            main_verb = token
                            break
                if has_action_verb:
                    break
        
        # Process this cluster as a potential event if it contains indicators or action verbs
            if has_indicator or has_action_verb:
                # Create a unified event from the entire cluster
                cluster_doc = self.nlp(cluster_text)
            
            # Extract event components from the entire cluster text
                components = self._extract_event_components(cluster_doc, main_verb)
            
            # Only include if we found a valid action
                if components["action"]:
                # Detect event type
                    event_type = self._classify_event_type(cluster_text)
                
                # Detect temporal aspects
                    temporal_context = self._detect_event_temporality(cluster_text)
                
                # Process status, importance, and regularity
                    event_status = self._detect_completion(cluster_text)
                    event_importance = self._detect_event_importance(cluster_text)
                    event_regularity = self._detect_event_regularity(cluster_text)
                
                # Enhanced event with more metadata
                    events.append({
                        "text": cluster_text,
                        "components": components,
                        "type": event_type,
                        "completed": event_status == "COMPLETED",
                        "status": event_status,
                        "importance": event_importance,
                        "temporality": temporal_context,
                        "regularity": event_regularity,
                        "sentiment": self._analyze_sentiment(cluster_text),
                        # Add location of the cluster in the document
                        "span": {
                            "start_sent": min(cluster_indices),
                            "end_sent": max(cluster_indices),
                            "sentence_count": len(cluster_indices)
                        }
                    })
    
        return events
    
    def _calculate_sentence_continuity(self, sent, prev_sent, sent_features, prev_features, 
                                    sent_idx, prev_idx, sentence_embeddings):
        """
        Calculate a continuity score between two sentences (higher = more continuous).
        Uses multiple signals to determine if sentences belong to the same event.
        """
        continuity_signals = []
        
        # 1. Semantic similarity (using embeddings)
        semantic_similarity = 0.0
        if sentence_embeddings:
            try:
                sent_emb = sentence_embeddings[sent_idx]
                prev_emb = sentence_embeddings[prev_idx]
                semantic_similarity = np.dot(sent_emb, prev_emb) / (np.linalg.norm(sent_emb) * np.linalg.norm(prev_emb))
                continuity_signals.append(("semantic", semantic_similarity))
            except Exception as e:
                logger.error(f"Error calculating semantic similarity: {e}")
        
        # 2. Entity continuity - shared entities indicate continuity
        sent_entities = [e["text"] for e in sent_features["entities"]]
        prev_entities = [e["text"] for e in prev_features["entities"]]
        
        shared_entities = set(sent_entities).intersection(set(prev_entities))
        entity_continuity = len(shared_entities) / max(1, max(len(sent_entities), len(prev_entities)))
        continuity_signals.append(("entity", entity_continuity))
        
        # 3. Topic continuity - shared topics indicate continuity
        sent_topics = sent_features["topics"]
        prev_topics = prev_features["topics"]
        
        shared_topics = set(sent_topics).intersection(set(prev_topics))
        topic_continuity = len(shared_topics) / max(1, max(len(sent_topics), len(prev_topics)))
        continuity_signals.append(("topic", topic_continuity))
        
        # 4. Subject/speaker continuity - same speaker indicates continuity
        speaker_continuity = 0.0
        if sent_features["speaker"] and prev_features["speaker"]:
            if sent_features["speaker"] == prev_features["speaker"]:
                speaker_continuity = 1.0
        continuity_signals.append(("speaker", speaker_continuity))
        
        # 5. Narrative continuity - coreference, pronouns indicating continuation
        narrative_continuity = 0.0
        sent_text = sent.text.lower()
        
        # Check for pronouns that typically refer back to previous context
        referential_pronouns = ["this", "that", "these", "those", "it", "they", "he", "she"]
        has_referential_pronoun = any(token.text.lower() in referential_pronouns 
                                    for token in sent if token.pos_ == "PRON")
        
        # Check for discourse connectives that continue a narrative
        continuation_markers = ["and", "also", "as well", "too", "additionally", "moreover", 
                            "furthermore", "likewise", "similarly", "in addition"]
        has_continuation_marker = any(marker in sent_text for marker in continuation_markers)
        
        if has_referential_pronoun or has_continuation_marker:
            narrative_continuity = 0.8
        continuity_signals.append(("narrative", narrative_continuity))
        
        # Calculate weighted continuity score
        weights = {
            "semantic": 0.3,
            "entity": 0.25,
            "topic": 0.2,
            "speaker": 0.15,
            "narrative": 0.1
        }
        
        weighted_sum = sum(score * weights[signal_type] for signal_type, score in continuity_signals)
        return weighted_sum
    
    def _calculate_sentence_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two sentences."""
        try:
            # Use the embedding model to calculate similarity
            embedding1 = self.embedding_model.encode(text1)
            embedding2 = self.embedding_model.encode(text2)
            
            return float(util.pytorch_cos_sim(embedding1, embedding2).item())
        except Exception as e:
            logger.error(f"Error calculating sentence similarity: {e}")
            return 0.0
    
    def _have_shared_topic(self, sent1, sent2) -> bool:
        """Check if two sentences share a common topic or theme."""
        # Extract keywords from both sentences
        keywords1 = self._extract_important_keywords(sent1.text)
        keywords2 = self._extract_important_keywords(sent2.text)
        
        # Check for shared keywords
        shared_keywords = set(keywords1) & set(keywords2)
        
        # If we have shared keywords, it's a strong indicator
        if len(shared_keywords) >= 2:
            return True
        
        # Extract topics from both sentences
        topics1 = set(self._detect_topics(sent1.doc))
        topics2 = set(self._detect_topics(sent2.doc))
        
        # Check for shared topics
        return bool(topics1 & topics2)
    
    def _generate_thread_id(self, components: Dict[str, Any]) -> str:
        """Generate a stable thread ID based on key components."""
        # Combine key components to create a stable identifier
        key_parts = []
        
        # Add action
        if components.get("action"):
            key_parts.append(components["action"].lower())
        
        # Add main participants (sorted for stability)
        if components.get("participants"):
            participants = sorted(p.lower() for p in components["participants"])
            key_parts.extend(participants)
        
        # Add location if present
        if components.get("location"):
            key_parts.append(components["location"].lower())
        
        # Create a stable hash of the combined components
        thread_key = "_".join(key_parts)
        return hashlib.md5(thread_key.encode()).hexdigest()
    
    def _extract_thread_metadata(self, sentences: List, components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata specific to thread tracking."""
        metadata = {
            "first_mention_date": None,
            "last_mention_date": None,
            "mention_count": 1,
            "previous_mentions": [],
            "related_threads": [],
            "progression_stage": self._detect_progression_stage(sentences),
            "is_complete": self._is_thread_complete(sentences),
            "participants_history": self._extract_participants_history(sentences),
            "location_changes": self._extract_location_changes(sentences),
            "temporal_progression": self._extract_temporal_progression(sentences)
        }
        
        return metadata
    
    def _detect_progression_stage(self, sentences: List) -> str:
        """Detect the current stage of the thread progression."""
        combined_text = " ".join(s.text for s in sentences)
        
        # Define stage indicators
        planning_indicators = {"plan", "will", "going to", "intend", "prepare"}
        ongoing_indicators = {"currently", "now", "in progress", "working on"}
        completion_indicators = {"finished", "completed", "done", "accomplished"}
        
        text_lower = combined_text.lower()
        
        # Check for completion first (highest priority)
        if any(indicator in text_lower for indicator in completion_indicators):
            return "COMPLETED"
        
        # Check for ongoing status
        if any(indicator in text_lower for indicator in ongoing_indicators):
            return "ONGOING"
        
        # Check for planning stage
        if any(indicator in text_lower for indicator in planning_indicators):
            return "PLANNING"
        
        return "UNKNOWN"
    
    def _is_thread_complete(self, sentences: List) -> bool:
        """Determine if a thread appears to be complete."""
        last_sent = sentences[-1].text.lower()
        
        # Check for completion indicators in the last sentence
        completion_indicators = {
            "finished", "completed", "done", "accomplished", "ended",
            "concluded", "resolved", "achieved", "succeeded"
        }
        
        return any(indicator in last_sent for indicator in completion_indicators)
    
    def _extract_participants_history(self, sentences: List) -> List[Dict[str, Any]]:
        """Track how participants are mentioned across the thread."""
        participant_mentions = []
        
        for sent in sentences:
            participants = set()
            for ent in sent.ents:
                if ent.label_ in ["PERSON", "ORG"]:
                    participants.add(ent.text)
            
            if participants:
                participant_mentions.append({
                    "sentence_index": sent.start,
                    "participants": list(participants)
                })
        
        return participant_mentions
    
    def _extract_location_changes(self, sentences: List) -> List[Dict[str, Any]]:
        """Track location changes throughout the thread."""
        location_changes = []
        
        for sent in sentences:
            locations = set()
            for ent in sent.ents:
                if ent.label_ in ["LOC", "GPE", "FAC"]:
                    locations.add(ent.text)
            
            if locations:
                location_changes.append({
                    "sentence_index": sent.start,
                    "locations": list(locations)
                })
        
        return location_changes
    
    def _extract_temporal_progression(self, sentences: List) -> List[Dict[str, Any]]:
        """Track temporal progression throughout the thread."""
        temporal_markers = []
        
        for sent in sentences:
            temporal_info = self._detect_event_temporality(sent.text)
            if temporal_info.get("time_expressions") or temporal_info.get("relative_time"):
                temporal_markers.append({
                    "sentence_index": sent.start,
                    "temporal_info": temporal_info
                })
        
        return temporal_markers
    
    def _extract_event_components(self, sent, main_verb=None) -> Dict[str, Any]:
        """Extract components of an event from a sentence with improved parsing."""
        components = {
            "action": "",
            "participants": [],
            "location": "",
            "time": [],
            "purpose": "",
            "manner": "",
            "instruments": []
        }
        
        # Handle both SpaCy sentence objects and SpaCy doc objects
        if hasattr(sent, 'doc'):
            # This is a spaCy span (sentence)
            doc = sent.doc
            sent_tokens = [token for token in sent]
        else:
            # This is already a spaCy doc
            doc = sent
            sent_tokens = [token for token in doc]
        
        # If no main verb was passed, find the main verb
        if not main_verb:
            preferred_verbs = ["go", "meet", "talk", "visit", "work", "start", "finish",
                              "attend", "celebrate", "plan", "organize", "participate",
                              "complete", "create", "make", "develop", "discuss", "learn"]
                              
            # First try to find one of our preferred verbs
            for token in sent_tokens:
                if token.pos_ == "VERB" and token.lemma_ in preferred_verbs:
                    main_verb = token
                    break
                    
            # If no preferred verb found, take any non-stop verb
            if not main_verb:
                for token in sent_tokens:
                    if token.pos_ == "VERB" and not token.is_stop:
                        main_verb = token
                        break
        
        if not main_verb:
            # No verb found, try to extract meaningful phrases
            for token in sent_tokens:
                if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                    for child in token.children:
                        if child.dep_ == "compound":
                            components["action"] = f"{child.text} {token.text}"
                            break
                    if components["action"]:
                        break
                        
            if not components["action"]:
                # Just use first few words as action
                components["action"] = " ".join([t.text for t in sent_tokens[:min(3, len(sent_tokens))]])
                
            # Extract potential participants (named entities)
            if hasattr(sent, 'ents'):
                entities = sent.ents
            else:
                entities = doc.ents
                
            for ent in entities:
                if ent.label_ in ["PERSON", "ORG", "GPE"]:
                    components["participants"].append(ent.text)
                elif ent.label_ in ["LOC", "GPE", "FAC"]:
                    components["location"] = ent.text
                elif ent.label_ in ["DATE", "TIME"]:
                    components["time"].append(ent.text)
                    
            return components
        
        # Extract the main action (verb phrase)
        components["action"] = self._get_verb_phrase(main_verb)
        
        # Extract participants (subjects, objects and named entities)
        for token in sent_tokens:
            # Get subjects
            if token.dep_ in ["nsubj", "nsubjpass"] and token.head == main_verb:
                # Extract full noun phrase
                subject = self._get_span_text(token)
                if subject and subject.lower() not in ["i", "me", "myself", "we", "us", "ourselves"]:
                    components["participants"].append(subject)
            
            # Get objects
            elif token.dep_ in ["dobj", "pobj"] and (token.head == main_verb or 
                                                   (token.head.dep_ == "prep" and token.head.head == main_verb)):
                # Extract full noun phrase
                obj = self._get_span_text(token)
                if obj and obj.lower() not in ["it", "that", "this"]:
                    components["participants"].append(obj)
            
            # Get purpose (why the event happened)
            elif token.dep_ == "prep" and token.text.lower() == "to" and token.head == main_verb:
                for child in token.children:
                    if child.dep_ == "pobj":
                        components["purpose"] = self._get_span_text(child)
                        break
            
            # Get manner (how the event happened)
            elif token.dep_ in ["advmod", "acomp"] and token.head == main_verb:
                components["manner"] = token.text
            
            # Get instruments (with what the event happened)
            elif token.dep_ == "prep" and token.text.lower() == "with" and token.head == main_verb:
                for child in token.children:
                    if child.dep_ == "pobj":
                        components["instruments"].append(self._get_span_text(child))
        
        # Extract named entities
        if hasattr(sent, 'ents'):
            entities = sent.ents
        else:
            entities = doc.ents
            
        for ent in entities:
            if ent.label_ == "PERSON" and ent.text not in components["participants"]:
                components["participants"].append(ent.text)
            elif ent.label_ in ["LOC", "GPE", "FAC"] and not components["location"]:
                components["location"] = ent.text
            elif ent.label_ in ["DATE", "TIME"] and ent.text not in components["time"]:
                components["time"].append(ent.text)
        
        # Look for location prepositions
        if not components["location"]:
            for token in sent_tokens:
                if token.dep_ == "prep" and token.text.lower() in ["at", "in", "on"]:
                    for child in token.children:
                        if child.dep_ == "pobj":
                            loc = self._get_span_text(child)
                            if loc:
                                components["location"] = loc
                                break
        
        return components
    
    def _get_verb_phrase(self, verb_token):
        """Extract the full verb phrase with improved handling of phrasal verbs."""
        if not verb_token:
            return ""
            
        # Get all parts of the verb phrase
        verb_tokens = [verb_token]
        
        # Get auxiliary verbs
        for token in verb_token.lefts:
            if token.dep_ == "aux" or token.dep_ == "auxpass":
                verb_tokens.append(token)
        
        # Get particles (for phrasal verbs)
        for token in verb_token.rights:
            if token.dep_ == "prt":
                verb_tokens.append(token)
                
        # Sort tokens by their position in the sentence
        verb_tokens.sort(key=lambda x: x.i)
        
        # Join the tokens to form the verb phrase
        verb_phrase = " ".join([token.text for token in verb_tokens])
        
        return verb_phrase
    
    def _get_span_text(self, token):
        """Get the full text span for a token and its children with improved handling."""
        # Handle edge cases
        if not token:
            return ""
            
        # For pronouns, just return the token
        if token.pos_ == "PRON":
            return token.text
            
        # Get all tokens in the subtree
        subtree_tokens = list(token.subtree)
        
        # Sort by position in the text
        subtree_tokens.sort(key=lambda x: x.i)
        
        # Filter out certain tokens (like punctuation at the end)
        filtered_tokens = [t for t in subtree_tokens if not (t.is_punct and t.i == subtree_tokens[-1].i)]
        
        if not filtered_tokens:
            return token.text
            
        # Get the text span
        start = filtered_tokens[0].i
        end = filtered_tokens[-1].i + 1
        
        return token.doc[start:end].text
    
    def _classify_event_type(self, text: str) -> str:
        """
        Classify event type with advanced categorization.
        Uses both rule-based and semantic indicators for better accuracy.
        """
        # Convert to lowercase for matching
        text_lower = text.lower()
        
        # Calculate scores for each category using our categorized indicators
        scores = {}
        
        for category, terms in self.categorized_indicators.items():
            # Count occurrences of category terms
            score = sum(1 for term in terms if term in text_lower)
            
            # Add extra weight for specific strong indicators
            if score > 0:
                strong_indicators = terms[:5]  # First items tend to be stronger indicators
                if any(indicator in text_lower for indicator in strong_indicators):
                    score += 2
                    
            scores[category] = score
        
        # Return highest scoring category, or OTHER if no clear match
        max_score = max(scores.values()) if scores else 0
        
        if max_score > 0:
            # Check if we have a clear winner
            max_categories = [category for category, score in scores.items() if score == max_score]
            if len(max_categories) == 1:
                return max_categories[0]
            else:
                # If tie, use semantic similarity with category descriptions
                if self.embedding_model:
                    try:
                        text_embedding = self.embedding_model.encode(text_lower)
                        category_similarities = {}
                        
                        for category in max_categories:
                            # Create a description for the category
                            category_desc = f"{category.lower()}: " + ", ".join(self.categorized_indicators[category][:10])
                            category_embedding = self.embedding_model.encode(category_desc)
                            
                            # Calculate similarity
                            similarity = util.pytorch_cos_sim(text_embedding, category_embedding).item()
                            category_similarities[category] = similarity
                        
                        # Return the category with highest semantic similarity
                        return max(category_similarities.items(), key=lambda x: x[1])[0]
                    except Exception as e:
                        logger.error(f"Error in semantic category matching: {e}")
                
                # Fallback to first category if semantic matching fails
                return max_categories[0]
        
        return "OTHER"
    def _identify_thematic_cluster(self, doc):
        """
        Identify thematic clusters across multiple sentences.
        Returns groups of sentences that discuss the same theme/topic/event.
        Enhanced to accurately detect distinct event/narrative threads in general text.
        """
        sentences = list(doc.sents)
        clusters = []
        current_cluster = []
        
        # Skip processing if there are no sentences
        if not sentences:
            return []

        # Create sentence embeddings for semantic similarity comparison
        sentence_embeddings = None
        if self.embedding_model:
            try:
                sentence_embeddings = [self.embedding_model.encode(sent.text) for sent in sentences]
            except Exception as e:
                logger.error(f"Error creating sentence embeddings: {e}")
        
        # First pass - group sentences into initial clusters
        for i, sent in enumerate(sentences):
            # Extract key features from this sentence
            sent_features = self._extract_sentence_features(sent)
            
            if not current_cluster:
                # First sentence always starts a cluster
                current_cluster.append(i)
                continue
            
            # Check if this sentence belongs to the current cluster or starts a new one
            prev_idx = current_cluster[-1]
            prev_sent = sentences[prev_idx]
            prev_features = self._extract_sentence_features(prev_sent)
            
            # Calculate continuity score between sentences
            continuity_score = self._calculate_sentence_continuity(
                sent, prev_sent, 
                sent_features, prev_features,
                i, prev_idx, 
                sentence_embeddings
            )
            
            # Detect event boundaries based on multiple signals
            is_event_boundary = self._detect_event_boundary(
                sent, prev_sent, 
                sent_features, prev_features, 
                continuity_score
            )
            
            if is_event_boundary:
                # This sentence starts a new event/thread
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = [i]
            else:
                # This sentence continues the current cluster
                current_cluster.append(i)
        
        # Add the last cluster if it exists
        if current_cluster:
            clusters.append(current_cluster)
        
        # Second pass - merge related clusters that might have been split
        i = 0
        while i < len(clusters) - 1:
            cluster1 = clusters[i]
            cluster2 = clusters[i + 1]
            
            # Calculate cross-cluster coherence
            coherence = self._calculate_cluster_coherence(cluster1, cluster2, sentences, sentence_embeddings)
            
            if coherence > 0.65:  # High coherence threshold for merging
                # Merge clusters
                clusters[i] = cluster1 + cluster2
                clusters.pop(i + 1)
            else:
                i += 1
        
        # Return sentence clusters with their text
        return [(cluster, " ".join([sentences[idx].text for idx in cluster])) for cluster in clusters]

    
    def _extract_sentence_features(self, sent):
        """
        Extract key features from a sentence for event boundary detection.
        """
        features = {
            "entities": [],
            "topics": [],
            "temporal_markers": [],
            "location_markers": [],
            "action_verbs": [],
            "speaker": None,
            "has_transition_marker": False
        }
        
        # Extract entities
        for ent in sent.ents:
            features["entities"].append({
                "text": ent.text.lower(),
                "type": ent.label_
            })
            
            # Track locations separately
            if ent.label_ in ["LOC", "GPE", "FAC"]:
                features["location_markers"].append(ent.text.lower())
        
        # Extract main verbs (actions)
        for token in sent:
            if token.pos_ == "VERB" and not token.is_stop:
                features["action_verbs"].append(token.lemma_.lower())
        
        # Extract temporal markers
        sent_text = sent.text.lower()
        temporal_markers = [
            "morning", "afternoon", "evening", "night", "today", "yesterday", 
            "tomorrow", "later", "before", "after", "during", "while",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
        ]
        
        for marker in temporal_markers:
            if marker in sent_text:
                features["temporal_markers"].append(marker)
        
        # Check for transition markers that often indicate new events
        transition_markers = [
            "then", "next", "after that", "afterwards", "meanwhile", "in the meantime",
            "later on", "subsequently", "following that", "at that point",
            "first", "second", "third", "finally", "lastly"
        ]
        
        for marker in transition_markers:
            if marker in sent_text:
                features["has_transition_marker"] = True
                break
        
        # Detect the speaker/subject focus (e.g., "I", "we", "they")
        for token in sent:
            if token.dep_ == "nsubj" and token.pos_ == "PRON":
                features["speaker"] = token.text.lower()
                break
        
        # Extract basic topics (simple approach)
        for token in sent:
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                features["topics"].append(token.lemma_.lower())
        
        return features
    
    
    def _calculate_cluster_coherence(self, cluster1, cluster2, sentences, sentence_embeddings):
        """
        Calculate coherence between two clusters to determine if they should be merged.
        """
        # Compare the last sentence of cluster1 with the first sentence of cluster2
        last_idx = cluster1[-1]
        first_idx = cluster2[0]
        
        last_sent = sentences[last_idx]
        first_sent = sentences[first_idx]
        
        # Extract features
        last_features = self._extract_sentence_features(last_sent)
        first_features = self._extract_sentence_features(first_sent)
        
        # Calculate continuity score
        continuity = self._calculate_sentence_continuity(
            first_sent, last_sent,
            first_features, last_features,
            first_idx, last_idx,
            sentence_embeddings
        )
        
        # Check for narrative continuity between the clusters
        narrative_markers = ["consequently", "therefore", "as a result", "for this reason",
                            "because of this", "this led to", "this resulted in"]
        
        first_text = first_sent.text.lower()
        has_narrative_marker = any(marker in first_text for marker in narrative_markers)
        
        if has_narrative_marker:
            continuity += 0.2
        
        return min(continuity, 1.0)  # Cap at 1.0
    def _detect_event_boundary(self, sent, prev_sent, sent_features, prev_features, continuity_score):
        """
        Determine if a sentence marks the boundary of a new event/thread.
        Returns True if this likely starts a new event, False if it continues the previous event.
        """
        # Strong signals that this is a new event
        sent_text = sent.text.lower()
        
        # 1. Explicit time shift indicates a new event
        time_shift = False
        if sent_features["temporal_markers"] and not prev_features["temporal_markers"]:
            time_shift = True
        elif (sent_features["temporal_markers"] and prev_features["temporal_markers"] and 
            not set(sent_features["temporal_markers"]).intersection(set(prev_features["temporal_markers"]))):
            time_shift = True
        
        # 2. Explicit location shift indicates a new event
        location_shift = False
        if sent_features["location_markers"] and prev_features["location_markers"]:
            # Different locations mentioned
            if not set(sent_features["location_markers"]).intersection(set(prev_features["location_markers"])):
                location_shift = True
        
        # 3. Strong transition markers indicate a new event
        has_strong_transition = sent_features["has_transition_marker"]
        
        # 4. Activity change indicates a new event
        activity_keywords = {
            "work": ["work", "office", "job", "meeting", "client", "project", "boss", "colleague"],
            "leisure": ["relax", "watch", "play", "enjoy", "entertainment", "movie", "game", "fun"],
            "exercise": ["run", "walk", "gym", "workout", "exercise", "train", "fitness", "sport"],
            "social": ["friend", "talk", "chat", "conversation", "party", "gathering", "meet"],
            "food": ["eat", "food", "meal", "breakfast", "lunch", "dinner", "restaurant", "cook"],
            "creative": ["paint", "draw", "write", "create", "art", "craft", "creative", "design"],
            "travel": ["drive", "travel", "trip", "journey", "visit", "explore", "tour"]
        }
        
        current_activities = set()
        previous_activities = set()
        
        for activity, keywords in activity_keywords.items():
            if any(keyword in sent_text for keyword in keywords):
                current_activities.add(activity)
            if any(keyword in prev_sent.text.lower() for keyword in keywords):
                previous_activities.add(activity)
        
        activity_shift = (len(current_activities) > 0 and len(previous_activities) > 0 and 
                        not current_activities.intersection(previous_activities))
        
        # 5. Significant change in the action verbs
        action_shift = False
        if (sent_features["action_verbs"] and prev_features["action_verbs"] and 
        not set(sent_features["action_verbs"]).intersection(set(prev_features["action_verbs"]))):
            action_shift = True
        
        # Weigh all signals to make the final decision
        new_event_signals = [
            time_shift, 
            location_shift, 
            has_strong_transition,
            activity_shift,
            action_shift
        ]
        
        # Count the number of signals indicating a new event
        signal_count = sum(1 for signal in new_event_signals if signal)
        
        # Decision thresholds
        if signal_count >= 2:  # Multiple strong signals
            return True
        elif signal_count == 1 and continuity_score < 0.4:  # One strong signal + low continuity
            return True
        elif continuity_score < 0.25:  # Very low continuity alone can indicate a new event
            return True
        
        return False
    
    def _detect_event_temporality(self, text: str) -> Dict[str, Any]:
        """
        Detect when an event occurred, its duration, and other temporal aspects.
        """
        text_lower = text.lower()
        
        # Initialize temporal information
        temporal_info = {
            "tense": self._detect_tense(text),
            "time_expressions": [],
            "relative_time": None,
            "duration": None
        }
        
        # Extract date patterns
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text)
            temporal_info["time_expressions"].extend(matches)
        
        # Extract time patterns
        for pattern in self.time_patterns:
            matches = re.findall(pattern, text)
            temporal_info["time_expressions"].extend(matches)
        
        # Extract duration
        for pattern in self.duration_patterns:
            matches = re.findall(pattern, text)
            if matches:
                temporal_info["duration"] = matches[0]
        
        # Determine relative time
        for timeframe, indicators in self.temporal_references.items():
            if any(indicator in text_lower for indicator in indicators):
                temporal_info["relative_time"] = timeframe
                break
        
        return temporal_info
    
    def _detect_tense(self, text: str) -> str:
        """Detect the primary tense used in the text."""
        # Process with spaCy to get verb information
        doc = self.nlp(text)
        
        tense_indicators = {
            "PAST": 0,
            "PRESENT": 0,
            "FUTURE": 0
        }
        
        # Past tense indicators
        past_auxiliaries = ["had", "did", "was", "were"]
        past_endings = ["ed", "d", "t"]  # Common past tense endings
        
        # Present tense indicators
        present_auxiliaries = ["am", "is", "are", "do", "does"]
        
        # Future tense indicators
        future_auxiliaries = ["will", "shall", "going to", "about to"]
        
        # Analyze each verb
        for token in doc:
            if token.pos_ == "VERB":
                # Check for past tense
                if token.tag_ in ["VBD", "VBN"]:
                    tense_indicators["PAST"] += 1
                # Check for present tense
                elif token.tag_ in ["VBP", "VBZ", "VBG"]:
                    tense_indicators["PRESENT"] += 1
                # For other verbs, check the text
                else:
                    text_lower = token.text.lower()
                    
                    # Check endings for past tense
                    if any(text_lower.endswith(ending) for ending in past_endings):
                        tense_indicators["PAST"] += 1
            
            # Check for auxiliaries that indicate tense
            elif token.pos_ == "AUX":
                text_lower = token.text.lower()
                
                if text_lower in past_auxiliaries:
                    tense_indicators["PAST"] += 1
                elif text_lower in present_auxiliaries:
                    tense_indicators["PRESENT"] += 1
                elif text_lower in future_auxiliaries or "will" in text_lower or "going to" in text_lower:
                    tense_indicators["FUTURE"] += 1
        
        # Check for future indicators in the whole text
        text_lower = text.lower()
        if "will" in text_lower or "going to" in text_lower or "plan to" in text_lower:
            tense_indicators["FUTURE"] += 2
        
        # Determine predominant tense
        max_tense = max(tense_indicators.items(), key=lambda x: x[1])
        
        if max_tense[1] > 0:
            return max_tense[0]
        
        # Default to PRESENT if no clear indicators
        return "PRESENT"
    
    def _detect_event_importance(self, text: str) -> str:
        """Detect the relative importance or significance of an event."""
        text_lower = text.lower()
        
        # High importance indicators
        high_importance = [
            "important", "significant", "crucial", "critical", "vital", "essential", 
            "major", "key", "milestone", "big", "huge", "massive", "enormous",
            "extremely", "very", "really", "definitely", "absolutely", 
            "can't believe", "amazing", "incredible", "extraordinary", "exceptional",
            "life-changing", "transformative", "never forget", "highlight",
            "best", "worst", "most", "greatest", "finally"
        ]
        
        # Medium importance indicators
        medium_importance = [
            "good", "bad", "interesting", "nice", "fun", "challenging", "tough",
            "enjoyed", "happy", "unhappy", "pleased", "disappointed", "exciting",
            "significant", "memorable", "noticeable", "impactful", "meaningful",
            "better than expected", "worse than expected", "unusual", "special"
        ]
        
        # Low importance indicators
        low_importance = [
            "routine", "usual", "regular", "common", "ordinary", "typical", "normal",
            "everyday", "standard", "average", "moderate", "mild", "minor", "small",
            "little", "slight", "somewhat", "kind of", "sort of", "just", "only"
        ]
        
        # Count indicators of each importance level
        high_count = sum(1 for term in high_importance if term in text_lower)
        medium_count = sum(1 for term in medium_importance if term in text_lower)
        low_count = sum(1 for term in low_importance if term in text_lower)
        
        # Apply weights
        score = (high_count * 3) - (low_count * 2) + (medium_count * 1)
        
        if score >= 3:
            return "HIGH"
        elif score <= -2:
            return "LOW"
        else:
            return "MEDIUM"
    
    def _detect_event_regularity(self, text: str) -> str:
        """Detect whether an event is regular/recurring or a one-time occurrence."""
        text_lower = text.lower()
        
        # Regular event indicators
        regular_indicators = [
            "every", "each", "daily", "weekly", "monthly", "yearly", "annually",
            "regularly", "routine", "always", "usually", "typically", "normally",
            "again", "another", "once again", "as usual", "like always",
            "same as", "like last time", "recurring", "repeated", "repetitive",
            "habitually", "customarily", "traditionally", "consistently"
        ]
        
        # One-time event indicators
        onetime_indicators = [
            "first time", "once", "one-time", "unique", "special", "singular",
            "exceptional", "unusual", "rare", "never before", "never again",
            "once in a lifetime", "once-in-a-lifetime", "unprecedented",
            "extraordinary", "exceptional", "surprising", "unexpected"
        ]
        
        # Check frequency patterns
        for pattern in self.frequency_patterns:
            if re.search(pattern, text_lower):
                return "RECURRING"
        
        # Count occurrences of indicators
        regular_count = sum(1 for indicator in regular_indicators if indicator in text_lower)
        onetime_count = sum(1 for indicator in onetime_indicators if indicator in text_lower)
        
        if regular_count > onetime_count:
            return "RECURRING"
        elif onetime_count > regular_count:
            return "ONE_TIME"
        
        # Default to ONE_TIME if no clear indicators
        return "ONE_TIME"
    
    def _detect_completion(self, text: str) -> str:
        """
        Detect the completion status of an event with finer granularity.
        Returns: "COMPLETED", "IN_PROGRESS", "PLANNED", "CANCELLED", or "UNCERTAIN"
        """
        # Convert to lowercase for matching
        text_lower = text.lower()
        
        # Define completion indicators with enhanced detection
        completion_indicators = {
            "COMPLETED": [
                "finished", "completed", "ended", "done", "wrapped up",
                "concluded", "over", "accomplished", "achieved", "finalized",
                "closed", "delivered", "submitted", "turned in", "handed in",
                "published", "released", "launched", "shipped", "deployed",
                "graduated", "won", "succeeded", "passed", "completed",
                "received", "got back", "heard back", "returned", "arrived"
            ],
            
            "IN_PROGRESS": [
                "working on", "in progress", "ongoing", "continuing", "still",
                "in the middle of", "in the midst of", "developing", "evolving",
                "advancing", "proceeding", "underway", "in motion", "in action",
                "active", "live", "current", "present", "happening now", 
                "unfinished", "incomplete", "halfway", "midway", "partly"
            ],
            
            "PLANNED": [
                "will", "going to", "planning to", "intend to", "aim to",
                "expect to", "hope to", "scheduled", "upcoming", "future",
                "soon", "next", "tomorrow", "later", "eventually",
                "preparation", "preparing", "getting ready", "setting up",
                "organizing", "arranging", "booking", "scheduling"
            ],
            
            "CANCELLED": [
                "cancelled", "canceled", "abandoned", "dropped", "gave up",
                "stopped", "halted", "terminated", "ended", "aborted",
                "scrapped", "scratched", "called off", "abolished", "dismissed",
                "rejected", "refused", "declined", "failed to", "couldn't",
                "not able to", "didn't happen", "fall through", "fall apart"
            ]
        }
        
        # Check tense
        tense = self._detect_tense(text)
        
        # Check for explicit status indicators
        for status, indicators in completion_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return status
                
        # Tense-based inference (if no explicit indicators)
        if tense == "PAST":
            # Past tense suggests completion unless there are ongoing indicators
            ongoing_indicators = ["still", "continuing", "ongoing", "in progress", "working on"]
            if not any(indicator in text_lower for indicator in ongoing_indicators):
                return "COMPLETED"
        elif tense == "FUTURE":
            return "PLANNED"
        elif tense == "PRESENT":
            # Present tense often indicates in-progress
            return "IN_PROGRESS"
                
        # Default to uncertain if we can't determine
        return "UNCERTAIN"
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using transformer model if available, 
        fallback to enhanced lexicon-based approach.
        """
        if not text:
            return {"label": "NEUTRAL", "score": 0.5}
            
        if self.sentiment_analyzer:
            try:
                # Use transformer model - truncate to avoid token limit issues
                max_length = 512
                text_to_analyze = text[:max_length]
                
                result = self.sentiment_analyzer(text_to_analyze)
                label = result[0]['label']
                score = result[0]['score']
                
                # Convert to standard format
                if label == "POSITIVE":
                    return {"label": "POSITIVE", "score": score}
                elif label == "NEGATIVE":
                    return {"label": "NEGATIVE", "score": 1.0 - score}  # Invert score for negative
                else:
                    return {"label": "NEUTRAL", "score": 0.5}
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {e}")
                # Fall back to lexicon-based approach
        
        # Enhanced lexicon-based sentiment analysis
        positive_words = ["happy", "glad", "excited", "amazing", "great", "good", "wonderful", 
                         "enjoyed", "love", "fantastic", "excellent", "pleased", "delighted", 
                         "thrilled", "joy", "success", "successful", "achievement", "proud",
                         "impressive", "positive", "beneficial", "favorable", "satisfying", 
                         "rewarding", "gratifying", "uplifting", "encouraging", "hopeful"]
        
        negative_words = ["sad", "angry", "upset", "terrible", "bad", "awful", "disappointed", 
                         "hate", "dislike", "frustrating", "annoying", "horrible", "miserable", 
                         "depressed", "anxious", "worried", "stressed", "fear", "failed", "failure",
                         "regret", "sorry", "unfortunate", "negative", "detrimental", "adverse",
                         "unfavorable", "unsatisfactory", "discouraging", "depressing", "distressing"]
        
        text_lower = text.lower()
        doc = self.nlp(text_lower)
        
        # Count positive and negative words with negation handling
        pos_count = 0
        neg_count = 0
        
        # Look for negation before sentiment-bearing words
        for i, token in enumerate(doc):
            # Check if token is a sentiment word
            if token.text in positive_words:
                # Check for preceding negation
                negated = False
                for j in range(max(0, i-3), i):
                    if doc[j].text in self.negations:
                        negated = True
                        break
                
                if negated:
                    neg_count += 1
                else:
                    pos_count += 1
                    
            elif token.text in negative_words:
                # Check for preceding negation
                negated = False
                for j in range(max(0, i-3), i):
                    if doc[j].text in self.negations:
                        negated = True
                        break
                
                if negated:
                    pos_count += 1
                else:
                    neg_count += 1
        
        # Calculate sentiment score
        total = pos_count + neg_count
        if total == 0:
            return {"label": "NEUTRAL", "score": 0.5}
        
        pos_ratio = pos_count / total
        
        if pos_ratio > 0.6:
            return {"label": "POSITIVE", "score": min(0.5 + pos_ratio * 0.5, 1.0)}
        elif pos_ratio < 0.4:
            return {"label": "NEGATIVE", "score": max(0.5 - (1 - pos_ratio) * 0.5, 0.0)}
        else:
            return {"label": "NEUTRAL", "score": 0.5}
    
    def _analyze_emotions(self, text: str, doc) -> Dict[str, Any]:
        """
        Perform detailed emotion analysis to detect specific emotions,
        their intensity, and targets.
        """
        # Initialize emotion data
        emotion_data = {
            "primary_emotion": None,
            "secondary_emotion": None,
            "intensity": 0.0,
            "emotions_detected": [],
            "emotional_phrases": [],
            "targets": {}  # Maps emotions to their targets
        }
        
        # Use the dedicated emotion analyzer if available
        if self.emotion_analyzer:
            try:
                # Truncate text to avoid token limits
                max_length = 512
                text_to_analyze = text[:max_length]
                
                # Get emotions from model
                result = self.emotion_analyzer(text_to_analyze)
                
                # Extract the detected emotion label and score
                emotion_label = result[0]['label']
                emotion_score = result[0]['score']
                
                # Map to standard emotion categories
                emotion_mapping = {
                    "sadness": "SADNESS",
                    "joy": "JOY",
                    "love": "JOY",
                    "anger": "ANGER",
                    "fear": "FEAR",
                    "surprise": "SURPRISE",
                    "disgust": "DISGUST"
                }
                
                detected_emotion = emotion_mapping.get(emotion_label, emotion_label.upper())
                
                # Add to detected emotions
                emotion_data["emotions_detected"].append({
                    "emotion": detected_emotion,
                    "confidence": emotion_score
                })
                
                # Set primary emotion
                emotion_data["primary_emotion"] = detected_emotion
                emotion_data["intensity"] = emotion_score
                
            except Exception as e:
                logger.error(f"Error in emotion analysis with model: {e}")
        
        # Use lexicon-based approach (either as fallback or to supplement the model)
        detected_emotions = {}
        
        # Split text into sentences for more granular analysis
        sentences = [sent.text for sent in doc.sents]
        
        for sentence in sentences:
            sent_doc = self.nlp(sentence)
            
            # Look for explicit emotion words
            for token in sent_doc:
                token_text = token.lemma_.lower()
                
                # Check each emotion category
                for emotion, terms in self.emotion_lexicon.items():
                    if token_text in terms:
                        # Get base intensity for this emotion term
                        intensity = terms[token_text]
                        
                        # Check for intensifiers
                        for i in range(max(0, token.i - 3), token.i):
                            if sent_doc[i].text.lower() in self.intensifiers:
                                intensity = min(1.0, intensity * 1.3)  # Boost intensity
                                break
                        
                        # Check for negations
                        negated = False
                        for i in range(max(0, token.i - 3), token.i):
                            if sent_doc[i].text.lower() in self.negations:
                                negated = True
                                break
                        
                        # If negated, flip to opposite emotion
                        if negated:
                            # Simple emotion flipping (could be more sophisticated)
                            opposite_emotions = {
                                "joy": "sadness",
                                "sadness": "joy",
                                "anger": "calm",
                                "fear": "confidence",
                                "disgust": "appreciation",
                                "surprise": "expectation",
                                "trust": "distrust",
                                "anticipation": "surprise"
                            }
                            
                            if emotion in opposite_emotions:
                                emotion = opposite_emotions[emotion]
                        
                        # Add or update emotion
                        if emotion in detected_emotions:
                            # Take highest intensity
                            detected_emotions[emotion] = max(detected_emotions[emotion], intensity)
                        else:
                            detected_emotions[emotion] = intensity
                        
                        # Find potential targets of the emotion
                        targets = self._find_emotion_targets(token, sent_doc)
                        
                        if targets:
                            if emotion not in emotion_data["targets"]:
                                emotion_data["targets"][emotion] = []
                            emotion_data["targets"][emotion].extend(targets)
                        
                        # Add emotional phrase
                        span_start = max(0, token.i - 2)
                        span_end = min(len(sent_doc), token.i + 3)
                        emotional_phrase = sent_doc[span_start:span_end].text
                        emotion_data["emotional_phrases"].append({
                            "text": emotional_phrase,
                            "emotion": emotion,
                            "intensity": intensity
                        })
        
        # Add detected emotions from lexicon (if not already set by model)
        for emotion, intensity in detected_emotions.items():
            emotion_data["emotions_detected"].append({
                "emotion": emotion.upper(),
                "confidence": intensity
            })
        
        # Sort emotions by intensity
        emotion_data["emotions_detected"].sort(key=lambda x: x["confidence"], reverse=True)
        
        # Set primary and secondary emotions if not already set
        if not emotion_data["primary_emotion"] and emotion_data["emotions_detected"]:
            emotion_data["primary_emotion"] = emotion_data["emotions_detected"][0]["emotion"]
            emotion_data["intensity"] = emotion_data["emotions_detected"][0]["confidence"]
            
            if len(emotion_data["emotions_detected"]) > 1:
                emotion_data["secondary_emotion"] = emotion_data["emotions_detected"][1]["emotion"]
        
        # Deduplicate targets
        for emotion in emotion_data["targets"]:
            emotion_data["targets"][emotion] = list(set(emotion_data["targets"][emotion]))
        
        # Update emotion history if tracking a user
        if self.user_id and emotion_data["primary_emotion"]:
            self.emotion_history.append({
                "date": datetime.now().isoformat(),
                "emotion": emotion_data["primary_emotion"],
                "intensity": emotion_data["intensity"]
            })
        
        return emotion_data
    
    def _find_emotion_targets(self, emotion_token, doc) -> List[str]:
        """Find the potential targets of an emotion in text."""
        targets = []
        
        # Case 1: Direct object of emotion verb
        if emotion_token.pos_ == "VERB":
            for token in doc:
                if token.head == emotion_token and token.dep_ == "dobj":
                    targets.append(self._get_span_text(token))
        
        # Case 2: Subject of copula + emotion
        elif emotion_token.pos_ == "ADJ":
            for token in doc:
                if token.head == emotion_token and token.dep_ in ["nsubj", "nsubjpass"]:
                    # Skip first person pronouns
                    if token.text.lower() not in ["i", "we", "me", "us", "myself", "ourselves"]:
                        targets.append(self._get_span_text(token))
        
        # Case 3: Emotion "about" or "at" something
        for token in doc:
            if token.head == emotion_token and token.dep_ == "prep" and token.text.lower() in ["about", "at", "with", "toward", "towards"]:
                for child in token.children:
                    if child.dep_ == "pobj":
                        targets.append(self._get_span_text(child))
        
        # Case 4: Named entities in the same sentence
        if not targets:
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT"]:
                    # Skip if the entity is the subject of the emotion
                    if ent[0].dep_ not in ["nsubj", "nsubjpass"]:
                        targets.append(ent.text)
        
        return targets
    
    def _extract_temporal_information(self, text: str, doc) -> Dict[str, Any]:
        """
        Extract and normalize temporal information from the text.
        Identifies dates, times, durations, and temporal relationships.
        """
        temporal_info = {
            "dates": [],
            "times": [],
            "durations": [],
            "temporal_expressions": [],
            "temporal_ordering": [],
            "tense": self._detect_tense(text)
        }
        
        # Extract named entities related to time
        for ent in doc.ents:
            if ent.label_ == "DATE":
                temporal_info["dates"].append({
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
            elif ent.label_ == "TIME":
                temporal_info["times"].append({
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        
        # Extract temporal expressions using regex patterns
        # Date patterns
        for pattern in self.date_patterns:
            for match in re.finditer(pattern, text):
                temporal_info["temporal_expressions"].append({
                    "type": "DATE",
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Time patterns
        for pattern in self.time_patterns:
            for match in re.finditer(pattern, text):
                temporal_info["temporal_expressions"].append({
                    "type": "TIME",
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Duration patterns
        for pattern in self.duration_patterns:
            for match in re.finditer(pattern, text):
                temporal_info["durations"].append({
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Extract temporal ordering relationships
        temporal_ordering_markers = [
            ("before", "after"), 
            ("earlier", "later"), 
            ("first", "then", "finally"),
            ("initially", "subsequently", "eventually"),
            ("start", "middle", "end"),
            ("previous", "current", "next")
        ]
        
        # Check for temporal sequences in the text
        for sentence in doc.sents:
            sent_text = sentence.text.lower()
            
            for marker_set in temporal_ordering_markers:
                # Check if multiple markers from the same set appear
                markers_found = [marker for marker in marker_set if marker in sent_text]
                
                if len(markers_found) > 1:
                    # Record the temporal ordering relationship
                    temporal_info["temporal_ordering"].append({
                        "markers": markers_found,
                        "sentence": sentence.text
                    })
        
        # Detect relative temporal references
        for timeframe, indicators in self.temporal_references.items():
            found_indicators = [indicator for indicator in indicators if indicator in text.lower()]
            
            if found_indicators:
                temporal_info["temporal_expressions"].append({
                    "type": f"RELATIVE_{timeframe}",
                    "indicators": found_indicators
                })
        
        return temporal_info
    
    def _extract_narrative_elements(self, text: str, doc) -> Dict[str, Any]:
        """
        Extract narrative elements like goals, conflicts, learnings, and narrative arc.
        Identifies the storytelling components in the journal entry.
        """
        narrative_elements = {
            "goals": [],
            "conflicts": [],
            "learnings": [],
            "reflections": [],
            "narrative_arc": None,
            "emotional_arc": [],
            "narrative_roles": {}
        }
    
    # Process each sentence
        for sent in doc.sents:
            sent_text = sent.text.lower()
        
        # Check for narrative elements
            for element_type, indicators in self.narrative_elements.items():
                element_type_lower = element_type.lower()
            
            # Make sure the lowercased key exists in narrative_elements
            # Fix for the KeyError: 'goal' issue
                if element_type_lower not in narrative_elements:
                # Map singular key to plural if needed (e.g., 'GOAL' to 'goals')
                    if element_type_lower + 's' in narrative_elements:
                        element_type_lower = element_type_lower + 's'
                    else:
                    # Skip if we can't find a matching key
                        continue
                    
                for indicator in indicators:
                    if indicator in sent_text:
                    # Extract the relevant phrase or clause
                        narrative_elements[element_type_lower].append({
                            "text": sent.text,
                            "indicator": indicator
                        })
                        break
        
        # Check for narrative arc markers
            for arc_stage, markers in self.narrative_arcs.items():
                if any(marker in sent_text for marker in markers):
                    narrative_elements["narrative_arc"] = arc_stage
                    break
        
        # Extract narrative roles (characters)
            for role, indicators in self.narrative_roles.items():
                if any(indicator in sent_text for indicator in indicators):
                # Find entities that might fulfill this role
                    entities = [ent.text for ent in sent.ents if ent.label_ == "PERSON"]
                
                    if entities:
                        if role not in narrative_elements["narrative_roles"]:
                            narrative_elements["narrative_roles"][role] = []
                        narrative_elements["narrative_roles"][role].extend(entities)
    
    # Detect emotional arc
        current_emotion = None
        for sent in doc.sents:
        # Analyze emotion for this sentence
            sent_emotion = self._analyze_sentiment(sent.text)
        
        # If different from current emotion, we have an emotional shift
            if current_emotion and sent_emotion["label"] != current_emotion["label"]:
                narrative_elements["emotional_arc"].append({
                    "from": current_emotion["label"],
                    "to": sent_emotion["label"],
                    "text": sent.text
                })
        
            current_emotion = sent_emotion
    
    # Detect causal relationships
        for sent in doc.sents:
            sent_text = sent.text.lower()
        
        # Check for cause indicators
            for indicator in self.causal_signals["CAUSE"]:
                if indicator in sent_text:
                # Find the cause and effect parts
                    parts = re.split(r'\b' + re.escape(indicator) + r'\b', sent_text, 1)
                
                    if len(parts) == 2:
                        if "causal_relationships" not in narrative_elements:
                            narrative_elements["causal_relationships"] = []
                        
                        narrative_elements["causal_relationships"].append({
                            "type": "CAUSE",
                            "indicator": indicator,
                            "parts": [p.strip() for p in parts],
                            "text": sent.text
                        })
        
        # Check for effect indicators
            for indicator in self.causal_signals["EFFECT"]:
                if indicator in sent_text:
                # Find the cause and effect parts
                    parts = re.split(r'\b' + re.escape(indicator) + r'\b', sent_text, 1)
                
                    if len(parts) == 2:
                        if "causal_relationships" not in narrative_elements:
                            narrative_elements["causal_relationships"] = []
                        
                        narrative_elements["causal_relationships"].append({
                            "type": "EFFECT",
                            "indicator": indicator,
                            "parts": [p.strip() for p in parts],
                            "text": sent.text
                        })
    
    # Deduplicate and clean up
        for key in ["goals", "conflicts", "learnings", "reflections"]:
            unique_items = {}
            for item in narrative_elements[key]:
                if item["text"] not in unique_items:
                    unique_items[item["text"]] = item
        
            narrative_elements[key] = list(unique_items.values())
    
        for role in narrative_elements["narrative_roles"]:
            narrative_elements["narrative_roles"][role] = list(set(narrative_elements["narrative_roles"][role]))
    
        return narrative_elements
    
    def _extract_relationships(self, doc, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities mentioned in the text.
        Identifies how people, places, and concepts are connected.
        """
        relationships = []
        text = doc.text
        entity_texts = [entity["text"] for entity in entities]
        
        # Find co-occurring entities
        for sent in doc.sents:
            # Get entities in this sentence
            sent_entities = [ent.text for ent in sent.ents if ent.text in entity_texts]
            
            # If we have multiple entities in the sentence, they might be related
            if len(sent_entities) >= 2:
                for pair in combinations(sent_entities, 2):
                    # Find relationship indicators between them
                    relationship_type = self._detect_relationship_type(pair[0], pair[1], sent.text)
                    
                    if relationship_type:
                        relationships.append({
                            "entity1": pair[0],
                            "entity2": pair[1],
                            "type": relationship_type,
                            "context": sent.text
                        })
                        
                        # Update relationship graph if tracking user
                        if self.user_id:
                            self._update_relationship_graph(pair[0], pair[1], relationship_type)
        
        # Look for explicit relationship indicators
        relationship_patterns = [
            # Family
            (r'(?P<person1>[A-Z][a-z]+ [A-Z][a-z]+)[\s\w]* (?:is|as) (?:my|his|her|their) (?P<relation>mother|father|brother|sister|uncle|aunt|cousin|grandparent|son|daughter)', "FAMILY"),
            
            # Work
            (r'(?P<person1>[A-Z][a-z]+ [A-Z][a-z]+)[\s\w]* (?:is|as) (?:my|his|her|their) (?P<relation>colleague|coworker|boss|employee|manager|supervisor|client|partner)', "PROFESSIONAL"),
            
            # Friendship
            (r'(?P<person1>[A-Z][a-z]+ [A-Z][a-z]+)[\s\w]* (?:is|as) (?:my|his|her|their) (?P<relation>friend|best friend|buddy|pal)', "FRIENDSHIP"),
            
            # Romantic
            (r'(?P<person1>[A-Z][a-z]+ [A-Z][a-z]+)[\s\w]* (?:is|as) (?:my|his|her|their) (?P<relation>boyfriend|girlfriend|partner|spouse|husband|wife|fiancé|fiancée)', "ROMANTIC"),
            
            # Location relationship
            (r'(?P<person1>[A-Z][a-z]+ [A-Z][a-z]+) (?:lives|works|stays|resides) (?:in|at|near) (?P<place>[A-Z][a-z]+)', "LOCATION"),
        ]
        
        for pattern, rel_type in relationship_patterns:
            for match in re.finditer(pattern, text):
                match_dict = match.groupdict()
                if 'person1' in match_dict and ('person2' in match_dict or 'place' in match_dict or 'relation' in match_dict):
                    entity1 = match_dict['person1']
                    
                    # Determine entity2 based on available fields
                    entity2 = match_dict.get('person2', match_dict.get('place', match_dict.get('relation', '')))
                    
                    relationships.append({
                        "entity1": entity1,
                        "entity2": entity2,
                        "type": rel_type,
                        "context": match.group(0)
                    })
                    
                    # Update relationship graph
                    if self.user_id:
                        self._update_relationship_graph(entity1, entity2, rel_type)
        
        return relationships
    
    def _detect_relationship_type(self, entity1: str, entity2: str, context: str) -> Optional[str]:
        """Detect the type of relationship between two entities based on context."""
        context_lower = context.lower()
        
        # Family relationship indicators
        family_indicators = ["mother", "father", "sister", "brother", "daughter", "son",
                           "aunt", "uncle", "cousin", "grandmother", "grandfather", 
                           "family", "relative", "parent", "child", "sibling"]
        
        # Friend relationship indicators
        friend_indicators = ["friend", "buddy", "pal", "mate", "bestie", "bff",
                           "hung out", "together", "meet up", "catch up"]
        
        # Work relationship indicators
        work_indicators = ["colleague", "coworker", "boss", "employee", "supervisor",
                         "manager", "client", "team", "project", "work", "job", 
                         "office", "professional", "business"]
        
        # Romantic relationship indicators
        romantic_indicators = ["girlfriend", "boyfriend", "partner", "spouse", "husband",
                             "wife", "date", "dating", "romantic", "love", "relationship",
                             "together", "couple"]
        
        # Location relationship indicators (person to place)
        location_indicators = ["lives in", "works in", "visited", "went to", "located in",
                             "based in", "moved to", "staying in", "traveling to"]
        
        # Check for indicators in the context
        if any(indicator in context_lower for indicator in family_indicators):
            return "FAMILY"
        elif any(indicator in context_lower for indicator in friend_indicators):
            return "FRIENDSHIP"
        elif any(indicator in context_lower for indicator in work_indicators):
            return "PROFESSIONAL"
        elif any(indicator in context_lower for indicator in romantic_indicators):
            return "ROMANTIC"
        elif any(indicator in context_lower for indicator in location_indicators):
            return "LOCATION"
        
        # If no specific indicator found, check if these entities are in the relationship graph
        if self.user_id and self.relationship_graph.has_edge(entity1, entity2):
            return self.relationship_graph[entity1][entity2]["type"]
        
        # Default to generic association if no pattern detected
        return "ASSOCIATED"
    
    def _update_relationship_graph(self, entity1: str, entity2: str, relationship_type: str):
        """Update the relationship graph with new information."""
        if not self.user_id:
            return
            
        # Add nodes if they don't exist
        if not self.relationship_graph.has_node(entity1):
            self.relationship_graph.add_node(entity1)
        
        if not self.relationship_graph.has_node(entity2):
            self.relationship_graph.add_node(entity2)
        
        # Update or add the edge with the relationship type
        if self.relationship_graph.has_edge(entity1, entity2):
            # If we already have this relationship, only update if new info is more specific
            current_type = self.relationship_graph[entity1][entity2]["type"]
            if current_type == "ASSOCIATED" or current_type == relationship_type:
                self.relationship_graph[entity1][entity2]["type"] = relationship_type
                self.relationship_graph[entity1][entity2]["count"] += 1
        else:
            self.relationship_graph.add_edge(entity1, entity2, type=relationship_type, count=1)
    
    def _extract_locations(self, doc, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract location information with enhanced understanding.
        Identifies places, settings, and spatial relationships.
        """
        locations = []
        
        # Extract location entities
        location_entities = [entity for entity in entities if entity["type"] in ["LOC", "GPE", "FAC"]]
        
        # Extract from prepositions
        preposition_locations = []
        
        for token in doc:
            if token.dep_ == "prep" and token.text.lower() in ["in", "at", "near", "from", "to"]:
                for child in token.children:
                    if child.dep_ == "pobj" and not any(ent["text"] == child.text for ent in location_entities):
                        # Extract full location phrase
                        location_text = self._get_span_text(child)
                        
                        preposition_locations.append({
                            "text": location_text,
                            "preposition": token.text,
                            "context": token.sent.text,
                            "source": "preposition"
                        })
        
        # Combine both sources of locations
        for entity in location_entities:
            # Find contexts where this location is mentioned
            contexts = []
            for sent in doc.sents:
                if entity["text"] in sent.text:
                    contexts.append(sent.text)
            
            locations.append({
                "text": entity["text"],
                "type": entity["type"],
                "contexts": contexts,
                "metadata": entity.get("metadata", {}),
                "source": "named_entity"
            })
        
        # Add preposition-based locations
        for loc in preposition_locations:
            # Check if this location is already included
            if not any(existing["text"].lower() == loc["text"].lower() for existing in locations):
                locations.append({
                    "text": loc["text"],
                    "type": "LOCATION",
                    "contexts": [loc["context"]],
                    "preposition": loc["preposition"],
                    "source": loc["source"]
                })
        
        # Classify locations
        classified_locations = self._classify_locations(locations, doc)
        
        return classified_locations
    
    def _classify_locations(self, locations: List[Dict[str, Any]], doc) -> List[Dict[str, Any]]:
        """Classify and enrich location information."""
        enriched_locations = []
        
        location_types = {
            "HOME": ["home", "house", "apartment", "place", "flat", "condo", "my place", "where I live"],
            "WORK": ["office", "workplace", "work", "company", "job", "headquarters", "desk"],
            "EDUCATION": ["school", "university", "college", "campus", "class", "classroom", "lecture"],
            "RECREATION": ["park", "beach", "restaurant", "cafe", "gym", "theater", "cinema", "mall", "bar"],
            "TRANSPORT": ["airport", "station", "terminal", "bus stop", "subway", "train", "plane"],
            "MEDICAL": ["hospital", "clinic", "doctor", "dentist", "medical center", "pharmacy"],
            "SHOPPING": ["store", "shop", "market", "supermarket", "mall"],
            "OUTDOORS": ["mountain", "lake", "river", "forest", "trail", "hiking", "camping", "beach"]
        }
        
        for location in locations:
            location_text = location["text"].lower()
            location_type = "GENERAL"
            
            # Check specific location types
            for type_name, indicators in location_types.items():
                if any(indicator in location_text for indicator in indicators):
                    location_type = type_name
                    break
            
            # Check contexts for additional indicators
            for context in location.get("contexts", []):
                context_lower = context.lower()
                for type_name, indicators in location_types.items():
                    if any(indicator in context_lower for indicator in indicators):
                        location_type = type_name
                        break
            
            # Set frequency if in user memory
            frequency = 1
            if self.user_id and location["text"] in self.user_entities:
                frequency = self.user_entities[location["text"]]["count"]
            
            # Determine familiarity
            familiarity = "UNKNOWN"
            if frequency > 10:
                familiarity = "FREQUENT"
            elif frequency > 3:
                familiarity = "FAMILIAR"
            elif frequency > 1:
                familiarity = "VISITED"
            
            # Add enriched location
            enriched_location = location.copy()
            enriched_location["subtype"] = location_type
            enriched_location["frequency"] = frequency
            enriched_location["familiarity"] = familiarity
            
            # Add sentiment about this location
            sent_doc = None
            for context in location.get("contexts", []):
                sent_doc = self.nlp(context)
                break
                
            if sent_doc:
                loc_sentiment = self._analyze_sentiment(sent_doc.text)
                enriched_location["sentiment"] = loc_sentiment
            
            enriched_locations.append(enriched_location)
        
        return enriched_locations
    
    def _detect_reflections(self, text: str, doc) -> List[Dict[str, Any]]:
        """
        Detect reflective thinking and personal insights in the journal entry.
        Identifies areas where the person is analyzing their thoughts, feelings, or experiences.
        """
        reflections = []
        
        # Reflection indicators
        reflection_indicators = [
            # Cognitive reflection
            "realized", "understand", "recognize", "reflect", "thinking about", 
            "thought about", "came to understand", "insight", "epiphany",
            "makes sense", "clear to me", "see now", "perspective", "point of view",
            
            # Learning and growth
            "learned", "grew", "discovered", "found out", "developed", 
            "improved", "progress", "better at", "journey", "path",
            
            # Emotional reflection
            "feel about", "felt about", "emotional", "processing", "working through",
            "dealing with", "coping with", "handling", "managing emotions",
            
            # Self-awareness
            "self-aware", "aware of", "noticed I", "pattern", "tendency",
            "characteristic", "trait", "habit", "typical of me", "just like me",
            
            # Meaning-making
            "meaning", "purpose", "significance", "important to me", "value",
            "priority", "what matters", "meaningful", "fulfilling", "satisfaction",
            
            # Future-oriented
            "goal", "aspiration", "want to", "hope to", "plan to", "intend to",
            "future", "forward", "next step", "direction", "path forward"
        ]
        
        # Check each sentence for reflection markers
        for sent in doc.sents:
            sent_text = sent.text.lower()
            
            # Look for reflection indicators
            reflection_found = False
            for indicator in reflection_indicators:
                if indicator in sent_text:
                    # Determine reflection type
                    if any(term in indicator for term in ["realized", "understand", "thinking", "thought"]):
                        reflection_type = "COGNITIVE"
                    elif any(term in indicator for term in ["learned", "grew", "discovered", "developed"]):
                        reflection_type = "LEARNING"
                    elif any(term in indicator for term in ["feel", "felt", "emotional", "processing"]):
                        reflection_type = "EMOTIONAL"
                    elif any(term in indicator for term in ["self", "aware", "noticed", "pattern", "tendency"]):
                        reflection_type = "SELF_AWARENESS"
                    elif any(term in indicator for term in ["meaning", "purpose", "significance", "important"]):
                        reflection_type = "MEANING"
                    elif any(term in indicator for term in ["goal", "aspiration", "want", "hope", "plan", "future"]):
                        reflection_type = "FUTURE"
                    else:
                        reflection_type = "GENERAL"
                    
                    reflections.append({
                        "text": sent.text,
                        "type": reflection_type,
                        "indicator": indicator
                    })
                    
                    reflection_found = True
                    break
            
            # Additionally, check for questions as reflections
            if not reflection_found and sent_text.endswith("?") and len(sent_text.split()) > 5:
                # Check if it's a self-directed question
                if any(term in sent_text for term in ["i ", "me", "my ", "myself", "we ", "our "]):
                    reflections.append({
                        "text": sent.text,
                        "type": "SELF_QUESTIONING",
                        "indicator": "question"
                    })
        
        return reflections
    
    def _extract_key_actions(self, doc) -> List[Dict[str, Any]]:
        """
        Extract key actions performed by the journal writer.
        Identifies what the person did, is doing, or plans to do.
        """
        actions = []
        
        # Find main verbs with first-person subjects
        for sent in doc.sents:
            main_verbs = []
            
            # Find main verbs in the sentence
            for token in sent:
                if token.pos_ == "VERB" and not token.is_stop:
                    # Check if it has a first-person subject
                    has_first_person = False
                    
                    for child in token.children:
                        if child.dep_ == "nsubj" and child.text.lower() in ["i", "we"]:
                            has_first_person = True
                            break
                    
                    if has_first_person:
                        main_verbs.append(token)
            
            # Get full verb phrases
            for verb in main_verbs:
                verb_phrase = self._get_verb_phrase(verb)
                
                # Get objects and complements
                objects = []
                for child in verb.children:
                    if child.dep_ in ["dobj", "pobj", "attr", "acomp"]:
                        objects.append(self._get_span_text(child))
                
                # Determine action status
                if verb.tag_ in ["VBD", "VBN"]:  # Past tense
                    status = "COMPLETED"
                elif verb.tag_ in ["VBG"]:  # Present progressive
                    status = "IN_PROGRESS"
                elif any(aux.text.lower() in ["will", "going to", "plan to"] for aux in verb.children if aux.dep_ == "aux"):
                    status = "PLANNED"
                else:
                    status = "GENERAL"
                
                # Create action entry
                action = {
                    "verb": verb.text,
                    "lemma": verb.lemma_,
                    "phrase": verb_phrase,
                    "objects": objects,
                    "status": status,
                    "context": sent.text
                }
                
                actions.append(action)
        
        return actions
    
    def _extract_questions(self, doc) -> List[Dict[str, Any]]:
        """
        Extract questions posed in the journal entry.
        Identifies what the person is wondering about or questioning.
        """
        questions = []
        
        # Extract sentences ending with question marks
        for sent in doc.sents:
            if sent.text.strip().endswith("?"):
                # Determine question type
                question_type = self._determine_question_type(sent)
                
                # Determine if self-directed or external
                is_self_directed = any(token.text.lower() in ["i", "me", "my", "myself", "we", "us", "our", "ourselves"] 
                                     for token in sent)
                
                # Extract mentioned entities
                entities = [ent.text for ent in sent.ents]
                
                # Create question entry
                question = {
                    "text": sent.text,
                    "type": question_type,
                    "self_directed": is_self_directed,
                    "mentioned_entities": entities
                }
                
                questions.append(question)
        
        return questions
    
    def _determine_question_type(self, question_sent) -> str:
        """Determine the type of question being asked."""
        first_word = question_sent[0].text.lower()
        
        # Wh-questions
        if first_word in ["what", "when", "where", "why", "who", "whom", "whose", "which", "how"]:
            # Subcategorize wh-questions
            if first_word == "what":
                return "WHAT_QUESTION"
            elif first_word == "when":
                return "WHEN_QUESTION"  # Temporal
            elif first_word == "where":
                return "WHERE_QUESTION"  # Spatial
            elif first_word == "why":
                return "WHY_QUESTION"  # Causal
            elif first_word in ["who", "whom", "whose"]:
                return "WHO_QUESTION"  # Person
            elif first_word == "how":
                if len(question_sent) > 1 and question_sent[1].text.lower() in ["many", "much"]:
                    return "QUANTITY_QUESTION"  # Quantitative
                elif len(question_sent) > 1 and question_sent[1].text.lower() in ["long", "often", "frequently"]:
                    return "FREQUENCY_QUESTION"  # Frequency/duration
                else:
                    return "HOW_QUESTION"  # Process/method
            else:
                return "WH_QUESTION"
                
        # Yes/no questions
        elif first_word in ["is", "are", "was", "were", "do", "does", "did", "have", "has", "had", 
                          "can", "could", "will", "would", "should", "may", "might"]:
            return "YES_NO_QUESTION"
            
        # Alternative questions (containing "or")
        elif " or " in question_sent.text:
            return "ALTERNATIVE_QUESTION"
            
        # Tag questions
        elif ", " in question_sent.text and any(tag in question_sent.text.lower() for tag in 
                                             ["isn't it", "aren't they", "don't you", "didn't I", "won't he"]):
            return "TAG_QUESTION"
        
        # Default
        return "GENERAL_QUESTION"
    
    def _detect_recurring_themes(self, text: str, topics: List[str], entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect recurring themes and patterns based on user history.
        Identifies what topics, entities, and concerns appear frequently in journal entries.
        """
        if not self.user_id:
            return []
            
        recurring_themes = []
        entity_texts = [entity["text"] for entity in entities]
        
        # Check user entities for recurring people, places, etc.
        frequent_entities = []
        for entity in entity_texts:
            if entity in self.user_entities and self.user_entities[entity]["count"] > 3:
                entity_data = self.user_entities[entity]
                frequent_entities.append({
                    "text": entity,
                    "count": entity_data["count"],
                    "metadata": entity_data["metadata"]
                })
        
        # Sort by frequency
        frequent_entities.sort(key=lambda x: x["count"], reverse=True)
        
        # Get top 5 frequent entities
        top_entities = frequent_entities[:5]
        if top_entities:
            recurring_themes.append({
                "type": "RECURRING_ENTITIES",
                "entities": top_entities
            })
        
        # Check for recurring topics
        frequent_topics = {}
        for topic in topics:
            topic_lower = topic.lower()
            if topic_lower in self.user_concepts:
                concept_data = self.user_concepts[topic_lower]
                frequent_topics[topic] = concept_data["count"]
        
        # Get top topics
        if frequent_topics:
            top_topics = sorted(frequent_topics.items(), key=lambda x: x[1], reverse=True)[:5]
            recurring_themes.append({
                "type": "RECURRING_TOPICS",
                "topics": [{"text": topic, "count": count} for topic, count in top_topics]
            })
        
        # Detect frequently occurring emotional patterns
        if self.emotion_history and len(self.emotion_history) >= 3:
            # Count emotion occurrences
            emotion_counts = Counter([e["emotion"] for e in self.emotion_history])
            predominant_emotion = emotion_counts.most_common(1)[0][0]
            
            # Check if current entry contains the predominant emotion
            current_emotions = self._analyze_emotions(text, self.nlp(text))
            if predominant_emotion == current_emotions.get("primary_emotion"):
                recurring_themes.append({
                    "type": "EMOTIONAL_PATTERN",
                    "emotion": predominant_emotion,
                    "frequency": emotion_counts[predominant_emotion]
                })
        
        # Detect recurring event types
        if self.narrative_memory:
            event_types = [mem.get("event_type") for mem in self.narrative_memory if "event_type" in mem]
            if event_types:
                event_type_counts = Counter(event_types)
                top_event_types = event_type_counts.most_common(3)
                
                recurring_themes.append({
                    "type": "RECURRING_EVENT_TYPES",
                    "event_types": [{"type": et, "count": count} for et, count in top_event_types]
                })
        
        return recurring_themes
    
    def _detect_implicit_references(self, text: str, doc) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect implicit references to entities, events, or concepts.
        Identifies references that aren't explicitly named but can be inferred.
        """
        implicit_references = {
            "entities": [],
            "events": [],
            "concepts": []
        }
        
        # Look for demonstrative pronouns that might refer to entities
        for token in doc:
            if token.tag_ == "DT" and token.text.lower() in ["this", "that", "these", "those"]:
                # Check if it's followed by a noun
                has_noun = False
                for child in token.children:
                    if child.pos_ in ["NOUN", "PROPN"]:
                        has_noun = True
                        break
                
                # If not followed by a noun, it's likely referring to something implicit
                if not has_noun:
                    # Try to determine the referent
                    referent = self._determine_implicit_referent(token, doc)
                    
                    if referent:
                        implicit_references["entities"].append({
                            "text": token.text,
                            "sentence": token.sent.text,
                            "likely_referent": referent,
                            "confidence": 0.7  # Arbitrary confidence
                        })
        
        # Look for temporal references that might refer to events
        temporal_refs = ["then", "after that", "before that", "at that time", "the next day"]
        for ref in temporal_refs:
            if ref in text.lower():
                # Find sentences containing these references
                for sent in doc.sents:
                    if ref in sent.text.lower():
                        # Try to determine the event being referenced
                        event_ref = self._determine_implicit_event(sent, ref)
                        
                        if event_ref:
                            implicit_references["events"].append({
                                "text": ref,
                                "sentence": sent.text,
                                "likely_event": event_ref,
                                "confidence": 0.6
                            })
        
        # Look for conceptual references
        concept_refs = ["the situation", "the problem", "the issue", "the matter", 
                       "the topic", "the subject", "the idea", "the concept"]
        
        for ref in concept_refs:
            if ref in text.lower():
                # Find sentences containing these references
                for sent in doc.sents:
                    if ref in sent.text.lower():
                        # Try to determine the concept being referenced
                        concept_ref = self._determine_implicit_concept(sent, ref)
                        
                        if concept_ref:
                            implicit_references["concepts"].append({
                                "text": ref,
                                "sentence": sent.text,
                                "likely_concept": concept_ref,
                                "confidence": 0.5
                            })
        
        return implicit_references
    
    def _determine_implicit_referent(self, token, doc) -> Optional[str]:
        """Determine the likely referent of an implicit entity reference."""
        # Check previous sentences for entities
        sent_idx = 0
        for i, sent in enumerate(doc.sents):
            if token in sent:
                sent_idx = i
                break
        
        previous_entities = []
        for i, sent in enumerate(doc.sents):
            if i < sent_idx:  # Only check previous sentences
                for ent in sent.ents:
                    previous_entities.append((ent.text, sent_idx - i))  # Include distance
        
        if previous_entities:
            # Sort by distance (closest first)
            previous_entities.sort(key=lambda x: x[1])
            return previous_entities[0][0]
        
        return None
    
    def _determine_implicit_event(self, sent, ref) -> Optional[str]:
        """Determine the likely event being implicitly referenced."""
        # Simple approach: get the previous sentence
        doc = sent.doc
        prev_sent = None
        
        for i, s in enumerate(doc.sents):
            if s.text == sent.text and i > 0:
                sentences = list(doc.sents)
                prev_sent = sentences[i-1].text
                break
        
        if prev_sent:
            # Return the previous sentence as the likely event context
            return prev_sent
        
        return None
    
    def _determine_implicit_concept(self, sent, ref) -> Optional[str]:
        """Determine the likely concept being implicitly referenced."""
        # Simple approach: look for abstract nouns in previous sentences
        abstract_concepts = ["idea", "thought", "concept", "theory", "belief", "opinion", 
                           "view", "perspective", "approach", "strategy", "method", 
                           "plan", "proposal", "suggestion", "decision", "choice"]
        
        doc = sent.doc
        
        # Find the current sentence index
        sent_idx = 0
        for i, s in enumerate(doc.sents):
            if s.text == sent.text:
                sent_idx = i
                break
        
        # Check previous sentences for abstract concepts
        for i in range(max(0, sent_idx - 3), sent_idx):
            prev_sent = list(doc.sents)[i]
            
            # Check if any abstract concept words appear
            for concept in abstract_concepts:
                if concept in prev_sent.text.lower():
                    return prev_sent.text
        
        return None
    
    def _extract_causal_relationships(self, text: str, doc) -> List[Dict[str, Any]]:
        """
        Extract causal relationships between events, actions, or states.
        Identifies cause-effect relationships in the journal entry.
        """
        causal_relationships = []
        
        # Causal markers
        cause_markers = ["because", "since", "as", "due to", "thanks to", "caused by", "result of"]
        effect_markers = ["therefore", "thus", "consequently", "as a result", "so", "hence", "that's why"]
        
        # Check each sentence for causal relationships
        for sent in doc.sents:
            sent_text = sent.text.lower()
            
            # Check for explicit cause markers
            for marker in cause_markers:
                if marker in sent_text:
                    # Split by the marker to get cause and effect
                    parts = re.split(r'\b' + re.escape(marker) + r'\b', sent_text, 1)
                    
                    if len(parts) == 2:
                        relation = {
                            "type": "CAUSE",
                            "marker": marker,
                            "effect": parts[0].strip(),
                            "cause": parts[1].strip(),
                            "text": sent.text
                        }
                        causal_relationships.append(relation)
            
            # Check for explicit effect markers
            for marker in effect_markers:
                if marker in sent_text:
                    # Split by the marker to get cause and effect
                    parts = re.split(r'\b' + re.escape(marker) + r'\b', sent_text, 1)
                    
                    if len(parts) == 2:
                        relation = {
                            "type": "EFFECT",
                            "marker": marker,
                            "cause": parts[0].strip(),
                            "effect": parts[1].strip(),
                            "text": sent.text
                        }
                        causal_relationships.append(relation)
        
        # Also check for implicit causal relationships across sentences
        sentences = list(doc.sents)
        for i in range(1, len(sentences)):
            prev_sent = sentences[i-1].text.lower()
            curr_sent = sentences[i].text.lower()
            
            # Check if current sentence starts with an effect marker
            for marker in effect_markers:
                if curr_sent.startswith(marker):
                    relation = {
                        "type": "IMPLICIT_EFFECT",
                        "marker": marker,
                        "cause": prev_sent,
                        "effect": curr_sent.replace(marker, "").strip(),
                        "text": f"{sentences[i-1].text} {sentences[i].text}"
                    }
                    causal_relationships.append(relation)
        
        return causal_relationships
    
    def _generate_summary(self, text: str, entities: List[Dict[str, Any]], 
                         events: List[Dict[str, Any]], emotions: Dict[str, Any], 
                         topics: List[str]) -> Dict[str, Any]:
        """
        Generate a structured summary of the journal entry.
        Creates a concise representation of the key information.
        """
        # Calculate entry characteristics
        num_sentences = len(list(self.nlp(text).sents))
        word_count = len(text.split())
        
        # Get primary sentiment
        sentiment = self._analyze_sentiment(text)
        
        # Get primary emotion if available
        primary_emotion = emotions.get("primary_emotion") if emotions else None
        
        # Determine if the entry is event-focused, emotional, or reflective
        event_ratio = len(events) / max(1, num_sentences)
        
        reflection_indicators = ["realized", "learned", "understand", "thinking", "reflect"]
        reflection_count = sum(1 for indicator in reflection_indicators if indicator in text.lower())
        reflection_ratio = reflection_count / max(1, num_sentences)
        
        emotion_words = []
        for emotion_category in self.emotion_lexicon.values():
            emotion_words.extend(emotion_category.keys())
        
        emotion_count = sum(1 for word in emotion_words if word in text.lower())
        emotion_ratio = emotion_count / max(1, word_count)
        
        # Determine entry focus
        if event_ratio > 0.5:
            entry_focus = "EVENT_FOCUSED"
        elif emotion_ratio > 0.1:
            entry_focus = "EMOTIONAL"
        elif reflection_ratio > 0.2:
            entry_focus = "REFLECTIVE"
        else:
            entry_focus = "BALANCED"
        
        # Generate key points
        key_points = []
        
        # Add most significant events
        if events:
            # Sort events by importance
            sorted_events = sorted(events, key=lambda x: x.get("importance", "MEDIUM"))
            top_events = sorted_events[:min(3, len(sorted_events))]
            
            for event in top_events:
                key_points.append({
                    "type": "EVENT",
                    "text": event.get("text", ""),
                    "importance": event.get("importance", "MEDIUM")
                })
        
        # Add emotions if significant
        if primary_emotion:
            key_points.append({
                "type": "EMOTION",
                "text": f"Expressed {primary_emotion}",
                "importance": "HIGH" if emotions.get("intensity", 0) > 0.7 else "MEDIUM"
            })
        
        # Add top entities
        top_entities = sorted(entities, key=lambda x: x.get("mentions", 1), reverse=True)[:3]
        for entity in top_entities:
            key_points.append({
                "type": "ENTITY",
                "text": f"Mentioned {entity['text']}",
                "importance": "MEDIUM"
            })
        
        # Build the summary
        summary = {
            "entry_focus": entry_focus,
            "sentiment": sentiment["label"],
            "main_topics": topics[:3] if topics else [],
            "key_points": key_points,
            "word_count": word_count,
            "event_count": len(events),
            "entity_count": len(entities),
            "primary_emotion": primary_emotion
        }
        
        return summary
    
    def _update_memory(self, text: str, entities: List[Dict[str, Any]], 
                      events: List[Dict[str, Any]], emotions: Dict[str, Any], 
                      topics: List[str], entry_date: Optional[str] = None):
        """
        Update user-specific memory with information from this entry.
        Builds up a personalized knowledge base about the user's journal.
        """
        if not self.user_id:
            return
            
        current_date = entry_date or datetime.now().isoformat()
        
        # Update concept/topic memory
        for topic in topics:
            topic_lower = topic.lower()
            
            # Increment count
            self.user_concepts[topic_lower]["count"] += 1
            
            # Set first mentioned date if not already set
            if not self.user_concepts[topic_lower]["first_mentioned"]:
                self.user_concepts[topic_lower]["first_mentioned"] = current_date
            
            # Update related concepts
            for other_topic in topics:
                other_lower = other_topic.lower()
                if other_lower != topic_lower:
                    self.user_concepts[topic_lower]["related_to"].add(other_lower)
        
        # Store narrative memory
        event_summary = None
        if events:
            primary_event = max(events, key=lambda x: x.get("importance", "MEDIUM"))
            event_summary = {
                "event_type": primary_event.get("type", "OTHER"),
                "description": primary_event.get("text", ""),
                "date": current_date,
                "entities": [e["text"] for e in entities if e["text"] in primary_event.get("text", "")],
                "sentiment": self._analyze_sentiment(primary_event.get("text", ""))["label"]
            }
            
            self.narrative_memory.append(event_summary)
        
        # Clean up memory (limit size)
        if len(self.narrative_memory) > 100:
            self.narrative_memory = self.narrative_memory[-100:]
        
        # Save memory to disk
        self._save_memory()
    
    def _detect_questions(self, doc) -> List[Dict[str, Any]]:
        """
        Detect questions the user is asking themselves in their journal.
        Identifies areas of uncertainty or curiosity.
        """
        questions = []
        
        for sent in doc.sents:
            # Check if sentence ends with question mark
            if sent.text.strip().endswith("?"):
                # Determine if it's self-reflective or about external matters
                is_self_reflective = any(token.text.lower() in ["i", "me", "my", "myself"] for token in sent)
                
                # Determine question type by first word
                question_type = "GENERAL"
                first_word = sent[0].text.lower()
                
                if first_word == "why":
                    question_type = "WHY"  # Reasons, causes
                elif first_word == "how":
                    question_type = "HOW"  # Methods, processes
                elif first_word == "what":
                    question_type = "WHAT"  # Definitions, identifications
                elif first_word == "when":
                    question_type = "WHEN"  # Timing
                elif first_word == "where":
                    question_type = "WHERE"  # Location
                elif first_word == "who":
                    question_type = "WHO"  # People
                elif first_word in ["should", "could", "would"]:
                    question_type = "HYPOTHETICAL"  # Hypotheticals
                elif first_word in ["am", "is", "are", "was", "were", "will", "do", "does", "did"]:
                    question_type = "YES_NO"  # Yes/No questions
                
                questions.append({
                    "text": sent.text,
                    "type": question_type,
                    "self_reflective": is_self_reflective
                })
        
        return questions
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate a vector embedding for semantic search and similarity."""
        if self.embedding_model:
            try:
                # Generate embedding and convert to list for database storage
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
        
        # Return empty list if embedding model not available
        return []
    
    def _detect_topics(self, doc) -> List[str]:
        """Detect main topics with improved topic modeling."""
        # Extract noun chunks as potential topics
        noun_chunks = list(doc.noun_chunks)
        
        # Filter out pronouns and determiners
        filtered_chunks = []
        for chunk in noun_chunks:
            # Skip chunks that start with pronouns or determiners
            if chunk[0].pos_ not in ["PRON", "DET"]:
                # Skip chunks that are too short
                if len(chunk) > 1 or (len(chunk) == 1 and len(chunk[0].text) > 3):
                    # Skip common stop phrases
                    if chunk.text.lower() not in ["lot", "kind", "sort", "type", "way", "thing", "stuff"]:
                        filtered_chunks.append(chunk.text)
        
        # Count occurrences to find most common topics
        topic_counter = Counter(filtered_chunks)
        
        # Also add named entities as topics if they're not already included
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART"]:
                if ent.text not in topic_counter:
                    topic_counter[ent.text] = 1
                else:
                    topic_counter[ent.text] += 1
        
        # Get most frequent topics
        top_topics = [topic for topic, _ in topic_counter.most_common(5)]
        
        # Try to extract abstract topics if we don't have enough concrete ones
        if len(top_topics) < 3:
            # Look for abstract concepts
            abstract_topics = self._extract_abstract_topics(doc)
            
            # Add any new topics
            for topic in abstract_topics:
                if topic not in top_topics:
                    top_topics.append(topic)
                    
                    # Stop if we have enough
                    if len(top_topics) >= 5:
                        break
        
        return top_topics
    
    def _extract_abstract_topics(self, doc) -> List[str]:
        """Extract abstract topics or themes from the text."""
        abstract_concepts = [
            "work", "career", "job", "business", "company", "profession", 
            "education", "school", "study", "learning", "knowledge", "class",
            "health", "fitness", "well-being", "diet", "exercise", "medical",
            "relationship", "family", "friend", "partner", "marriage", "love",
            "finance", "money", "investment", "saving", "expense", "budget",
            "travel", "trip", "journey", "exploration", "vacation", "destination",
            "hobby", "interest", "passion", "activity", "entertainment", "leisure",
            "challenge", "problem", "issue", "difficulty", "obstacle", "hurdle",
            "goal", "objective", "aim", "aspiration", "dream", "ambition",
            "emotion", "feeling", "mood", "mental health", "psychology", "mindset"
        ]
        
        # Count occurrences of abstract concepts
        concept_counts = Counter()
        
        for token in doc:
            if token.lemma_.lower() in abstract_concepts:
                concept_counts[token.lemma_.lower()] += 1
        
        # Get most common concepts
        return [concept for concept, _ in concept_counts.most_common(5)]
    
    def find_similar_events(self, new_event: Dict[str, Any], existing_events: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
        """
        Enhanced function to find similar events with better matching capabilities.
        Uses multiple matching techniques with a weighted approach.
        
        Args:
            new_event: The event to find matches for
            existing_events: List of previous events to search through
            
        Returns:
            Tuple containing the most similar event and the similarity score
        """
        if not existing_events:
            return None, 0
        
        # Extract new event text and type
        new_text = new_event.get("text", "")
        new_type = new_event.get("type", "OTHER")
        
        # Extract new event metadata
        new_components = new_event.get("components", {})
        new_participants = set(new_components.get("participants", []))
        new_location = new_components.get("location", "").lower()
        new_times = set(new_components.get("time", []))
        new_action = new_components.get("action", "").lower()
        
        # Extract important keywords from the new event text
        new_keywords = self._extract_important_keywords(new_text)
        
        # Calculate similarity with each existing event
        max_similarity = 0
        most_similar_event = None
        debug_info = {}
        
        for event in existing_events:
            # We'll allow matching completed events too, but with a slight penalty
            completion_penalty = 0.85 if event.get("status") == "completed" else 1.0
            
            # Extract existing event data
            existing_text = event.get("text", "")
            existing_name = event.get("name", "")
            if not existing_text:
                continue
            
            # Extract keywords from existing event
            existing_keywords = self._extract_important_keywords(existing_text + " " + existing_name)
            
            # Initialize similarity components
            embedding_similarity = 0
            keyword_similarity = 0
            component_similarity = 0
            
            # 1. Calculate embedding similarity if possible
            if self.embedding_model:
                try:
                    new_embedding = self.embedding_model.encode(new_text)
                    
                    # If embedding exists in the event
                    if "embedding" in event:
                        existing_embedding = torch.tensor(event["embedding"])
                        embedding_similarity = util.pytorch_cos_sim(
                            new_embedding, 
                            existing_embedding
                        ).item()
                    else:
                        # Generate embedding if not stored
                        existing_embedding = self.embedding_model.encode(existing_text)
                        embedding_similarity = util.pytorch_cos_sim(
                            new_embedding, 
                            existing_embedding
                        ).item()
                        
                except Exception as e:
                    logger.error(f"Error calculating embedding similarity: {e}")
                    embedding_similarity = 0
            
            # 2. Calculate keyword similarity (more precise than pure embedding)
            if new_keywords and existing_keywords:
                common_keywords = set(new_keywords).intersection(set(existing_keywords))
                if common_keywords:
                    keyword_similarity = len(common_keywords) / max(len(new_keywords), len(existing_keywords))
                    
                    # Boost for highly significant matching keywords
                    significant_matches = [k for k in common_keywords if len(k) > 4]  # Longer words tend to be more specific
                    if significant_matches:
                        keyword_boost = min(0.3, 0.1 * len(significant_matches))
                        keyword_similarity = min(1.0, keyword_similarity + keyword_boost)
            
            # 3. Calculate component similarity (structured data)
            component_scores = []
            
            # 3.1 Participants similarity
            participant_similarity = 0
            existing_participants = set(event.get("participants", []))
            if new_participants and existing_participants:
                common_participants = new_participants.intersection(existing_participants)
                if common_participants:
                    participant_similarity = len(common_participants) / max(len(new_participants), len(existing_participants))
                    component_scores.append(("participants", participant_similarity))
            
            # 3.2 Location similarity
            location_similarity = 0
            existing_location = event.get("location", "").lower()
            if new_location and existing_location:
                # Check for exact or partial location match
                if new_location == existing_location:
                    location_similarity = 1.0
                elif new_location in existing_location or existing_location in new_location:
                    location_similarity = 0.7
                component_scores.append(("location", location_similarity))
            
            # 3.3 Action similarity (the verb/action in events)
            action_similarity = 0
            existing_action = ""
            if "action" in event:
                existing_action = event["action"].lower()
            elif "description" in event and len(event["description"].split()) >= 3:
                # Try to extract action from description if not directly available
                doc = self.nlp(event["description"])
                for token in doc:
                    if token.pos_ == "VERB" and not token.is_stop:
                        existing_action = token.lemma_
                        break
            
            if new_action and existing_action:
                # Check if actions are similar
                if new_action == existing_action:
                    action_similarity = 1.0
                elif new_action in existing_action or existing_action in new_action:
                    action_similarity = 0.7
                else:
                    # Check for related verbs
                    related_verb_pairs = [
                        # Test and result related
                        ("take", "receive"), ("take", "get"), ("do", "receive"), ("finish", "receive"),
                        ("complete", "get"), ("submit", "receive"), ("attend", "get"),
                        
                        # General action sequences
                        ("start", "finish"), ("begin", "complete"), ("attend", "leave"),
                        ("go", "return"), ("plan", "attend"), ("organize", "attend"),
                        ("invite", "meet"), ("schedule", "attend"), ("book", "visit")
                    ]
                    
                    for v1, v2 in related_verb_pairs:
                        if (new_action == v1 and existing_action == v2) or (new_action == v2 and existing_action == v1):
                            action_similarity = 0.8
                            break
                
                component_scores.append(("action", action_similarity))
            
            # 3.4 Time references similarity
            time_similarity = 0
            existing_times = set()
            if "time_mentions" in event:
                existing_times = set(event.get("time_mentions", []))
            
            if new_times and existing_times:
                common_times = new_times.intersection(existing_times)
                if common_times:
                    time_similarity = len(common_times) / max(len(new_times), len(existing_times))
                    component_scores.append(("time", time_similarity))
            
            # 3.5 Event name appearing in text
            name_similarity = 0
            if existing_name and new_text:
                name_keywords = self._extract_important_keywords(existing_name)
                text_keywords = self._extract_important_keywords(new_text)
                
                if name_keywords and text_keywords:
                    common_words = set(name_keywords).intersection(set(text_keywords))
                    if common_words:
                        name_similarity = len(common_words) / len(name_keywords)
                        component_scores.append(("name", name_similarity))
            
            # Calculate overall component similarity (average of all components)
            if component_scores:
                component_similarity = sum(score for _, score in component_scores) / len(component_scores)
            
            # 4. Semantic similarity for event sequences/stage recognition
            # (e.g., planning->execution->results)
            sequence_similarity = self._calculate_event_sequence_similarity(new_event, event)
            
            # 5. Type compatibility (same or related types)
            type_compatibility = self._calculate_event_type_compatibility(new_type, event.get("type", "OTHER"))
            
            # 6. Calculate final similarity with weighted components
            # Weights for different similarity measures - can be adjusted
            weights = {
                "embedding": 0.25,      # Semantic similarity
                "keyword": 0.25,        # Specific keyword matches
                "component": 0.20,      # Structured metadata
                "sequence": 0.20,       # Sequential event detection
                "type": 0.10            # Event type compatibility
            }
            
            # Combined similarity score
            combined_similarity = (
                embedding_similarity * weights["embedding"] +
                keyword_similarity * weights["keyword"] +
                component_similarity * weights["component"] +
                sequence_similarity * weights["sequence"] +
                type_compatibility * weights["type"]
            )
            
            # Apply completion penalty
            combined_similarity *= completion_penalty
            
            # Save debug info
            debug_info[event.get("id", f"event_{id(event)}")] = {
                "embedding": embedding_similarity,
                "keyword": keyword_similarity,
                "component": component_similarity,
                "component_details": component_scores,
                "sequence": sequence_similarity,
                "type_compatibility": type_compatibility,
                "combined": combined_similarity
            }
            
            # Track highest similarity
            if combined_similarity > max_similarity:
                max_similarity = combined_similarity
                most_similar_event = event
        
        # Add debug info to the most similar event if found
        if most_similar_event:
            most_similar_event["debug_similarity"] = debug_info[most_similar_event.get("id", f"event_{id(most_similar_event)}")]
        
        return most_similar_event, max_similarity
    
    def _calculate_event_sequence_similarity(self, new_event: Dict[str, Any], existing_event: Dict[str, Any]) -> float:
        """Calculate similarity based on event sequence patterns."""
        sequence_similarity = 0
        
        # Define event sequences that typically occur together
        event_sequences = {
            # Test sequence
            "test": ["prepare", "study", "take", "write", "finish", "complete", "receive", "get", "results", "grade", "score"],
            
            # Meeting sequence
            "meeting": ["schedule", "plan", "prepare", "attend", "participate", "discuss", "decide", "follow-up", "minutes"],
            
            # Trip sequence
            "trip": ["plan", "book", "pack", "travel", "arrive", "stay", "return", "unpack"],
            
            # Project sequence
            "project": ["assign", "plan", "work", "develop", "create", "build", "test", "review", "finish", "submit", "present"],
            
            # Social event sequence
            "social": ["invite", "plan", "attend", "meet", "enjoy", "leave", "follow-up"],
            
            # Medical sequence
            "medical": ["symptoms", "appointment", "visit", "diagnose", "treat", "prescribe", "follow-up", "recover"]
        }
        
        new_text = new_event.get("text", "").lower()
        existing_text = existing_event.get("text", "").lower()
        
        # Check if our events might be in a sequence
        for sequence_name, sequence_terms in event_sequences.items():
            new_matches = sum(1 for term in sequence_terms if term in new_text)
            existing_matches = sum(1 for term in sequence_terms if term in existing_text)
            
            if new_matches > 0 and existing_matches > 0:
                # They both match the same sequence
                sequence_similarity = 0.5
                
                # Check if they represent different stages of the same sequence
                new_stage_index = -1
                existing_stage_index = -1
                
                for i, term in enumerate(sequence_terms):
                    if term in new_text and new_stage_index == -1:
                        new_stage_index = i
                    if term in existing_text and existing_stage_index == -1:
                        existing_stage_index = i
                
                if new_stage_index != -1 and existing_stage_index != -1 and new_stage_index != existing_stage_index:
                    # If they're different stages of the same sequence, boost similarity
                    # Stages that are closer together get a higher boost
                    stage_distance = abs(new_stage_index - existing_stage_index)
                    if stage_distance <= 2:  # Adjacent or near stages
                        sequence_similarity = 0.9  # Strong boost for sequential events
                    elif stage_distance <= 4:
                        sequence_similarity = 0.7  # Medium boost
                    else:
                        sequence_similarity = 0.5  # Smaller boost for distant stages
                
                # Exit after finding a matching sequence
                if sequence_similarity > 0:
                    break
        
        return sequence_similarity
    
    def _calculate_event_type_compatibility(self, new_type: str, existing_type: str) -> float:
        """Calculate compatibility between event types."""
        if new_type == existing_type:
            return 1.0
            
        # Define related types
        related_types = {
            "WORK": ["EDUCATION", "PROFESSIONAL", "MEETING", "PROJECT"],
            "EDUCATION": ["WORK", "ACADEMIC", "LEARNING", "TEST"],
            "SOCIAL": ["FRIENDSHIP", "RELATIONSHIP", "GATHERING", "PARTY"],
            "HEALTH": ["MEDICAL", "FITNESS", "WELLNESS", "APPOINTMENT"],
            "TRAVEL": ["TRIP", "VACATION", "JOURNEY", "EXPLORATION"],
            "CREATIVE": ["HOBBY", "ART", "MUSIC", "WRITING", "PROJECT"]
        }
        
        # Check if types are related
        for primary_type, related in related_types.items():
            if new_type == primary_type and existing_type in related:
                return 0.8
            elif existing_type == primary_type and new_type in related:
                return 0.8
        
        # Default modest compatibility
        return 0.3
    
    def _extract_important_keywords(self, text):
        """Extract important keywords from text (nouns, verbs, and named entities)."""
        if not text:
            return []
        
        # Process with spaCy
        doc = self.nlp(text.lower())
        
        # Extract keywords (nouns, important verbs, and named entities)
        keywords = []
        
        # Add named entities
        for ent in doc.ents:
            if len(ent.text) > 1:  # Skip single-character entities
                keywords.append(ent.text)
        
        # Add important nouns and verbs (non-stopwords)
        for token in doc:
            if token.is_stop or token.is_punct or len(token.text) <= 2:
                continue
            
            if token.pos_ in ["NOUN", "PROPN"]:
                keywords.append(token.text)
            elif token.pos_ == "VERB":
                # Only include meaningful verbs
                if token.lemma_ not in ["be", "have", "do", "say", "go", "get", "make", "see", "know", "think", "take"]:
                    keywords.append(token.lemma_)
        
        # Remove duplicates and return
        return list(set(keywords))