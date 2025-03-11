import pymongo
import bcrypt
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
from pymongo import UpdateOne, InsertOne
import json
from bson.json_util import dumps, loads
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MongoDBClient:
    """Enhanced MongoDB client for Reflectify application with advanced vector and narrative support."""
    
    def __init__(self, connection_string="mongodb://localhost:27017/", 
                 use_atlas_vector_search=False,
                 vector_dimensions=768):
        """
        Initialize MongoDB connection with enhanced vector and narrative capabilities.
        
        Args:
            connection_string: MongoDB connection string
            use_atlas_vector_search: Set to True if using MongoDB Atlas with vector search capability
            vector_dimensions: Dimension of embedding vectors (depends on the model used in NLP engine)
        """
        self.client = pymongo.MongoClient(connection_string)
        self.db = self.client["reflectify_app_enhanced"]
        self.vector_dimensions = vector_dimensions
        self.use_atlas_vector_search = use_atlas_vector_search
        
        # Create core collections
        self.users = self.db["users"]
        self.journal_entries = self.db["journal_entries"]
        self.events = self.db["events"]
        self.entities = self.db["entities"]
        
        # Create additional collections for complex narrative understanding
        self.narratives = self.db["narratives"]  # Store narrative arcs and story structures
        self.entity_vectors = self.db["entity_vectors"]  # Store entity-specific embeddings
        self.event_vectors = self.db["event_vectors"]  # Store event-specific embeddings
        self.narrative_vectors = self.db["narrative_vectors"]  # Store narrative-level embeddings
        self.paragraph_vectors = self.db["paragraph_vectors"]  # Store paragraph-level embeddings
        self.entity_relationships = self.db["entity_relationships"]  # Store relationship graphs
        self.emotional_arcs = self.db["emotional_arcs"]  # Store emotional trajectories
        self.concepts = self.db["concepts"]  # Store abstract concepts and themes
        self.temporal_sequences = self.db["temporal_sequences"]  # Store temporal event sequences
        
        # Add new collections for thread tracking
        self.threads = self.db["threads"]
        self.thread_updates = self.db["thread_updates"]
        self.thread_relationships = self.db["thread_relationships"]
        self.thread_vectors = self.db["thread_vectors"]
        
        # Create core indexes
        self._create_indexes()
        
        # Set up vector search capability
        self._setup_vector_search()
    
    def _create_indexes(self):
        """Create database indexes for efficient queries."""
    # Try to drop existing indexes first to avoid conflicts
        try:
            self.entities.drop_index("user_id_1_text_1")
        except Exception as e:
            # Index might not exist, that's fine
            pass
        self.users.create_index("username", unique=True)
        self.journal_entries.create_index("user_id")
        self.journal_entries.create_index("date")
        self.events.create_index([("user_id", 1), ("event_type", 1)])
        self.events.create_index([("user_id", 1), ("status", 1)])
        self.events.create_index([("user_id", 1), ("last_mentioned", -1)])
        self.entities.create_index([("user_id", 1), ("text", 1)], unique=True)
        self.entities.create_index([("user_id", 1), ("type", 1)])
        
        # Text indexes for basic search
        self.journal_entries.create_index([("text", "text")])
        
        # Narrative understanding indexes
        self.narratives.create_index([("user_id", 1), ("entry_id", 1)])
        self.narratives.create_index([("user_id", 1), ("narrative_type", 1)])
        
        self.entity_vectors.create_index([("user_id", 1), ("entity_id", 1)])
        self.event_vectors.create_index([("user_id", 1), ("event_id", 1)])
        self.narrative_vectors.create_index([("user_id", 1), ("narrative_id", 1)])
        self.paragraph_vectors.create_index([("user_id", 1), ("entry_id", 1)])
        
        self.entity_relationships.create_index([("user_id", 1), ("source_id", 1), ("target_id", 1)])
        self.entity_relationships.create_index([("user_id", 1), ("relationship_type", 1)])
        
        self.emotional_arcs.create_index([("user_id", 1), ("entry_id", 1)])
        self.emotional_arcs.create_index([("user_id", 1), ("primary_emotion", 1)])
        
        self.concepts.create_index([("user_id", 1), ("concept_name", 1)], unique=True)
        
        self.temporal_sequences.create_index([("user_id", 1), ("sequence_type", 1)])
        self.temporal_sequences.create_index([("user_id", 1), ("start_date", -1)])
        
        # Add new indexes for thread tracking
        self.threads.create_index([("user_id", 1), ("thread_id", 1)], unique=True)
        self.threads.create_index([("user_id", 1), ("status", 1)])
        self.threads.create_index([("user_id", 1), ("last_updated", -1)])
        self.threads.create_index([("user_id", 1), ("participants", 1)])
        
        self.thread_updates.create_index([("thread_id", 1), ("date", -1)])
        self.thread_updates.create_index([("user_id", 1), ("entry_id", 1)])
        
        self.thread_relationships.create_index([("thread_id", 1)])
        self.thread_relationships.create_index([("related_thread_id", 1)])
        
        self.thread_vectors.create_index([("thread_id", 1)])
        if self.use_atlas_vector_search:
            self.thread_vectors.create_index([("vector", "vector")])
    
    def _setup_vector_search(self):
        """Set up vector search indexes if using MongoDB Atlas."""
        if self.use_atlas_vector_search:
            try:
                # Create vector search index for journal entries
                self.db.command({
                    "createIndexes": "journal_entries",
                    "indexes": [{
                        "name": "vector_index",
                        "key": {"embedding": "vector"},
                        "vectorSearchOptions": {
                            "dimension": self.vector_dimensions,
                            "similarity": "cosine"
                        }
                    }]
                })
                
                # Create vector search indexes for other vector collections
                for collection_name in ["entity_vectors", "event_vectors", "narrative_vectors", "paragraph_vectors"]:
                    self.db.command({
                        "createIndexes": collection_name,
                        "indexes": [{
                            "name": "vector_index",
                            "key": {"vector": "vector"},
                            "vectorSearchOptions": {
                                "dimension": self.vector_dimensions,
                                "similarity": "cosine"
                            }
                        }]
                    })
                
                logger.info("Vector search indexes created successfully")
            except Exception as e:
                logger.warning(f"Could not create vector search indexes: {e}")
                logger.info("Falling back to in-memory vector search")
    
    def add_user(self, username: str, password: str) -> Optional[str]:
        """
        Add a new user with hashed password.
        Returns user_id if successful, None if username already exists.
        """
        # Check if username already exists
        if self.users.find_one({"username": username}):
            return None
        
        # Hash password
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        # Create user document
        user_id = str(uuid.uuid4())
        user_doc = {
            "user_id": user_id,
            "username": username,
            "password": hashed_password,
            "created_at": datetime.now(),
            "settings": {
                "theme": "light",
                "notifications": True,
                "reminder_time": "20:00",
                "narrative_analysis": True,
                "emotional_tracking": True,
                "relationship_tracking": True
            },
            "nlp_settings": {
                "model_preferences": {
                    "embedding_model": "all-mpnet-base-v2",
                    "emotion_detection": True,
                    "narrative_analysis": True
                }
            }
        }
        
        # Insert into database
        self.users.insert_one(user_doc)
        return user_id
    
    def verify_user(self, username: str, password: str) -> Optional[str]:
        """Verify user credentials and return user_id if valid."""
        user = self.users.find_one({"username": username})
        if not user:
            return None
        
        # Verify password
        if bcrypt.checkpw(password.encode('utf-8'), user["password"]):
            return user["user_id"]
        
        return None
    
    def get_user_settings(self, user_id: str) -> Dict:
        """Get user settings including NLP preferences."""
        user = self.users.find_one({"user_id": user_id})
        if not user:
            return {}
        
        settings = user.get("settings", {})
        settings.update({"nlp_settings": user.get("nlp_settings", {})})
        return settings
    
    def update_user_settings(self, user_id: str, settings: Dict) -> bool:
        """Update user settings."""
        result = self.users.update_one(
            {"user_id": user_id},
            {"$set": {"settings": settings}}
        )
        return result.modified_count > 0
    
    def add_journal_entry(self, user_id: str, text: str, 
                         date: datetime, 
                         processed_data: Dict,
                         embedding: List[float]) -> str:
        """
        Add a journal entry with rich NLP processing data and embeddings.
        
        Args:
            user_id: User identifier
            text: Raw journal entry text
            date: Date of the journal entry
            processed_data: Structured data from NLP processing
            embedding: Vector embedding of the entire entry
            
        Returns:
            entry_id: Unique identifier for the journal entry
        """
        entry_id = str(uuid.uuid4())
        entry_doc = {
            "entry_id": entry_id,
            "user_id": user_id,
            "text": text,
            "date": date,
            "nlp_data": processed_data,
            "embedding": embedding,
            "created_at": datetime.now(),
            "word_count": len(text.split()),
            "summary": processed_data.get("summary", {}),
            "processing_version": processed_data.get("meta", {}).get("processing_version", "1.0")
        }
        
        # Insert the main entry document
        self.journal_entries.insert_one(entry_doc)
        
        # Process and store narrative elements separately
        self._store_narrative_elements(user_id, entry_id, date, processed_data)
        
        # Process and store entities with their embeddings
        self._store_entities(user_id, entry_id, date, processed_data)
        
        # Process and store events with their embeddings
        self._store_events(user_id, entry_id, date, processed_data)
        
        # Store emotional arc
        self._store_emotional_arc(user_id, entry_id, date, processed_data)
        
        # Store concepts and themes
        self._store_concepts(user_id, entry_id, date, processed_data)
        
        # Store paragraph-level embeddings if available
        if "paragraphs" in processed_data:
            self._store_paragraph_embeddings(user_id, entry_id, processed_data["paragraphs"])
        
        # Update relationship graph
        self._update_relationship_graph(user_id, processed_data)
        
        # Process temporal sequences
        self._process_temporal_sequences(user_id, entry_id, date, processed_data)
        
        return entry_id
    
    def _store_narrative_elements(self, user_id: str, entry_id: str, 
                                date: datetime, processed_data: Dict):
        """Store narrative elements for advanced story understanding."""
        narrative_elements = processed_data.get("enhanced", {}).get("narrative_elements", {})
        if not narrative_elements:
            return
        
        # Extract narrative arc, goals, conflicts, etc.
        narrative_id = str(uuid.uuid4())
        narrative_doc = {
            "narrative_id": narrative_id,
            "user_id": user_id,
            "entry_id": entry_id,
            "date": date,
            "narrative_arc": narrative_elements.get("narrative_arc"),
            "goals": narrative_elements.get("goals", []),
            "conflicts": narrative_elements.get("conflicts", []),
            "learnings": narrative_elements.get("learnings", []),
            "reflections": narrative_elements.get("reflections", []),
            "emotional_arc": narrative_elements.get("emotional_arc", []),
            "narrative_roles": narrative_elements.get("narrative_roles", {})
        }
        
        # Add any causal relationships
        if "causal_relationships" in narrative_elements:
            narrative_doc["causal_relationships"] = narrative_elements["causal_relationships"]
        
        # Store narrative document
        self.narratives.insert_one(narrative_doc)
        
        # If an embedding for narrative elements exists, store it
        if "narrative_embedding" in processed_data:
            self.narrative_vectors.insert_one({
                "narrative_id": narrative_id,
                "user_id": user_id,
                "entry_id": entry_id,
                "date": date,
                "vector": processed_data["narrative_embedding"],
                "narrative_type": narrative_elements.get("narrative_arc", "GENERAL"),
                "created_at": datetime.now()
            })
    
    def _store_entities(self, user_id: str, entry_id: str, 
                       date: datetime, processed_data: Dict):
        """Store enriched entity information with vector embeddings."""
        entities = processed_data.get("core", {}).get("entities", [])
        if not entities:
            return
        
        entity_operations = []
        entity_vector_operations = []
        
        for entity in entities:
            entity_id = hashlib.md5(entity["text"].lower().encode()).hexdigest()
            
            # Check if this entity already exists
            existing = self.entities.find_one({
                "user_id": user_id,
                "entity_id": entity_id
            })
            
            entity_metadata = {
                "type": entity.get("type"),
                "subtype": entity.get("subtype"),
                "sentiment": entity.get("sentiment"),
                "metadata": entity.get("metadata", {})
            }
            
            if existing:
                # Update existing entity
                entity_operations.append(
                    UpdateOne(
                        {"entity_id": entity_id, "user_id": user_id},
                        {"$set": {
                            "last_mentioned": date,
                            "mention_count": existing.get("mention_count", 0) + 1
                        },
                        "$addToSet": {
                            "entry_ids": entry_id,
                            "contexts": {"text": entity.get("context", ""), "entry_id": entry_id}
                        },
                        "$push": {
                            "history": {
                                "date": date,
                                "entry_id": entry_id,
                                "sentiment": entity.get("sentiment", {}).get("label", "NEUTRAL")
                            }
                        }}
                    )
                )
            else:
                # Create new entity
                entity_operations.append(
                    InsertOne({
                        "entity_id": entity_id,
                        "user_id": user_id,
                        "text": entity["text"],
                        "type": entity.get("type"),
                        "subtype": entity.get("subtype"),
                        "first_mentioned": date,
                        "last_mentioned": date,
                        "mention_count": 1,
                        "entry_ids": [entry_id],
                        "contexts": [{"text": entity.get("context", ""), "entry_id": entry_id}],
                        "history": [{
                            "date": date,
                            "entry_id": entry_id,
                            "sentiment": entity.get("sentiment", {}).get("label", "NEUTRAL")
                        }],
                        "metadata": entity.get("metadata", {})
                    })
                )
            
            # Store entity vector if available
            if "entity_embeddings" in processed_data and entity["text"] in processed_data["entity_embeddings"]:
                entity_vector_operations.append({
                    "entity_id": entity_id,
                    "user_id": user_id,
                    "text": entity["text"],
                    "entry_id": entry_id,
                    "date": date,
                    "vector": processed_data["entity_embeddings"][entity["text"]],
                    "type": entity.get("type"),
                    "subtype": entity.get("subtype"),
                    "metadata": entity_metadata,
                    "created_at": datetime.now()
                })
        
        # Execute bulk operations
        if entity_operations:
            self.entities.bulk_write(entity_operations)
        
        # Store entity vectors
        if entity_vector_operations:
            self.entity_vectors.insert_many(entity_vector_operations)
    
    def _store_events(self, user_id: str, entry_id: str, 
                     date: datetime, processed_data: Dict):
        """Store enriched event information with thread tracking."""
        events = processed_data.get("core", {}).get("events", [])
        if not events:
            return
        
        event_docs = []
        event_vector_docs = []
        thread_updates = []
        new_threads = []
        thread_vectors = []
        
        for event in events:
            event_id = str(uuid.uuid4())
            thread_id = event.get("thread_id")
            
            # Create event document
            event_doc = {
                "event_id": event_id,
                "thread_id": thread_id,
                "user_id": user_id,
                "entry_id": entry_id,
                "date": date,
                "name": event.get("name", ""),  # Add name field
                "text": event.get("text", ""),
                "type": event.get("type", "OTHER"),
                "status": event.get("status", "UNKNOWN"),
                "completed": event.get("completed", False),
                "importance": event.get("importance", "MEDIUM"),
                "temporality": event.get("temporality", {}),
                "components": event.get("components", {}),
                "sentiment": event.get("sentiment", {}),
                "entities": [],
                "created_at": datetime.now(),
                "last_mentioned": date
            }
            
            # Extract entities mentioned in this event
            event_text = event.get("text", "").lower()
            for entity in processed_data.get("core", {}).get("entities", []):
                if entity["text"].lower() in event_text:
                    entity_id = hashlib.md5(entity["text"].lower().encode()).hexdigest()
                    event_doc["entities"].append({
                        "entity_id": entity_id,
                        "text": entity["text"],
                        "type": entity.get("type"),
                        "role": self._determine_entity_role(entity["text"], event)
                    })
            
            event_docs.append(event_doc)
            
            # Store event vector if available
            if "event_embeddings" in processed_data and event_id in processed_data["event_embeddings"]:
                event_vector_docs.append({
                    "event_id": event_id,
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "entry_id": entry_id,
                    "date": date,
                    "vector": processed_data["event_embeddings"][event_id],
                    "type": event.get("type", "OTHER"),
                    "status": event.get("status", "UNKNOWN"),
                    "created_at": datetime.now()
                })
            
            # Handle thread tracking
            thread_metadata = event.get("thread_metadata", {})
            
            # Check if this thread already exists
            existing_thread = self.threads.find_one({
                "user_id": user_id,
                "thread_id": thread_id
            })
            
            if existing_thread:
                # Update existing thread
                self.threads.update_one(
                    {"thread_id": thread_id},
                    {
                        "$set": {
                            "last_updated": date,
                            "status": thread_metadata.get("progression_stage", "UNKNOWN"),
                            "is_complete": thread_metadata.get("is_complete", False)
                        },
                        "$inc": {"update_count": 1},
                        "$addToSet": {
                            "participants": {"$each": event_doc["components"].get("participants", [])},
                            "locations": event_doc["components"].get("location"),
                            "entry_ids": entry_id
                        }
                    }
                )
            else:
                # Create new thread
                new_threads.append({
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "created_at": date,
                    "last_updated": date,
                    "status": thread_metadata.get("progression_stage", "UNKNOWN"),
                    "is_complete": thread_metadata.get("is_complete", False),
                    "type": event.get("type", "OTHER"),
                    "participants": event_doc["components"].get("participants", []),
                    "locations": [event_doc["components"].get("location")] if event_doc["components"].get("location") else [],
                    "entry_ids": [entry_id],
                    "update_count": 1,
                    "first_event_id": event_id
                })
            
            # Create thread update
            thread_updates.append({
                "thread_id": thread_id,
                "user_id": user_id,
                "entry_id": entry_id,
                "event_id": event_id,
                "date": date,
                "update_type": "continuation" if thread_metadata.get("previous_mentions") else "creation",
                "progression_stage": thread_metadata.get("progression_stage", "UNKNOWN"),
                "participants_snapshot": thread_metadata.get("participants_history", []),
                "location_changes": thread_metadata.get("location_changes", []),
                "temporal_markers": thread_metadata.get("temporal_progression", []),
                "created_at": datetime.now()
            })
            
            # Store thread vector
            if "event_embeddings" in processed_data and event_id in processed_data["event_embeddings"]:
                thread_vectors.append({
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "vector": processed_data["event_embeddings"][event_id],
                    "date": date,
                    "created_at": datetime.now()
                })
        
        # Bulk insert all documents
        if event_docs:
            self.events.insert_many(event_docs)
        
        if event_vector_docs:
            self.event_vectors.insert_many(event_vector_docs)
        
        if new_threads:
            self.threads.insert_many(new_threads)
        
        if thread_updates:
            self.thread_updates.insert_many(thread_updates)
        
        if thread_vectors:
            self.thread_vectors.insert_many(thread_vectors)
    
    def _determine_entity_role(self, entity_text: str, event: Dict) -> str:
        """Determine the role of an entity in an event."""
        components = event.get("components", {})
        
        # Check if entity is a participant
        if entity_text in components.get("participants", []):
            return "PARTICIPANT"
        
        # Check if entity is a location
        if entity_text == components.get("location", ""):
            return "LOCATION"
        
        # Default role
        return "MENTIONED"
    
    def _store_emotional_arc(self, user_id: str, entry_id: str, 
                            date: datetime, processed_data: Dict):
        """Store emotional arc information for tracking emotional narratives."""
        emotions = processed_data.get("enhanced", {}).get("emotions", {})
        if not emotions:
            return
        
        # Create emotional arc document
        emotional_arc = {
            "user_id": user_id,
            "entry_id": entry_id,
            "date": date,
            "primary_emotion": emotions.get("primary_emotion"),
            "secondary_emotion": emotions.get("secondary_emotion"),
            "intensity": emotions.get("intensity", 0.0),
            "emotions_detected": emotions.get("emotions_detected", []),
            "emotional_phrases": emotions.get("emotional_phrases", []),
            "targets": emotions.get("targets", {}),
            "created_at": datetime.now()
        }
        
        self.emotional_arcs.insert_one(emotional_arc)
    
    def _store_concepts(self, user_id: str, entry_id: str, 
                       date: datetime, processed_data: Dict):
        """Store abstract concepts and themes for conceptual analysis."""
        topics = processed_data.get("core", {}).get("topics", [])
        recurring_themes = processed_data.get("contextual", {}).get("recurring_themes", [])
        
        concepts_to_update = []
        
        # Process topics as concepts
        for topic in topics:
            concept_id = hashlib.md5(topic.lower().encode()).hexdigest()
            
            # Check if concept already exists
            existing = self.concepts.find_one({
                "user_id": user_id,
                "concept_id": concept_id
            })
            
            if existing:
                # Update existing concept
                concepts_to_update.append(
                    UpdateOne(
                        {"concept_id": concept_id, "user_id": user_id},
                        {"$set": {
                            "last_mentioned": date,
                            "mention_count": existing.get("mention_count", 0) + 1
                        },
                        "$addToSet": {
                            "entry_ids": entry_id
                        },
                        "$push": {
                            "history": {
                                "date": date,
                                "entry_id": entry_id
                            }
                        }}
                    )
                )
            else:
                # Create new concept
                concepts_to_update.append(
                    InsertOne({
                        "concept_id": concept_id,
                        "user_id": user_id,
                        "concept_name": topic,
                        "type": "TOPIC",
                        "first_mentioned": date,
                        "last_mentioned": date,
                        "mention_count": 1,
                        "entry_ids": [entry_id],
                        "history": [{
                            "date": date,
                            "entry_id": entry_id
                        }],
                        "created_at": datetime.now()
                    })
                )
        
        # Process recurring themes
        for theme_group in recurring_themes:
            theme_type = theme_group.get("type", "")
            
            if theme_type == "RECURRING_TOPICS" and "topics" in theme_group:
                for topic_info in theme_group["topics"]:
                    topic = topic_info.get("text", "")
                    if topic:
                        concept_id = hashlib.md5(topic.lower().encode()).hexdigest()
                        
                        # Check if concept already exists
                        existing = self.concepts.find_one({
                            "user_id": user_id,
                            "concept_id": concept_id
                        })
                        
                        if existing:
                            # Update existing concept
                            concepts_to_update.append(
                                UpdateOne(
                                    {"concept_id": concept_id, "user_id": user_id},
                                    {"$set": {
                                        "last_mentioned": date,
                                        "mention_count": topic_info.get("count", 1),
                                        "recurring": True
                                    },
                                    "$addToSet": {
                                        "entry_ids": entry_id
                                    }}
                                )
                            )
        
        # Execute bulk operations
        if concepts_to_update:
            self.concepts.bulk_write(concepts_to_update)
    
    def _store_paragraph_embeddings(self, user_id: str, entry_id: str, paragraphs: List[Dict]):
        """Store paragraph-level embeddings for fine-grained semantic search."""
        if not paragraphs:
            return
        
        paragraph_vectors = []
        
        for i, paragraph in enumerate(paragraphs):
            if "embedding" in paragraph:
                paragraph_vectors.append({
                    "user_id": user_id,
                    "entry_id": entry_id,
                    "paragraph_id": f"{entry_id}_p{i}",
                    "paragraph_index": i,
                    "text": paragraph.get("text", ""),
                    "vector": paragraph["embedding"],
                    "created_at": datetime.now()
                })
        
        if paragraph_vectors:
            self.paragraph_vectors.insert_many(paragraph_vectors)
    
    def _update_relationship_graph(self, user_id: str, processed_data: Dict):
        """Update the relationship graph based on detected relationships."""
        relationships = processed_data.get("enhanced", {}).get("relationships", [])
        if not relationships:
            return
        
        relationship_docs = []
        
        for rel in relationships:
            entity1 = rel.get("entity1", "")
            entity2 = rel.get("entity2", "")
            rel_type = rel.get("type", "ASSOCIATED")
            
            if not entity1 or not entity2:
                continue
            
            # Create entity IDs
            entity1_id = hashlib.md5(entity1.lower().encode()).hexdigest()
            entity2_id = hashlib.md5(entity2.lower().encode()).hexdigest()
            
            # Check if relationship already exists
            existing = self.entity_relationships.find_one({
                "user_id": user_id,
                "source_id": entity1_id,
                "target_id": entity2_id
            })
            
            if existing:
                # Update relationship strength and context
                self.entity_relationships.update_one(
                    {
                        "user_id": user_id,
                        "source_id": entity1_id,
                        "target_id": entity2_id
                    },
                    {
                        "$set": {
                            "relationship_type": rel_type,
                            "last_updated": datetime.now()
                        },
                        "$inc": {
                            "strength": 1
                        },
                        "$push": {
                            "contexts": rel.get("context", "")
                        }
                    }
                )
            else:
                # Create new relationship
                relationship_docs.append({
                    "user_id": user_id,
                    "source_id": entity1_id,
                    "target_id": entity2_id,
                    "source_text": entity1,
                    "target_text": entity2,
                    "relationship_type": rel_type,
                    "strength": 1,
                    "contexts": [rel.get("context", "")],
                    "created_at": datetime.now(),
                    "last_updated": datetime.now()
                })
        
        if relationship_docs:
            self.entity_relationships.insert_many(relationship_docs)
    
    def _process_temporal_sequences(self, user_id: str, entry_id: str, 
                                   date: datetime, processed_data: Dict):
        """Process and store temporal sequences and story arcs."""
        # Extract temporal information
        temporal_info = processed_data.get("enhanced", {}).get("temporal_information", {})
        events = processed_data.get("core", {}).get("events", [])
        
        if not temporal_info or not events:
            return
        
        # Check for ordered events that might form a sequence
        ordered_events = []
        for event in events:
            if "temporality" in event:
                ordered_events.append({
                    "event_id": event.get("event_id", str(uuid.uuid4())),
                    "text": event.get("text", ""),
                    "type": event.get("type", "OTHER"),
                    "status": event.get("status", "UNKNOWN"),
                    "temporality": event.get("temporality", {})
                })
        
        # Sort events by their temporal references if possible
        if ordered_events:
            # Try to identify a sequence type
            sequence_type = self._identify_sequence_type(ordered_events)
            
            if sequence_type:
                # Create a new temporal sequence
                sequence_id = str(uuid.uuid4())
                sequence_doc = {
                    "sequence_id": sequence_id,
                    "user_id": user_id,
                    "entry_id": entry_id,
                    "sequence_type": sequence_type,
                    "events": ordered_events,
                    "start_date": date,
                    "created_at": datetime.now()
                }
                
                self.temporal_sequences.insert_one(sequence_doc)
                
                # Check if this sequence continues a previous one
                self._link_temporal_sequences(user_id, sequence_id, sequence_type, ordered_events)
    
    def _identify_sequence_type(self, events: List[Dict]) -> Optional[str]:
        """Identify the type of temporal sequence based on event patterns."""
        # Count event types
        type_counts = {}
        for event in events:
            event_type = event.get("type", "OTHER")  # Add default
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
        
        # Find the dominant event type
        if type_counts:
            dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]
            return f"SEQUENCE_{dominant_type}"
        
        return "GENERIC_SEQUENCE"
    
    def _link_temporal_sequences(self, user_id: str, sequence_id: str, 
                               sequence_type: str, events: List[Dict]):
        """Link related temporal sequences to track ongoing narratives."""
        # Find recent sequences of the same type
        cutoff_date = datetime.now() - timedelta(days=30)  # Look back 30 days
        
        recent_sequences = list(self.temporal_sequences.find({
            "user_id": user_id,
            "sequence_type": sequence_type,
            "start_date": {"$gte": cutoff_date},
            "sequence_id": {"$ne": sequence_id}  # Exclude the current sequence
        }).sort("start_date", -1).limit(5))
        
        if not recent_sequences:
            return
        
        # Check for potentially related sequences
        for recent in recent_sequences:
            recent_events = recent.get("events", [])
            
            # Simple heuristic: check for shared event texts or similar event types
            shared_text = any(re.get("text") == e.get("text") for re in recent_events for e in events)
            similar_types = any(re.get("type") == e.get("type") for re in recent_events for e in events)
            
            if shared_text or similar_types:
                # Link sequences
                self.temporal_sequences.update_one(
                    {"sequence_id": sequence_id},
                    {"$set": {"related_sequence_id": recent["sequence_id"]}}
                )
                
                self.temporal_sequences.update_one(
                    {"sequence_id": recent["sequence_id"]},
                    {"$set": {"continued_by_sequence_id": sequence_id}}
                )
                
                # Only link to one sequence
                break
    
    def get_journal_entries(self, user_id: str, limit: int = 50, 
                           skip: int = 0, start_date: datetime = None, 
                           end_date: datetime = None, 
                           include_nlp_data: bool = False) -> List[Dict]:
        """Get journal entries for a user with optional date filtering."""
        query = {"user_id": user_id}
        
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query["$gte"] = start_date
            if end_date:
                date_query["$lte"] = end_date
            query["date"] = date_query
        
        # Define projection based on whether to include NLP data
        projection = None if include_nlp_data else {"nlp_data": 0, "embedding": 0}
        
        entries = list(self.journal_entries.find(
            query,
            projection=projection,
            sort=[("date", pymongo.DESCENDING)],
            skip=skip,
            limit=limit
        ))
        
        # Convert MongoDB objects to serializable format
        for entry in entries:
            entry["_id"] = str(entry["_id"])
        
        return entries
    
    def get_journal_entry(self, entry_id: str, include_nlp_data: bool = False) -> Optional[Dict]:
        """Get a specific journal entry by ID with option to include NLP data."""
        projection = None if include_nlp_data else {"nlp_data": 0, "embedding": 0}
        
        entry = self.journal_entries.find_one(
            {"entry_id": entry_id},
            projection=projection
        )
        
        if entry:
            entry["_id"] = str(entry["_id"])
            
            # Add related information
            entry["narratives"] = self.get_narratives_for_entry(entry["user_id"], entry_id)
            entry["emotional_arc"] = self.get_emotional_arc_for_entry(entry["user_id"], entry_id)
        
        return entry
    
    def get_narratives_for_entry(self, user_id: str, entry_id: str) -> List[Dict]:
        """Get narrative information for a specific entry."""
        narratives = list(self.narratives.find(
            {"user_id": user_id, "entry_id": entry_id}
        ))
        
        for narrative in narratives:
            narrative["_id"] = str(narrative["_id"])
        
        return narratives
    
    def get_emotional_arc_for_entry(self, user_id: str, entry_id: str) -> Optional[Dict]:
        """Get emotional arc for a specific entry."""
        emotional_arc = self.emotional_arcs.find_one(
            {"user_id": user_id, "entry_id": entry_id}
        )
        
        if emotional_arc:
            emotional_arc["_id"] = str(emotional_arc["_id"])
        
        return emotional_arc
    
    def add_event(self, user_id: str, event_data: Dict) -> str:
        """Add a new event with rich metadata."""
        event_id = event_data.get("event_id", str(uuid.uuid4()))
        event_data["event_id"] = event_id
        event_data["user_id"] = user_id
        
        # Ensure created_at field exists
        if "created_at" not in event_data:
            event_data["created_at"] = datetime.now()
        
        # Extract entities if they exist
        if "entities" in event_data:
            for entity in event_data["entities"]:
                self._store_entity_from_event(user_id, entity, event_id)
        
        # Store event vector if available
        if "embedding" in event_data:
            self.event_vectors.insert_one({
                "event_id": event_id,
                "user_id": user_id,
                "vector": event_data["embedding"],
                "type": event_data.get("type", "OTHER"),
                "status": event_data.get("status", "UNKNOWN"),
                "created_at": datetime.now()
            })
            
            # Remove embedding from main document to avoid duplication
            del event_data["embedding"]
        
        self.events.insert_one(event_data)
        return event_id
    
    def _store_entity_from_event(self, user_id: str, entity: Dict, event_id: str):
        """Store entity information extracted from manually added events."""
        entity_id = hashlib.md5(entity["text"].lower().encode()).hexdigest()
        
        # Look for existing entity
        existing = self.entities.find_one({
            "user_id": user_id,
            "entity_id": entity_id
        })
        
        if existing:
            # Update existing entity
            self.entities.update_one(
                {"entity_id": entity_id, "user_id": user_id},
                {
                    "$addToSet": {"event_ids": event_id},
                    "$set": {"last_updated": datetime.now()}
                }
            )
        else:
            # Create new entity
            entity_doc = {
                "entity_id": entity_id,
                "user_id": user_id,
                "text": entity["text"],
                "type": entity.get("type", "UNKNOWN"),
                "subtype": entity.get("subtype"),
                "event_ids": [event_id],
                "created_at": datetime.now(),
                "last_updated": datetime.now()
            }
            self.entities.insert_one(entity_doc)
    
    def update_event(self, event_id: str, updates: Dict) -> bool:
        """Update an existing event."""
        # Update last_updated timestamp
        updates["last_updated"] = datetime.now()
        
        # Handle embedding updates separately
        if "embedding" in updates:
            self._update_event_embedding(event_id, updates["embedding"])
            del updates["embedding"]
        
        # Remove _id field if present to avoid MongoDB error
        if "_id" in updates:
            del updates["_id"]
        
        result = self.events.update_one(
            {"event_id": event_id},
            {"$set": updates}
        )
        
        return result.modified_count > 0
    
    def _update_event_embedding(self, event_id: str, embedding: List[float]):
        """Update or create event vector embedding."""
        event = self.events.find_one({"event_id": event_id})
        if not event:
            return
        
        # Check if vector exists
        existing_vector = self.event_vectors.find_one({"event_id": event_id})
        
        if existing_vector:
            # Update existing vector
            self.event_vectors.update_one(
                {"event_id": event_id},
                {"$set": {
                    "vector": embedding,
                    "last_updated": datetime.now()
                }}
            )
        else:
            # Create new vector document
            self.event_vectors.insert_one({
                "event_id": event_id,
                "user_id": event["user_id"],
                "vector": embedding,
                "type": event.get("type", "OTHER"),
                "status": event.get("status", "UNKNOWN"),
                "created_at": datetime.now()
            })
    
    def get_events(self, user_id: str, status: str = None, 
                  event_type: str = None, limit: int = 100,
                  start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Get events for a user with advanced filtering."""
        query = {"user_id": user_id}
        
        if status:
            query["status"] = status
            
        if event_type:
            query["type"] = event_type
        
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query["$gte"] = start_date
            if end_date:
                date_query["$lte"] = end_date
            query["last_mentioned"] = date_query
            
        events = list(self.events.find(
            query,
            sort=[("last_mentioned", pymongo.DESCENDING)],
            limit=limit
        ))
        
        for event in events:
            event["_id"] = str(event["_id"])
        
        return events
    
    def get_recent_events(self, user_id: str, days: int = 14) -> List[Dict]:
        """Get events from the past N days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        events = list(self.events.find(
            {
                "user_id": user_id,
                "last_mentioned": {"$gte": cutoff_date}
            },
            sort=[("last_mentioned", pymongo.DESCENDING)]
        ))
        
        for event in events:
            event["_id"] = str(event["_id"])
        
        return events
    
    def get_event(self, event_id: str) -> Optional[Dict]:
        """Get a specific event by ID."""
        event = self.events.find_one({"event_id": event_id})
        
        if event:
            event["_id"] = str(event["_id"])
            
            # Add related information
            user_id = event["user_id"]
            event["entities"] = self.get_entities_for_event(event_id)
            event["related_events"] = self.find_related_events(user_id, event_id)
        
        return event
    
    def get_entities_for_event(self, event_id: str) -> List[Dict]:
        """Get entities associated with a specific event."""
        event = self.events.find_one({"event_id": event_id})
        if not event:
            return []
        
        entities = []
        for entity_ref in event.get("entities", []):
            entity = self.entities.find_one({
                "entity_id": entity_ref.get("entity_id"),
                "user_id": event["user_id"]
            })
            
            if entity:
                entity["_id"] = str(entity["_id"])
                entity["role"] = entity_ref.get("role", "MENTIONED")
                entities.append(entity)
        
        return entities
    
    def find_related_events(self, user_id: str, event_id: str, limit: int = 5) -> List[Dict]:
        """Find events related to a specific event."""
        event = self.events.find_one({"event_id": event_id})
        if not event:
            return []
        
        # Get event vector
        event_vector = self.event_vectors.find_one({"event_id": event_id})
        
        related_events = []
        
        if event_vector and "vector" in event_vector:
            # Use vector similarity to find related events
            if self.use_atlas_vector_search:
                # Use Atlas vector search if available
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": "vector_index",
                            "queryVector": event_vector["vector"],
                            "path": "vector",
                            "numCandidates": 100,
                            "limit": limit + 1  # +1 to account for the event itself
                        }
                    },
                    {
                        "$match": {
                            "user_id": user_id,
                            "event_id": {"$ne": event_id}  # Exclude the current event
                        }
                    },
                    {"$limit": limit}
                ]
                
                vector_results = list(self.event_vectors.aggregate(pipeline))
                
                # Fetch full event documents
                for result in vector_results:
                    related_event = self.events.find_one({"event_id": result["event_id"]})
                    if related_event:
                        related_event["_id"] = str(related_event["_id"])
                        related_event["similarity"] = result.get("score", 0)
                        related_events.append(related_event)
            else:
                # Fallback to in-memory vector search
                all_vectors = list(self.event_vectors.find({
                    "user_id": user_id,
                    "event_id": {"$ne": event_id}
                }))
                
                if all_vectors:
                    # Calculate similarities
                    similarities = []
                    query_vector = np.array(event_vector["vector"])
                    
                    for vec in all_vectors:
                        if "vector" in vec:
                            event_embedding = np.array(vec["vector"])
                            similarity = np.dot(query_vector, event_embedding) / (
                                np.linalg.norm(query_vector) * np.linalg.norm(event_embedding)
                            )
                            
                            similarities.append((vec["event_id"], similarity))
                    
                    # Sort by similarity
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    # Get top related events
                    for event_id, similarity in similarities[:limit]:
                        related_event = self.events.find_one({"event_id": event_id})
                        if related_event:
                            related_event["_id"] = str(related_event["_id"])
                            related_event["similarity"] = similarity
                            related_events.append(related_event)
        
        # If no vector similarity or not enough results, fallback to other methods
        if len(related_events) < limit:
            # Try finding events with the same entities
            event_entities = set(e.get("entity_id") for e in event.get("entities", []))
            
            if event_entities:
                entity_based_events = list(self.events.find({
                    "user_id": user_id,
                    "event_id": {"$ne": event_id},
                    "entities.entity_id": {"$in": list(event_entities)}
                }).limit(limit - len(related_events)))
                
                for e in entity_based_events:
                    if not any(r["event_id"] == e["event_id"] for r in related_events):
                        e["_id"] = str(e["_id"])
                        e["relation_type"] = "SHARED_ENTITIES"
                        related_events.append(e)
            
            # If still not enough, try events of the same type
            if len(related_events) < limit and "type" in event:
                type_based_events = list(self.events.find({
                    "user_id": user_id,
                    "event_id": {"$ne": event_id},
                    "type": event["type"]
                }).limit(limit - len(related_events)))
                
                for e in type_based_events:
                    if not any(r["event_id"] == e["event_id"] for r in related_events):
                        e["_id"] = str(e["_id"])
                        e["relation_type"] = "SAME_TYPE"
                        related_events.append(e)
        
        return related_events
    
    def text_search(self, user_id: str, query: str, limit: int = 10) -> List[Dict]:
        """
        Perform text search across entries.
        This is a simple implementation using MongoDB's text index.
        """
        results = list(self.journal_entries.find(
            {
                "user_id": user_id,
                "$text": {"$search": query}
            },
            {
                "score": {"$meta": "textScore"},
                "entry_id": 1,
                "text": 1,
                "date": 1,
                "summary": 1
            },
            sort=[("score", {"$meta": "textScore"})],
            limit=limit
        ))
        
        for result in results:
            result["_id"] = str(result["_id"])
            result["relevance"] = float(result.pop("score"))
            
        return results
    
    def semantic_search(self, user_id: str, query: str, embedding: List[float], 
                       limit: int = 10, search_level: str = "entry", 
                       threshold: float = 0.6) -> List[Dict]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            user_id: User identifier
            query: Search query text
            embedding: Vector embedding of the query
            limit: Maximum number of results to return
            search_level: Level at which to search ('entry', 'paragraph', 'entity', 'event', 'narrative')
            threshold: Minimum similarity threshold
            
        Returns:
            List of matching documents with relevance scores
        """
        if self.use_atlas_vector_search:
            # Use MongoDB Atlas vector search capabilities
            return self._atlas_vector_search(user_id, embedding, limit, search_level, threshold)
        else:
            # Fallback to in-memory vector search
            return self._in_memory_vector_search(user_id, embedding, limit, search_level, threshold)
    
    def _atlas_vector_search(self, user_id: str, embedding: List[float], 
                           limit: int, search_level: str, threshold: float) -> List[Dict]:
        """Perform vector search using MongoDB Atlas vector search."""
        collection_map = {
            "entry": self.journal_entries,
            "paragraph": self.paragraph_vectors,
            "entity": self.entity_vectors,
            "event": self.event_vectors,
            "narrative": self.narrative_vectors
        }
        
        vector_path = "embedding" if search_level == "entry" else "vector"
        collection = collection_map.get(search_level, self.journal_entries)
        
        # Create aggregation pipeline for vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": embedding,
                    "path": vector_path,
                    "numCandidates": 100,
                    "limit": limit
                }
            },
            {
                "$match": {
                    "user_id": user_id
                }
            },
            {"$limit": limit}
        ]
        
        try:
            results = list(collection.aggregate(pipeline))
            
            # Process results based on search level
            for result in results:
                result["_id"] = str(result["_id"])
                result["relevance"] = result.pop("score") if "score" in result else 0.0
                
                # Add entry data for non-entry searches
                if search_level != "entry":
                    entry_id = result.get("entry_id")
                    if entry_id:
                        entry = self.journal_entries.find_one(
                            {"entry_id": entry_id},
                            {"text": 1, "date": 1}
                        )
                        if entry:
                            result["entry"] = {
                                "entry_id": entry_id,
                                "text": entry.get("text", ""),
                                "date": entry.get("date")
                            }
            
            return results
        except Exception as e:
            logger.error(f"Atlas vector search error: {e}")
            # Fallback to in-memory search
            return self._in_memory_vector_search(user_id, embedding, limit, search_level, threshold)
    
    def _in_memory_vector_search(self, user_id: str, embedding: List[float], 
                               limit: int, search_level: str, threshold: float) -> List[Dict]:
        """Perform vector search in-memory when Atlas vector search is not available."""
        collection_map = {
            "entry": (self.journal_entries, "embedding"),
            "paragraph": (self.paragraph_vectors, "vector"),
            "entity": (self.entity_vectors, "vector"),
            "event": (self.event_vectors, "vector"),
            "narrative": (self.narrative_vectors, "vector")
        }
        
        collection, vector_field = collection_map.get(search_level, (self.journal_entries, "embedding"))
        
        # Get all documents for the user
        documents = list(collection.find(
            {"user_id": user_id},
            {vector_field: 1, "entry_id": 1, "text": 1, "date": 1}
        ))
        
        query_embedding = np.array(embedding)
        results = []
        
        for doc in documents:
            if vector_field in doc:
                # Calculate cosine similarity
                doc_embedding = np.array(doc[vector_field])
                
                try:
                    similarity = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
                    
                    if similarity > threshold:
                        doc["_id"] = str(doc["_id"])
                        doc["relevance"] = float(similarity)
                        
                        # Add entry data for non-entry searches
                        if search_level != "entry" and "entry_id" in doc:
                            entry = self.journal_entries.find_one(
                                {"entry_id": doc["entry_id"]},
                                {"text": 1, "date": 1}
                            )
                            if entry:
                                doc["entry"] = {
                                    "entry_id": doc["entry_id"],
                                    "text": entry.get("text", ""),
                                    "date": entry.get("date")
                                }
                        
                        results.append(doc)
                except Exception as e:
                    logger.error(f"Error calculating similarity: {e}")
                    doc["relevance"] = 0.0
        
        # Sort by relevance (descending) and return top k
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:limit]
    
    def multi_facet_search(self, user_id: str, query: Dict, 
                          embedding: Optional[List[float]] = None, 
                          limit: int = 10) -> List[Dict]:
        """
        Perform advanced search with multiple facets (semantic, text, date, emotions, topics).
        
        Args:
            user_id: User identifier
            query: Complex query object with different facets:
                - text: Text to search for
                - dates: Date range
                - emotions: List of emotions to filter by
                - topics: List of topics to filter by
                - entities: List of entities to filter by
            embedding: Optional vector embedding of the query text
            limit: Maximum number of results to return
            
        Returns:
            List of matching entries with relevance scores
        """
        # Build base query
        base_query = {"user_id": user_id}
        score_fields = []
        
        # Add date constraints if provided
        if "dates" in query:
            date_query = {}
            if "start" in query["dates"]:
                date_query["$gte"] = query["dates"]["start"]
            if "end" in query["dates"]:
                date_query["$lte"] = query["dates"]["end"]
            if date_query:
                base_query["date"] = date_query
        
        # Add text search if provided
        if "text" in query and query["text"]:
            base_query["$text"] = {"$search": query["text"]}
            score_fields.append({"score": {"$meta": "textScore"}})
        
        # Add emotion filter if provided
        if "emotions" in query and query["emotions"]:
            # Find entries with these emotions
            emotion_entries = self.emotional_arcs.find(
                {
                    "user_id": user_id,
                    "primary_emotion": {"$in": query["emotions"]}
                },
                {"entry_id": 1}
            )
            emotion_entry_ids = [e["entry_id"] for e in emotion_entries]
            if emotion_entry_ids:
                base_query["entry_id"] = {"$in": emotion_entry_ids}
            else:
                # No entries match emotions, return empty result
                return []
        
        # Add topic filter if provided
        if "topics" in query and query["topics"]:
            # Find entries with these topics
            entries_with_topics = []
            for topic in query["topics"]:
                concept = self.concepts.find_one({
                    "user_id": user_id,
                    "concept_name": topic
                })
                if concept and "entry_ids" in concept:
                    entries_with_topics.extend(concept["entry_ids"])
            
            if entries_with_topics:
                if "entry_id" in base_query:
                    # Intersection with existing entry_id filter
                    base_query["entry_id"] = {
                        "$in": list(set(base_query["entry_id"]["$in"]) & set(entries_with_topics))
                    }
                else:
                    base_query["entry_id"] = {"$in": entries_with_topics}
            else:
                # No entries match topics, return empty result
                return []
        
        # Add entity filter if provided
        if "entities" in query and query["entities"]:
            # Find entries mentioning these entities
            entries_with_entities = []
            for entity_name in query["entities"]:
                entity_id = hashlib.md5(entity_name.lower().encode()).hexdigest()
                entity = self.entities.find_one({
                    "user_id": user_id,
                    "entity_id": entity_id
                })
                if entity and "entry_ids" in entity:
                    entries_with_entities.extend(entity["entry_ids"])
            
            if entries_with_entities:
                if "entry_id" in base_query:
                    # Intersection with existing entry_id filter
                    base_query["entry_id"] = {
                        "$in": list(set(base_query["entry_id"]["$in"]) & set(entries_with_entities))
                    }
                else:
                    base_query["entry_id"] = {"$in": entries_with_entities}
            else:
                # No entries match entities, return empty result
                return []
        
        # Execute base query to get candidate entries
        projection = {"text": 1, "date": 1, "embedding": 1, "summary": 1}
        for score_field in score_fields:
            projection.update(score_field)
        
        candidates = list(self.journal_entries.find(
            base_query,
            projection
        ))
        
        # If no candidates at this point, return empty list
        if not candidates:
            return []
        
        # If we have an embedding, compute semantic relevance
        if embedding is not None:
            query_embedding = np.array(embedding)
            
            for candidate in candidates:
                if "embedding" in candidate:
                    # Calculate semantic similarity
                    candidate_embedding = np.array(candidate["embedding"])
                    try:
                        semantic_score = np.dot(query_embedding, candidate_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
                        )
                        
                        # Store semantic score
                        candidate["semantic_score"] = float(semantic_score)
                    except Exception as e:
                        logger.error(f"Error calculating semantic similarity: {e}")
                        candidate["semantic_score"] = 0.0
                else:
                    candidate["semantic_score"] = 0.0
        
        # Calculate combined relevance scores
        for candidate in candidates:
            text_score = candidate.pop("score", 0) if "score" in candidate else 0
            semantic_score = candidate.pop("semantic_score", 0) if "semantic_score" in candidate else 0
            
            # Simple weighted combination of scores
            # Adjust weights based on presence of different query components
            weights = {
                "text": 1.0 if "text" in query and query["text"] else 0.0,
                "semantic": 1.0 if embedding is not None else 0.0
            }
            
            total_weight = sum(weights.values())
            if total_weight > 0:
                combined_score = (
                    (weights["text"] * text_score) +
                    (weights["semantic"] * semantic_score)
                ) / total_weight
            else:
                combined_score = 0.0
            
            candidate["relevance"] = float(combined_score)
            candidate["_id"] = str(candidate["_id"])
            
            # Remove embedding from result
            if "embedding" in candidate:
                del candidate["embedding"]
        
        # Sort by relevance and return top results
        candidates.sort(key=lambda x: x["relevance"], reverse=True)
        return candidates[:limit]
    
    def get_entities(self, user_id: str, entity_type: str = None, 
                    limit: int = 50, min_mentions: int = 1) -> List[Dict]:
        """Get entities for a user with optional type filtering."""
        query = {"user_id": user_id}
        
        if entity_type:
            query["type"] = entity_type
        
        if min_mentions > 1:
            query["mention_count"] = {"$gte": min_mentions}
        
        entities = list(self.entities.find(
            query,
            sort=[("mention_count", pymongo.DESCENDING)],
            limit=limit
        ))
        
        for entity in entities:
            entity["_id"] = str(entity["_id"])
        
        return entities
    
    def get_entity(self, user_id: str, entity_id: str) -> Optional[Dict]:
        """Get a specific entity by ID."""
        entity = self.entities.find_one({
            "user_id": user_id,
            "entity_id": entity_id
        })
        
        if entity:
            entity["_id"] = str(entity["_id"])
            
            # Add related entries
            entry_ids = entity.get("entry_ids", [])
            if entry_ids:
                related_entries = list(self.journal_entries.find(
                    {"entry_id": {"$in": entry_ids}},
                    {"text": 1, "date": 1, "summary": 1}
                ).limit(10))
                
                for entry in related_entries:
                    entry["_id"] = str(entry["_id"])
                
                entity["related_entries"] = related_entries
            
            # Add relationships
            relationships = list(self.entity_relationships.find({
                "user_id": user_id,
                "$or": [
                    {"source_id": entity_id},
                    {"target_id": entity_id}
                ]
            }))
            
            for rel in relationships:
                rel["_id"] = str(rel["_id"])
            
            entity["relationships"] = relationships
        
        return entity
    
    def get_entity_by_name(self, user_id: str, entity_name: str) -> Optional[Dict]:
        """Get an entity by its name."""
        entity_id = hashlib.md5(entity_name.lower().encode()).hexdigest()
        return self.get_entity(user_id, entity_id)
    
    def get_entity_relationships(self, user_id: str, min_strength: int = 1) -> Dict[str, Any]:
        """Get all entities and their relationships for network visualization."""
        entities = list(self.entities.find({
            "user_id": user_id,
            "mention_count": {"$gte": min_strength}
        }))
        
        relationships = list(self.entity_relationships.find({
            "user_id": user_id,
            "strength": {"$gte": min_strength}
        }))
        
        nodes = []
        links = []
        
        # Create nodes for entities
        for entity in entities:
            nodes.append({
                "id": entity["entity_id"],
                "name": entity["text"],
                "type": entity.get("type", "UNKNOWN"),
                "subtype": entity.get("subtype"),
                "mentions": entity.get("mention_count", 0)
            })
        
        # Create links between entities
        for rel in relationships:
            links.append({
                "source": rel["source_id"],
                "target": rel["target_id"],
                "type": rel["relationship_type"],
                "weight": rel["strength"]
            })
        
        return {
            "nodes": nodes,
            "links": links
        }
    
    def get_emotional_trends(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get emotional trends over time."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get emotional arcs from the specified period
        emotional_arcs = list(self.emotional_arcs.find({
            "user_id": user_id,
            "date": {"$gte": cutoff_date}
        }).sort("date", 1))
        
        # Prepare time series data
        emotions_by_date = {}
        emotion_counts = {}
        
        for arc in emotional_arcs:
            date_str = arc["date"].strftime("%Y-%m-%d")
            primary_emotion = arc.get("primary_emotion")
            
            if primary_emotion:
                # Track by date
                if date_str not in emotions_by_date:
                    emotions_by_date[date_str] = {}
                
                if primary_emotion not in emotions_by_date[date_str]:
                    emotions_by_date[date_str][primary_emotion] = 0
                
                emotions_by_date[date_str][primary_emotion] += 1
                
                # Track overall counts
                if primary_emotion not in emotion_counts:
                    emotion_counts[primary_emotion] = 0
                
                emotion_counts[primary_emotion] += 1
        
        # Create time series
        time_series = []
        for date_str, emotions in sorted(emotions_by_date.items()):
            time_series.append({
                "date": date_str,
                "emotions": emotions
            })
        
        # Create summary
        summary = []
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            summary.append({
                "emotion": emotion,
                "count": count
            })
        
        return {
            "time_series": time_series,
            "summary": summary
        }
    
    def get_narratives(self, user_id: str, 
                      narrative_type: str = None, 
                      limit: int = 10) -> List[Dict]:
        """Get narrative arcs and story structures."""
        query = {"user_id": user_id}
        
        if narrative_type:
            query["narrative_arc"] = narrative_type
        
        narratives = list(self.narratives.find(
            query,
            sort=[("date", pymongo.DESCENDING)],
            limit=limit
        ))
        
        for narrative in narratives:
            narrative["_id"] = str(narrative["_id"])
            
            # Add entry information
            entry = self.journal_entries.find_one(
                {"entry_id": narrative["entry_id"]},
                {"text": 1, "date": 1}
            )
            
            if entry:
                narrative["entry"] = {
                    "text": entry["text"],
                    "date": entry["date"]
                }
        
        return narratives
    
    def get_concepts(self, user_id: str, min_mentions: int = 2) -> List[Dict]:
        """Get abstract concepts and themes detected in the journal."""
        concepts = list(self.concepts.find({
            "user_id": user_id,
            "mention_count": {"$gte": min_mentions}
        }).sort("mention_count", -1))
        
        for concept in concepts:
            concept["_id"] = str(concept["_id"])
        
        return concepts
    
    def get_temporal_sequences(self, user_id: str, 
                              sequence_type: str = None) -> List[Dict]:
        """Get temporal sequences and event chains."""
        query = {"user_id": user_id}
        
        if sequence_type:
            query["sequence_type"] = sequence_type
        
        sequences = list(self.temporal_sequences.find(
            query,
            sort=[("start_date", pymongo.DESCENDING)]
        ))
        
        for sequence in sequences:
            sequence["_id"] = str(sequence["_id"])
            
            # Check for linked sequences
            if "related_sequence_id" in sequence:
                related = self.temporal_sequences.find_one(
                    {"sequence_id": sequence["related_sequence_id"]}
                )
                if related:
                    sequence["related_sequence"] = {
                        "sequence_id": related["sequence_id"],
                        "sequence_type": related["sequence_type"],
                        "start_date": related["start_date"],
                        "events": len(related.get("events", []))
                    }
        
        return sequences
    
    def generate_insights(self, user_id: str) -> Dict[str, Any]:
        """Generate insights from journal data."""
        insights = {
            "frequent_entities": [],
            "emotional_patterns": [],
            "recurring_topics": [],
            "narrative_patterns": [],
            "temporal_insights": []
        }
        
        # Get frequent entities
        top_entities = list(self.entities.find(
            {"user_id": user_id},
            sort=[("mention_count", pymongo.DESCENDING)],
            limit=10
        ))
        
        insights["frequent_entities"] = [
            {
                "text": entity["text"],
                "type": entity.get("type", "UNKNOWN"),
                "mentions": entity.get("mention_count", 0)
            } for entity in top_entities
        ]
        
        # Get emotional patterns
        emotion_counts = {}
        emotional_arcs = list(self.emotional_arcs.find({"user_id": user_id}))
        
        for arc in emotional_arcs:
            emotion = arc.get("primary_emotion")
            if emotion:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        insights["emotional_patterns"] = [
            {"emotion": emotion, "count": count}
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        
        # Get recurring topics
        top_concepts = list(self.concepts.find(
            {"user_id": user_id},
            sort=[("mention_count", pymongo.DESCENDING)],
            limit=5
        ))
        
        insights["recurring_topics"] = [
            {
                "name": concept["concept_name"],
                "mentions": concept.get("mention_count", 0)
            } for concept in top_concepts
        ]
        
        # Get narrative patterns
        narrative_types = {}
        narratives = list(self.narratives.find({"user_id": user_id}))
        
        for narrative in narratives:
            narrative_arc = narrative.get("narrative_arc")
            if narrative_arc:
                narrative_types[narrative_arc] = narrative_types.get(narrative_arc, 0) + 1
        
        insights["narrative_patterns"] = [
            {"type": arc_type, "count": count}
            for arc_type, count in sorted(narrative_types.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Get temporal insights (entry frequency by day of week)
        entries = list(self.journal_entries.find({"user_id": user_id}, {"date": 1}))
        day_counts = {}
        
        for entry in entries:
            if "date" in entry:
                day_of_week = entry["date"].strftime("%A")
                day_counts[day_of_week] = day_counts.get(day_of_week, 0) + 1
        
        insights["temporal_insights"] = [
            {"day": day, "entries": count}
            for day, count in day_counts.items()
        ]
        
        return insights
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all user data in a structured format."""
        # Get user information (excluding password)
        user = self.users.find_one({"user_id": user_id}, {"password": 0})
        if not user:
            return {"error": "User not found"}
        
        # Convert ObjectId to string
        user["_id"] = str(user["_id"])
        
        # Get all journal entries
        entries = list(self.journal_entries.find({"user_id": user_id}))
        for entry in entries:
            entry["_id"] = str(entry["_id"])
        
        # Get all entities
        entities = list(self.entities.find({"user_id": user_id}))
        for entity in entities:
            entity["_id"] = str(entity["_id"])
        
        # Get all events
        events = list(self.events.find({"user_id": user_id}))
        for event in events:
            event["_id"] = str(event["_id"])
        
        # Get all narratives
        narratives = list(self.narratives.find({"user_id": user_id}))
        for narrative in narratives:
            narrative["_id"] = str(narrative["_id"])
        
        # Get all emotion data
        emotions = list(self.emotional_arcs.find({"user_id": user_id}))
        for emotion in emotions:
            emotion["_id"] = str(emotion["_id"])
        
        # Get all concepts
        concepts = list(self.concepts.find({"user_id": user_id}))
        for concept in concepts:
            concept["_id"] = str(concept["_id"])
        
        # Get all relationships
        relationships = list(self.entity_relationships.find({"user_id": user_id}))
        for rel in relationships:
            rel["_id"] = str(rel["_id"])
        
        # Get all temporal sequences
        sequences = list(self.temporal_sequences.find({"user_id": user_id}))
        for seq in sequences:
            seq["_id"] = str(seq["_id"])
        
        # Combine all data
        export_data = {
            "user": user,
            "journal_entries": entries,
            "entities": entities,
            "events": events,
            "narratives": narratives,
            "emotions": emotions,
            "concepts": concepts,
            "relationships": relationships,
            "temporal_sequences": sequences,
            "export_date": datetime.now().isoformat()
        }
        
        return export_data

    def get_thread(self, user_id: str, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get a thread with all its updates and related information."""
        thread = self.threads.find_one({
            "user_id": user_id,
            "thread_id": thread_id
        })
        
        if not thread:
            return None
        
        # Get all updates for this thread
        updates = list(self.thread_updates.find(
            {"thread_id": thread_id}
        ).sort("date", 1))
        
        # Get all events in this thread
        events = list(self.events.find({
            "thread_id": thread_id
        }).sort("date", 1))
        
        # Get related threads
        related_threads = list(self.thread_relationships.find({
            "$or": [
                {"thread_id": thread_id},
                {"related_thread_id": thread_id}
            ]
        }))
        
        # Add all information to the thread
        thread["updates"] = updates
        thread["events"] = events
        thread["related_threads"] = related_threads
        
        return thread

    def get_active_threads(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get currently active threads for a user."""
        active_threads = list(self.threads.find({
            "user_id": user_id,
            "is_complete": False
        }).sort("last_updated", -1).limit(limit))
        
        # Add latest update for each thread
        for thread in active_threads:
            latest_update = self.thread_updates.find_one(
                {"thread_id": thread["thread_id"]},
                sort=[("date", -1)]
            )
            if latest_update:
                thread["latest_update"] = latest_update
        
        return active_threads

    def find_related_threads(self, user_id: str, thread_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find threads that are related to the given thread."""
        thread = self.threads.find_one({
            "user_id": user_id,
            "thread_id": thread_id
        })
        
        if not thread:
            return []
        
        # Get thread vector
        thread_vector = self.thread_vectors.find_one({"thread_id": thread_id})
        
        related_threads = []
        
        if thread_vector and "vector" in thread_vector:
            # Use vector similarity to find related threads
            if self.use_atlas_vector_search:
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": "vector_index",
                            "queryVector": thread_vector["vector"],
                            "path": "vector",
                            "numCandidates": 100,
                            "limit": limit + 1
                        }
                    },
                    {
                        "$match": {
                            "user_id": user_id,
                            "thread_id": {"$ne": thread_id}
                        }
                    },
                    {"$limit": limit}
                ]
                
                vector_results = list(self.thread_vectors.aggregate(pipeline))
                
                # Fetch full thread documents
                for result in vector_results:
                    related_thread = self.threads.find_one({"thread_id": result["thread_id"]})
                    if related_thread:
                        related_thread["similarity"] = result.get("score", 0)
                        related_threads.append(related_thread)
        
        # If not enough results from vector search, try other methods
        if len(related_threads) < limit:
            # Look for threads with shared participants
            shared_participant_threads = list(self.threads.find({
                "user_id": user_id,
                "thread_id": {"$ne": thread_id},
                "participants": {"$in": thread.get("participants", [])}
            }).limit(limit - len(related_threads)))
            
            for t in shared_participant_threads:
                if not any(rt["thread_id"] == t["thread_id"] for rt in related_threads):
                    t["relation_type"] = "SHARED_PARTICIPANTS"
                    related_threads.append(t)
        
        return related_threads

    def merge_threads(self, user_id: str, source_thread_id: str, target_thread_id: str) -> bool:
        """Merge two threads that are determined to be about the same topic."""
        source_thread = self.threads.find_one({
            "user_id": user_id,
            "thread_id": source_thread_id
        })
        
        target_thread = self.threads.find_one({
            "user_id": user_id,
            "thread_id": target_thread_id
        })
        
        if not source_thread or not target_thread:
            return False
        
        # Update all events to point to the target thread
        self.events.update_many(
            {"thread_id": source_thread_id},
            {"$set": {"thread_id": target_thread_id}}
        )
        
        # Update all vectors
        self.event_vectors.update_many(
            {"thread_id": source_thread_id},
            {"$set": {"thread_id": target_thread_id}}
        )
        
        # Move all updates to target thread
        self.thread_updates.update_many(
            {"thread_id": source_thread_id},
            {"$set": {"thread_id": target_thread_id}}
        )
        
        # Update target thread metadata
        self.threads.update_one(
            {"thread_id": target_thread_id},
            {
                "$addToSet": {
                    "participants": {"$each": source_thread.get("participants", [])},
                    "locations": {"$each": source_thread.get("locations", [])},
                    "entry_ids": {"$each": source_thread.get("entry_ids", [])}
                },
                "$inc": {"update_count": source_thread.get("update_count", 0)}
            }
        )
        
        # Delete the source thread
        self.threads.delete_one({"thread_id": source_thread_id})
        
        return True