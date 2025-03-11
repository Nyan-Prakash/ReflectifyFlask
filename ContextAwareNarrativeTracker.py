import uuid
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import networkx as nx
from collections import defaultdict

# Import the enhanced NLP engine and MongoDB client
from AdvancedNLPEngine import AdvancedNLPEngine
from MongoDBClient import MongoDBClient

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextAwareNarrativeTracker:
    """
    Enhanced event and narrative tracker with deep context awareness, 
    temporal reasoning, emotional understanding, and relationship tracking.
    """
    
    def __init__(self, nlp_engine: AdvancedNLPEngine, mongo_client: MongoDBClient):
        """
        Initialize with enhanced NLP engine and MongoDB client.
        
        Args:
            nlp_engine: Instance of the enhanced NLP engine for processing text
            mongo_client: Instance of the MongoDB client for data persistence
        """
        self.nlp_engine = nlp_engine
        self.mongo = mongo_client
        
        # Initialize narrative context tracking
        self.narrative_context = {
            "active_narratives": [],
            "recent_events": [],
            "recent_entities": set(),
            "ongoing_emotional_arcs": [],
            "temporal_sequences": [],
            "relationship_graph": nx.Graph()
        }
        
        # Time windows for context awareness
        self.short_term_window = 7  # 7 days for very recent context
        self.medium_term_window = 30  # 30 days for medium-term context
        self.long_term_window = 90  # 90 days for long-term context
    
    def process_journal_entry(self, user_id: str, entry_text: str, 
                             entry_id: str, entry_date: datetime) -> Dict[str, Any]:
        """
        Process a journal entry with comprehensive narrative understanding.
        
        Args:
            user_id: User identifier
            entry_text: Raw journal entry text
            entry_id: Entry identifier
            entry_date: Date of the journal entry
            
        Returns:
            Dictionary containing processed data and detected narratives
        """
        # First, process the entry with the NLP engine
        nlp_results = self.nlp_engine.process_entry(entry_text)
        
        # Generate entry embedding for semantic search
        entry_embedding = self.nlp_engine._generate_embedding(entry_text)
        
        # Update context with current entry information
        self._update_narrative_context(user_id, entry_id, nlp_results, entry_date)
        
        # Process events with enhanced context awareness
        processed_events = self._process_events(user_id, entry_id, nlp_results, entry_date)
        
        # Process entities with relationship tracking
        processed_entities = self._process_entities(user_id, entry_id, nlp_results, entry_date)
        
        # Process emotions and emotional arcs
        emotional_analysis = self._process_emotions(user_id, entry_id, nlp_results, entry_date)
        
        # Process narrative elements and story structure
        narrative_analysis = self._process_narrative_elements(user_id, entry_id, nlp_results, entry_date)
        
        # Detect and process temporal sequences
        temporal_analysis = self._process_temporal_sequences(user_id, entry_id, nlp_results, entry_date)
        
        # Generate embeddings for different components
        component_embeddings = self._generate_component_embeddings(nlp_results)
        
        # Store the processed entry with all enhanced data
        entry_data = {
            "entry_id": entry_id,
            "user_id": user_id,
            "text": entry_text,
            "date": entry_date,
            "processed_data": nlp_results,
            "embedding": entry_embedding
        }
        
        # Add component embeddings to NLP results for storage
        for key, embeddings in component_embeddings.items():
            entry_data["processed_data"][key] = embeddings
        
        # Store the journal entry with all the processed information
        self.mongo.add_journal_entry(
            user_id=user_id, 
            text=entry_text, 
            date=entry_date, 
            processed_data=nlp_results, 
            embedding=entry_embedding
        )
        
        # Prepare the response with all processed information
        response = {
            "entry_id": entry_id,
            "events": processed_events,
            "entities": processed_entities,
            "emotional_analysis": emotional_analysis,
            "narrative_analysis": narrative_analysis,
            "temporal_analysis": temporal_analysis,
            "detected_narratives": self._get_active_narratives(user_id, entry_date)
        }
        
        return response
    
    def _update_narrative_context(self, user_id: str, entry_id: str, 
                                nlp_results: Dict[str, Any], entry_date: datetime):
        """
        Update the narrative context with information from the current entry.
        
        Args:
            user_id: User identifier
            entry_id: Entry identifier
            nlp_results: Results from NLP processing
            entry_date: Date of the journal entry
        """
        # Get recent events for context
        short_term_date = entry_date - timedelta(days=self.short_term_window)
        medium_term_date = entry_date - timedelta(days=self.medium_term_window)
        
        # Get recent entries and events
        recent_entries = self.mongo.get_journal_entries(
            user_id, 
            start_date=medium_term_date, 
            end_date=entry_date
        )
        
        recent_events = self.mongo.get_events(
            user_id,
            start_date=medium_term_date,
            end_date=entry_date
        )
        
        # Get very recent events for higher contextual relevance
        very_recent_events = [e for e in recent_events 
                             if e.get("last_mentioned", datetime(1970, 1, 1)) >= short_term_date]
        
        # Get active narratives
        active_narratives = self.mongo.get_narratives(user_id, limit=10)
        
        # Update context container
        self.narrative_context["active_narratives"] = active_narratives
        self.narrative_context["recent_events"] = very_recent_events
        
        # Extract entities from NLP results
        current_entities = set()
        for entity in nlp_results.get("core", {}).get("entities", []):
            current_entities.add(entity["text"])
        
        # Extract recent entities from recent events
        recent_entities = set()
        for event in recent_events:
            for entity in event.get("entities", []):
                recent_entities.add(entity.get("text", ""))
        
        # Combine all entities for context
        self.narrative_context["recent_entities"] = recent_entities.union(current_entities)
        
        # Update emotional arcs context
        recent_emotions = self.mongo.emotional_arcs.find({
            "user_id": user_id,
            "date": {"$gte": medium_term_date, "$lt": entry_date}
        }).sort("date", -1).limit(10)
        
        self.narrative_context["ongoing_emotional_arcs"] = list(recent_emotions)
        
        # Get recent temporal sequences
        recent_sequences = self.mongo.temporal_sequences.find({
            "user_id": user_id,
            "start_date": {"$gte": medium_term_date, "$lt": entry_date}
        }).sort("start_date", -1).limit(5)
        
        self.narrative_context["temporal_sequences"] = list(recent_sequences)
        
        # Build relationship graph for context
        self._build_relationship_graph(user_id, medium_term_date, entry_date)
    
    def _build_relationship_graph(self, user_id: str, start_date: datetime, end_date: datetime):
        """
        Build a relationship graph from recent entries for contextual understanding.
        
        Args:
            user_id: User identifier
            start_date: Start date for relationship data
            end_date: End date for relationship data
        """
        # Create a new graph
        graph = nx.Graph()
        
        # Get entity relationships
        relationships = self.mongo.entity_relationships.find({
            "user_id": user_id,
            "last_updated": {"$gte": start_date, "$lt": end_date}
        })
        
        # Add nodes and edges to the graph
        for rel in relationships:
            source_id = rel.get("source_id")
            target_id = rel.get("target_id")
            rel_type = rel.get("relationship_type")
            strength = rel.get("strength", 1)
            
            if source_id and target_id:
                if not graph.has_node(source_id):
                    source_entity = self.mongo.entities.find_one({
                        "user_id": user_id,
                        "entity_id": source_id
                    })
                    if source_entity:
                        graph.add_node(
                            source_id,
                            type="entity",
                            text=source_entity.get("text", ""),
                            entity_type=source_entity.get("type", "UNKNOWN")
                        )
                
                if not graph.has_node(target_id):
                    target_entity = self.mongo.entities.find_one({
                        "user_id": user_id,
                        "entity_id": target_id
                    })
                    if target_entity:
                        graph.add_node(
                            target_id,
                            type="entity",
                            text=target_entity.get("text", ""),
                            entity_type=target_entity.get("type", "UNKNOWN")
                        )
                
                graph.add_edge(
                    source_id,
                    target_id,
                    type=rel_type,
                    strength=strength
                )
        
        # Add nodes for recent events
        for event in self.narrative_context["recent_events"]:
            event_id = event.get("event_id")
            if event_id and not graph.has_node(event_id):
                graph.add_node(
                    event_id,
                    type="event",
                    text=event.get("name", ""),
                    event_type=event.get("event_type", "OTHER"),
                    status=event.get("status", "unknown")
                )
                
                # Link events to their entities
                for entity_ref in event.get("entities", []):
                    entity_text = entity_ref.get("text", "")
                    if entity_text:
                        # Find entity ID
                        entity = self.mongo.get_entity_by_name(user_id, entity_text)
                        if entity:
                            entity_id = entity.get("entity_id")
                            if entity_id and graph.has_node(entity_id):
                                graph.add_edge(
                                    event_id,
                                    entity_id,
                                    type=entity_ref.get("relation", "mentioned"),
                                    strength=1.0
                                )
        
        # Store the graph in the context
        self.narrative_context["relationship_graph"] = graph
    
    def _process_events(self, user_id: str, entry_id: str, 
                      nlp_results: Dict[str, Any], entry_date: datetime) -> List[Dict[str, Any]]:
        """
        Process events with enhanced context awareness and narrative understanding.
        
        Args:
            user_id: User identifier
            entry_id: Entry identifier
            nlp_results: Results from NLP processing
            entry_date: Date of the journal entry
            
        Returns:
            List of processed events with context and narrative connections
        """
        processed_events = []
        raw_events = nlp_results.get("core", {}).get("events", [])
        
        if not raw_events:
            return processed_events
        
        # Get existing events for the user
        existing_events = self.mongo.get_events(user_id)
        
        # Get recent events (higher contextual relevance)
        recent_events = self.narrative_context.get("recent_events", [])
        
        # Generate event embeddings for semantic matching
        event_embeddings = {}
        for event_data in raw_events:
            event_text = event_data.get("text", "")
            if event_text:
                event_embeddings[event_data.get("event_id", str(uuid.uuid4()))] = self.nlp_engine._generate_embedding(event_text)
        
        # Process each extracted event
        for event_data in raw_events:
            # Generate a stable ID for this event if not present
            if "event_id" not in event_data:
                event_data["event_id"] = str(uuid.uuid4())
            
            # Generate event name if not present
            if "name" not in event_data:
                event_text = event_data.get("text", "")
                event_data["name"] = self._generate_event_name(
                    event_data.get("components", {}).get("action", ""),
                    event_data.get("components", {}).get("participants", []),
                    event_data.get("components", {}).get("location", ""),
                    event_text
                )
            
            # First, look for similar events in recent context (lower threshold)
            similar_recent_event, recent_similarity = self._find_similar_event(
                event_data,
                recent_events,
                event_embeddings.get(event_data["event_id"])
            )
            
            # Then look in all events (higher threshold)
            similar_event, similarity = self._find_similar_event(
                event_data,
                existing_events,
                event_embeddings.get(event_data["event_id"])
            )
            
            # Prefer recent context matches
            if recent_similarity > 0.6:  # Lower threshold for recent events
                # Update recent existing event
                updated_event = self._update_existing_event(
                    similar_recent_event, 
                    event_data, 
                    entry_id, 
                    entry_date,
                    event_embeddings.get(event_data["event_id"])
                )
                processed_events.append(updated_event)
            elif similarity > 0.7:  # Higher threshold for older events
                # Update existing event
                updated_event = self._update_existing_event(
                    similar_event, 
                    event_data, 
                    entry_id, 
                    entry_date,
                    event_embeddings.get(event_data["event_id"])
                )
                processed_events.append(updated_event)
            else:
                # Create new event
                new_event = self._create_new_event(
                    user_id, 
                    event_data, 
                    entry_id, 
                    entry_date,
                    event_embeddings.get(event_data["event_id"])
                )
                processed_events.append(new_event)
        
        # Store event embeddings for later use
        nlp_results["event_embeddings"] = event_embeddings
        
        return processed_events
    
    def _find_similar_event(self, 
                         event_data: Dict[str, Any], 
                         existing_events: List[Dict[str, Any]],
                         embedding: Optional[List[float]] = None) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Find similar events using multiple similarity metrics.
        
        Args:
            event_data: New event data
            existing_events: List of existing events
            embedding: Vector embedding for the event
            
        Returns:
            Tuple containing the most similar event (or None) and the similarity score
        """
        if not existing_events:
            return None, 0.0
        
        max_similarity = 0.0
        most_similar_event = None
        
        for existing_event in existing_events:
            # Calculate overall similarity using multiple metrics
            similarity = self._calculate_event_similarity(event_data, existing_event, embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_event = existing_event
        
        return most_similar_event, max_similarity
    
    def _calculate_event_similarity(self, 
                                  event1: Dict[str, Any], 
                                  event2: Dict[str, Any],
                                  embedding1: Optional[List[float]] = None) -> float:
        """
        Calculate similarity between events using multiple metrics.
        
        Args:
            event1: First event
            event2: Second event
            embedding1: Vector embedding for the first event
            
        Returns:
            Combined similarity score between 0 and 1
        """
        scores = []
        weights = []
        
        # 1. Text similarity (using embeddings if available)
        if embedding1 is not None and "embedding" in event2:
            try:
                embedding2 = event2["embedding"]
                vector1 = np.array(embedding1)
                vector2 = np.array(embedding2)
                
                # Calculate cosine similarity
                embedding_similarity = np.dot(vector1, vector2) / (
                    np.linalg.norm(vector1) * np.linalg.norm(vector2)
                )
                scores.append(float(embedding_similarity))
                weights.append(3.0)  # Higher weight for semantic similarity
            except Exception as e:
                logger.error(f"Error calculating embedding similarity: {e}")
        
        # 2. Component similarity (participants, location, action)
        # Get components from event1
        e1_components = event1.get("components", {})
        e1_participants = set(e1_components.get("participants", []))
        e1_location = e1_components.get("location", "").lower()
        e1_action = e1_components.get("action", "").lower()
        
        # Get components from event2
        e2_participants = set(event2.get("participants", []))
        e2_location = event2.get("location", "").lower()
        e2_action = event2.get("action", "").lower()
        
        # Calculate participant similarity
        if e1_participants and e2_participants:
            common_participants = e1_participants.intersection(e2_participants)
            participant_similarity = len(common_participants) / max(len(e1_participants), len(e2_participants))
            scores.append(participant_similarity)
            weights.append(2.0)  # Higher weight for shared participants
        
        # Calculate location similarity
        if e1_location and e2_location:
            if e1_location == e2_location:
                location_similarity = 1.0
            elif e1_location in e2_location or e2_location in e1_location:
                location_similarity = 0.7
            else:
                location_similarity = 0.0
            scores.append(location_similarity)
            weights.append(1.5)  # Medium weight for location
        
        # Calculate action similarity
        if e1_action and e2_action:
            if e1_action == e2_action:
                action_similarity = 1.0
            elif e1_action in e2_action or e2_action in e1_action:
                action_similarity = 0.7
            else:
                action_similarity = 0.0
            scores.append(action_similarity)
            weights.append(1.0)  # Lower weight for action
        
        # 3. Event type similarity
        if "type" in event1 and "type" in event2:
            type_similarity = 1.0 if event1["type"] == event2["type"] else 0.2
            scores.append(type_similarity)
            weights.append(0.5)  # Lower weight for type
        
        # 4. Temporal proximity (if events have timestamps)
        if "last_mentioned" in event2 and isinstance(event2["last_mentioned"], datetime):
            # If event1 has a date, use it; otherwise use context date from function call
            event1_date = datetime.now()  # Default to now if no date
            if "temporality" in event1 and "date" in event1["temporality"]:
                event1_date = event1["temporality"]["date"]
            
            days_diff = abs((event1_date - event2["last_mentioned"]).days)
            
            # Temporal decay: events closer in time are more likely to be related
            if days_diff <= 1:
                temporal_score = 1.0  # Same day or adjacent day
            elif days_diff <= 7:
                temporal_score = 0.9 - (days_diff - 1) * 0.1  # Linear decay over a week
            elif days_diff <= 30:
                temporal_score = 0.3 - (days_diff - 7) * 0.01  # Slower decay over a month
            else:
                temporal_score = max(0.0, 0.1 - (days_diff - 30) * 0.001)  # Very slow decay after a month
            
            scores.append(temporal_score)
            weights.append(1.0)
        
        # 5. Narrative context similarity (if events are part of the same narrative)
        narrative_similarity = self._calculate_narrative_context_similarity(event1, event2)
        if narrative_similarity > 0:
            scores.append(narrative_similarity)
            weights.append(1.5)
        
        # Calculate weighted average if we have scores
        if scores:
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            total_weight = sum(weights)
            return weighted_sum / total_weight
        
        return 0.0
    
    def _calculate_narrative_context_similarity(self, event1: Dict[str, Any], event2: Dict[str, Any]) -> float:
        """
        Calculate similarity based on narrative context.
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            Narrative context similarity score
        """
        # Check if both events are part of the same active narratives
        if "narrative_ids" in event1 and "narrative_ids" in event2:
            event1_narratives = set(event1["narrative_ids"])
            event2_narratives = set(event2["narrative_ids"])
            
            common_narratives = event1_narratives.intersection(event2_narratives)
            if common_narratives:
                return 0.8  # High similarity if part of the same narrative
        
        # Check if events are connected in the relationship graph
        graph = self.narrative_context.get("relationship_graph")
        if graph and "event_id" in event1 and "event_id" in event2:
            event1_id = event1["event_id"]
            event2_id = event2["event_id"]
            
            if graph.has_node(event1_id) and graph.has_node(event2_id):
                # Check if events are directly connected
                if graph.has_edge(event1_id, event2_id):
                    return 0.7
                
                # Check if events are connected through a common entity
                event1_neighbors = set(graph.neighbors(event1_id))
                event2_neighbors = set(graph.neighbors(event2_id))
                
                common_neighbors = event1_neighbors.intersection(event2_neighbors)
                if common_neighbors:
                    return 0.6
                
                # Check if events are connected through a short path (2-3 hops)
                try:
                    path_length = nx.shortest_path_length(graph, event1_id, event2_id)
                    if path_length == 2:
                        return 0.5
                    elif path_length == 3:
                        return 0.3
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
        
        # Check if events are part of the same temporal sequence
        if "sequence_id" in event1 and "sequence_id" in event2:
            if event1["sequence_id"] == event2["sequence_id"]:
                return 0.7
        
        # Default: no narrative connection found
        return 0.0
    
    def _generate_event_name(self, action: str, participants: List[str], location: str, full_text: str) -> str:
        """Generate a descriptive name for an event based on its components."""
        # If we have an action and participants, use those
        if action and participants:
            # Take up to 2 participants for the name
            participant_str = " and ".join(participants[:2])
            if location:
                return f"{action} with {participant_str} at {location}"
            return f"{action} with {participant_str}"
        
        # If we just have an action, use it with location
        if action and location:
            return f"{action} at {location}"
        
        # If we just have an action, use it
        if action:
            return action
            
        # If we have no structured components, use the first part of the text
        words = full_text.split()
        if len(words) > 6:
            return " ".join(words[:6]) + "..."
        return full_text
    
    def _update_existing_event(self, event: Dict[str, Any], new_data: Dict[str, Any], 
                             entry_id: str, entry_date: datetime,
                             embedding: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Update an existing event with new information from a journal entry.
        
        Args:
            event: Existing event to update
            new_data: New event data extracted from the current entry
            entry_id: Current entry ID
            entry_date: Date of the current entry
            embedding: Vector embedding for the event
            
        Returns:
            Updated event dictionary
        """
        # Create a copy of the event to update
        event_copy = event.copy()
        
        # Remove MongoDB _id field if present
        if '_id' in event_copy:
            del event_copy['_id']
        
        # Update basic fields
        event_copy["last_mentioned"] = entry_date
        event_copy["text"] = new_data.get("text", event_copy.get("text", ""))
        
        # Generate or update event name
        if "name" in new_data:
            event_copy["name"] = new_data["name"]
        elif not event_copy.get("name"):
            event_copy["name"] = self._generate_event_name(
                new_data.get("components", {}).get("action", ""),
                new_data.get("components", {}).get("participants", []),
                new_data.get("components", {}).get("location", ""),
                new_data.get("text", "")
            )
        
        # Update mentions
        if "mentions" not in event_copy:
            event_copy["mentions"] = []
        
        event_copy["mentions"].append({
            "entry_id": entry_id,
            "date": entry_date,
            "text": new_data.get("text", ""),
            "sentiment": new_data.get("sentiment", {}).get("label", "NEUTRAL")
        })
        
        # Update participants
        if "participants" not in event_copy:
            event_copy["participants"] = []
        
        new_participants = new_data.get("components", {}).get("participants", [])
        for participant in new_participants:
            if participant not in event_copy["participants"]:
                event_copy["participants"].append(participant)
                
                # Add entity for tracking relationships
                if "entities" not in event_copy:
                    event_copy["entities"] = []
                    
                event_copy["entities"].append({
                    "text": participant,
                    "type": "PERSON",
                    "relation": "participant"
                })
        
        # Update location if provided and not already set
        new_location = new_data.get("components", {}).get("location", "")
        if new_location and not event_copy.get("location"):
            event_copy["location"] = new_location
            
            # Add entity for tracking relationships
            if "entities" not in event_copy:
                event_copy["entities"] = []
                
            event_copy["entities"].append({
                "text": new_location,
                "type": "LOCATION",
                "relation": "location"
            })
        
        # Update time information
        new_times = new_data.get("components", {}).get("time", [])
        if new_times and "time_mentions" not in event_copy:
            event_copy["time_mentions"] = []
            
        for time in new_times:
            if "time_mentions" not in event_copy:
                event_copy["time_mentions"] = []
                
            if time not in event_copy["time_mentions"]:
                event_copy["time_mentions"].append(time)
        
        # Update event embedding if provided
        if embedding:
            event_copy["embedding"] = embedding
        
        # Update narrative connections
        self._update_event_narrative_connections(event_copy, new_data)
        
        # Save updated event
        self.mongo.update_event(event_copy["event_id"], event_copy)
        
        # Add flags for frontend
        event_copy["is_new"] = False
        event_copy["is_updated"] = True
        if new_data.get("status"):
            event_copy["status_changed"] = True
            event_copy["previous_status"] = event.get("status")
        
        return event_copy
    
    def _update_event_narrative_connections(self, event: Dict[str, Any], new_data: Dict[str, Any]):
        """
        Update event's connections to narratives and story arcs.
        
        Args:
            event: Event to update
            new_data: New event data with potential narrative elements
        """
        # Check for narrative elements in new data
        narrative_elements = new_data.get("narrative_elements", {})
        
        if not narrative_elements:
            return
        
        # Initialize narrative IDs list if needed
        if "narrative_ids" not in event:
            event["narrative_ids"] = []
        
        # Check if the new data connects to any active narratives
        for narrative in self.narrative_context.get("active_narratives", []):
            narrative_id = narrative.get("narrative_id")
            if not narrative_id:
                continue
            
            # Check for connections to this narrative
            is_connected = False
            
            # Check if the event has similar arc stage
            if narrative.get("narrative_arc") == narrative_elements.get("narrative_arc"):
                is_connected = True
            
            # Check if the event involves the same characters/entities
            narrative_roles = narrative.get("narrative_roles", {})
            for role, entities in narrative_roles.items():
                for entity in entities:
                    if entity in event.get("participants", []):
                        is_connected = True
                        break
            
            # If connected, add narrative ID
            if is_connected and narrative_id not in event["narrative_ids"]:
                event["narrative_ids"].append(narrative_id)
    
    def _create_new_event(self, user_id: str, event_data: Dict[str, Any], 
                        entry_id: str, entry_date: datetime,
                        embedding: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Create a new event from extracted data with rich metadata.
        
        Args:
            user_id: User identifier
            event_data: Event data extracted from NLP processing
            entry_id: Current entry ID
            entry_date: Date of the current entry
            embedding: Vector embedding for the event
            
        Returns:
            Newly created event dictionary
        """
        # Ensure the event has an ID
        event_id = event_data.get("event_id", str(uuid.uuid4()))
        
        # Generate event name from components
        components = event_data.get("components", {})
        action = components.get("action", "")
        participants = components.get("participants", [])
        location = components.get("location", "")
        
        # Create a meaningful name with available information
        event_name = self._generate_event_name(action, participants, location, event_data["text"])
        # Create entities list for relationship tracking
        entities = []
        
        # Add participants as entities
        for participant in participants:
            entities.append({
                "text": participant,
                "type": "PERSON",
                "relation": "participant"
            })
            
        # Add location as entity
        if location:
            entities.append({
                "text": location,
                "type": "LOCATION",
                "relation": "location"
            })
        
        # Determine event status
        status = "ongoing"
        if event_data.get("completed", False):
            status = "completed"
        elif event_data.get("status"):
            status = event_data["status"]
        
        # Create new event with rich metadata
        new_event = {
            "event_id": event_id,
            "user_id": user_id,
            "name": event_name,
            "description": event_data["text"],
            "action": action,
            "status": status,
            "event_type": event_data.get("type", "OTHER"),
            "importance": event_data.get("importance", "MEDIUM"),
            "first_mentioned": entry_date,
            "last_mentioned": entry_date,
            "created_at": datetime.now(),
            "mentions": [{
                "entry_id": entry_id,
                "date": entry_date,
                "text": event_data["text"],
                "sentiment": event_data.get("sentiment", {}).get("label", "NEUTRAL")
            }],
            "participants": participants,
            "location": location,
            "time_mentions": components.get("time", []),
            "entities": entities,
            "narrative_ids": [],  # Will be populated by _connect_event_to_narratives
            "temporality": event_data.get("temporality", {}),
            "regularity": event_data.get("regularity", "ONE_TIME")
        }
        
        # Add embedding if available
        if embedding:
            new_event["embedding"] = embedding
        
        # Connect to narratives and sequences
        self._connect_event_to_narratives(new_event, event_data, entry_date)
        
        # Save to database
        self.mongo.add_event(user_id, new_event)
        
        # Add flag for frontend
        new_event["is_new"] = True
        
        return new_event
    
    def _connect_event_to_narratives(self, event: Dict[str, Any], 
                                   event_data: Dict[str, Any],
                                   entry_date: datetime):
        """
        Connect a new event to existing narratives or create a new narrative.
        
        Args:
            event: Event to connect
            event_data: Raw event data
            entry_date: Date of the current entry
        """
        # Check for narrative elements
        narrative_elements = event_data.get("narrative_elements", {})
        
        if not narrative_elements:
            return
        
        # Check if this event connects to any active narratives
        connected_to_existing = False
        
        for narrative in self.narrative_context.get("active_narratives", []):
            narrative_id = narrative.get("narrative_id")
            if not narrative_id:
                continue
            
            # Check for connections to this narrative
            is_connected = False
            
            # Check if the event has similar arc stage
            if narrative.get("narrative_arc") == narrative_elements.get("narrative_arc"):
                is_connected = True
            
            # Check if the event involves the same characters/entities
            narrative_roles = narrative.get("narrative_roles", {})
            for role, entities in narrative_roles.items():
                for entity in entities:
                    if entity in event.get("participants", []):
                        is_connected = True
                        break
            
            # If connected, add narrative ID
            if is_connected:
                if "narrative_ids" not in event:
                    event["narrative_ids"] = []
                if narrative_id not in event["narrative_ids"]:
                    event["narrative_ids"].append(narrative_id)
                connected_to_existing = True
        
        # If not connected to any existing narrative, consider creating a new one
        if not connected_to_existing and narrative_elements.get("narrative_arc"):
            # Only create a new narrative if the event seems significant
            if event.get("importance") in ["HIGH", "MEDIUM"]:
                # Create a new narrative
                narrative_id = str(uuid.uuid4())
                
                narrative = {
                    "narrative_id": narrative_id,
                    "user_id": event["user_id"],
                    "entry_id": event["mentions"][0]["entry_id"],
                    "date": entry_date,
                    "narrative_arc": narrative_elements.get("narrative_arc"),
                    "goals": narrative_elements.get("goals", []),
                    "conflicts": narrative_elements.get("conflicts", []),
                    "learnings": narrative_elements.get("learnings", []),
                    "reflections": narrative_elements.get("reflections", []),
                    "emotional_arc": narrative_elements.get("emotional_arc", []),
                    "narrative_roles": narrative_elements.get("narrative_roles", {}),
                    "events": [event["event_id"]],
                    "created_at": datetime.now()
                }
                
                # Save to database
                self.mongo.narratives.insert_one(narrative)
                
                # Connect event to narrative
                if "narrative_ids" not in event:
                    event["narrative_ids"] = []
                event["narrative_ids"].append(narrative_id)
    
    def _process_entities(self, user_id: str, entry_id: str, 
                        nlp_results: Dict[str, Any], entry_date: datetime) -> List[Dict[str, Any]]:
        """
        Process entities with relationship tracking and context awareness.
        
        Args:
            user_id: User identifier
            entry_id: Entry identifier
            nlp_results: Results from NLP processing
            entry_date: Date of the journal entry
            
        Returns:
            List of processed entities with context and relationship data
        """
        processed_entities = []
        raw_entities = nlp_results.get("core", {}).get("entities", [])
        
        if not raw_entities:
            return processed_entities
        
        # Generate entity embeddings for semantic matching
        entity_embeddings = {}
        for entity_data in raw_entities:
            entity_text = entity_data.get("text", "")
            if entity_text:
                entity_embeddings[entity_text] = self.nlp_engine._generate_embedding(entity_text)
        
        # Store entity embeddings in NLP results
        nlp_results["entity_embeddings"] = entity_embeddings
        
        # Process each entity
        for entity_data in raw_entities:
            entity_text = entity_data.get("text", "")
            if not entity_text:
                continue
            
            # Look for existing entity
            existing_entity = self.mongo.get_entity_by_name(user_id, entity_text)
            
            if existing_entity:
                # Update existing entity
                updated_entity = self._update_existing_entity(
                    existing_entity,
                    entity_data,
                    entry_id,
                    entry_date,
                    entity_embeddings.get(entity_text)
                )
                processed_entities.append(updated_entity)
            else:
                # Create new entity
                new_entity = self._create_new_entity(
                    user_id,
                    entity_data,
                    entry_id,
                    entry_date,
                    entity_embeddings.get(entity_text)
                )
                processed_entities.append(new_entity)
        
        # Process relationships between entities
        self._process_entity_relationships(
            user_id,
            processed_entities,
            nlp_results.get("enhanced", {}).get("relationships", []),
            entry_date
        )
        
        return processed_entities
    
    def _update_existing_entity(self, entity: Dict[str, Any], new_data: Dict[str, Any],
                              entry_id: str, entry_date: datetime,
                              embedding: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Update an existing entity with new information.
        
        Args:
            entity: Existing entity to update
            new_data: New entity data
            entry_id: Current entry ID
            entry_date: Date of the current entry
            embedding: Vector embedding for the entity
            
        Returns:
            Updated entity dictionary
        """
        # Create a copy to avoid modifying the original
        entity_copy = entity.copy()
        # Get a mutable copy of the event

# Remove the '_id' field if it exists in the copy to avoid MongoDB error
        if '_id' in entity_copy:
            del entity_copy['_id']
        
        # Update metadata
        if "metadata" in new_data:
            if "metadata" not in entity_copy:
                entity_copy["metadata"] = {}
            
            for key, value in new_data["metadata"].items():
                entity_copy["metadata"][key] = value
        
        # Update type and subtype if more specific
        if "type" in new_data:
            if new_data["type"] != "UNKNOWN" and (
                "type" not in entity_copy or entity_copy["type"] == "UNKNOWN"
            ):
                entity_copy["type"] = new_data["type"]
        
        if "subtype" in new_data and new_data["subtype"]:
            entity_copy["subtype"] = new_data["subtype"]
        
        # Update mention count and timestamps
        entity_copy["mention_count"] = entity_copy.get("mention_count", 0) + 1
        entity_copy["last_mentioned"] = entry_date
        
        # Add entry to entry_ids
        if "entry_ids" not in entity_copy:
            entity_copy["entry_ids"] = []
        if entry_id not in entity_copy["entry_ids"]:
            entity_copy["entry_ids"].append(entry_id)
        
        # Add context
        if "contexts" not in entity_copy:
            entity_copy["contexts"] = []
        
        entity_copy["contexts"].append({
            "text": new_data.get("context", new_data.get("text", "")),
            "entry_id": entry_id,
            "date": entry_date,
            "sentiment": new_data.get("sentiment", {}).get("label", "NEUTRAL")
        })
        
        # Add to history
        if "history" not in entity_copy:
            entity_copy["history"] = []
        
        entity_copy["history"].append({
            "date": entry_date,
            "entry_id": entry_id,
            "sentiment": new_data.get("sentiment", {}).get("label", "NEUTRAL")
        })
        
        # Update embedding if provided
        if embedding:
            entity_copy["embedding"] = embedding
        
        # Save updated entity
        # Create a copy without the _id field
        entity_update = entity_copy.copy()
        if '_id' in entity_update:
            del entity_update['_id']  # Remove _id field before update

        self.mongo.entities.update_one(
            {"entity_id": entity_copy["entity_id"]},
            {"$set": entity_update}
)
        
        # Add flags for frontend
        entity_copy["is_new"] = False
        entity_copy["is_updated"] = True
        
        return entity_copy
    
    def _create_new_entity(self, user_id: str, entity_data: Dict[str, Any],
                         entry_id: str, entry_date: datetime,
                         embedding: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Create a new entity from extracted data.
        
        Args:
            user_id: User identifier
            entity_data: Entity data
            entry_id: Current entry ID
            entry_date: Date of the current entry
            embedding: Vector embedding for the entity
            
        Returns:
            Newly created entity dictionary
        """
        entity_text = entity_data["text"]
        entity_id = self._generate_entity_id(entity_text)
        
        # Create new entity
        new_entity = {
            "entity_id": entity_id,
            "user_id": user_id,
            "text": entity_text,
            "type": entity_data.get("type", "UNKNOWN"),
            "subtype": entity_data.get("subtype"),
            "first_mentioned": entry_date,
            "last_mentioned": entry_date,
            "mention_count": 1,
            "entry_ids": [entry_id],
            "contexts": [{
                "text": entity_data.get("context", entity_text),
                "entry_id": entry_id,
                "date": entry_date,
                "sentiment": entity_data.get("sentiment", {}).get("label", "NEUTRAL")
            }],
            "history": [{
                "date": entry_date,
                "entry_id": entry_id,
                "sentiment": entity_data.get("sentiment", {}).get("label", "NEUTRAL")
            }],
            "metadata": entity_data.get("metadata", {}),
            "created_at": datetime.now()
        }
        
        # Add embedding if available
        if embedding:
            new_entity["embedding"] = embedding
        
        # Save to database
        self.mongo.entities.insert_one(new_entity)
        
        # Add flag for frontend
        new_entity["is_new"] = True
        
        return new_entity
    
    def _generate_entity_id(self, entity_text: str) -> str:
        """
        Generate a stable entity ID from entity text.
        
        Args:
            entity_text: Entity text
            
        Returns:
            Entity ID
        """
        import hashlib
        return hashlib.md5(entity_text.lower().encode()).hexdigest()
    
    def _process_entity_relationships(self, user_id: str, entities: List[Dict[str, Any]],
                                    relationships: List[Dict[str, Any]], entry_date: datetime):
        """
        Process relationships between entities.
        
        Args:
            user_id: User identifier
            entities: Processed entities
            relationships: Detected relationships
            entry_date: Date of the current entry
        """
        # Process explicit relationships from NLP
        for relationship in relationships:
            source_text = relationship.get("entity1", "")
            target_text = relationship.get("entity2", "")
            relationship_type = relationship.get("type", "ASSOCIATED")
            
            if not source_text or not target_text:
                continue
            
            # Generate entity IDs
            source_id = self._generate_entity_id(source_text)
            target_id = self._generate_entity_id(target_text)
            
            # Store relationship
            existing = self.mongo.entity_relationships.find_one({
                "user_id": user_id,
                "source_id": source_id,
                "target_id": target_id
            })
            
            if existing:
                # Update existing relationship
                self.mongo.entity_relationships.update_one(
                    {
                        "user_id": user_id,
                        "source_id": source_id,
                        "target_id": target_id
                    },
                    {
                        "$set": {
                            "relationship_type": relationship_type,
                            "last_updated": datetime.now()
                        },
                        "$inc": {
                            "strength": 1
                        },
                        "$push": {
                            "contexts": relationship.get("context", "")
                        }
                    }
                )
            else:
                # Create new relationship
                self.mongo.entity_relationships.insert_one({
                    "user_id": user_id,
                    "source_id": source_id,
                    "target_id": target_id,
                    "source_text": source_text,
                    "target_text": target_text,
                    "relationship_type": relationship_type,
                    "strength": 1,
                    "contexts": [relationship.get("context", "")],
                    "created_at": datetime.now(),
                    "last_updated": datetime.now()
                })
        
        # Infer additional relationships from co-occurrence
        entity_ids = [e["entity_id"] for e in entities]
        
        # Only infer relationships if we have multiple entities
        if len(entity_ids) > 1:
            # Create relationships between co-occurring entities
            for i in range(len(entities)):
                for j in range(i+1, len(entities)):
                    # Skip if both are locations
                    if entities[i].get("type") == "LOCATION" and entities[j].get("type") == "LOCATION":
                        continue
                    
                    source_id = entities[i]["entity_id"]
                    target_id = entities[j]["entity_id"]
                    
                    # Determine relationship type
                    rel_type = "CO_OCCURRED"
                    
                    # Check for specific entity type combinations
                    if entities[i].get("type") == "PERSON" and entities[j].get("type") == "PERSON":
                        rel_type = "ASSOCIATED_WITH"
                    elif (entities[i].get("type") == "PERSON" and entities[j].get("type") == "LOCATION") or \
                         (entities[i].get("type") == "LOCATION" and entities[j].get("type") == "PERSON"):
                        rel_type = "LOCATED_AT"
                    elif (entities[i].get("type") == "PERSON" and entities[j].get("type") == "ORG") or \
                         (entities[i].get("type") == "ORG" and entities[j].get("type") == "PERSON"):
                        rel_type = "AFFILIATED_WITH"
                    
                    # Store relationship
                    existing = self.mongo.entity_relationships.find_one({
                        "user_id": user_id,
                        "source_id": source_id,
                        "target_id": target_id
                    })
                    
                    if existing:
                        # Update existing relationship
                        self.mongo.entity_relationships.update_one(
                            {
                                "user_id": user_id,
                                "source_id": source_id,
                                "target_id": target_id
                            },
                            {
                                "$inc": {
                                    "strength": 1
                                },
                                "$set": {
                                    "last_updated": datetime.now()
                                }
                            }
                        )
                    else:
                        # Create new relationship
                        self.mongo.entity_relationships.insert_one({
                            "user_id": user_id,
                            "source_id": source_id,
                            "target_id": target_id,
                            "source_text": entities[i]["text"],
                            "target_text": entities[j]["text"],
                            "relationship_type": rel_type,
                            "strength": 1,
                            "contexts": [],
                            "created_at": datetime.now(),
                            "last_updated": datetime.now(),
                            "inferred": True
                        })
    
    def _process_emotions(self, user_id: str, entry_id: str, 
                        nlp_results: Dict[str, Any], entry_date: datetime) -> Dict[str, Any]:
        """
        Process emotions and emotional arcs.
        
        Args:
            user_id: User identifier
            entry_id: Entry identifier
            nlp_results: Results from NLP processing
            entry_date: Date of the journal entry
            
        Returns:
            Dictionary with emotional analysis results
        """
        emotions = nlp_results.get("enhanced", {}).get("emotions", {})
        
        if not emotions:
            return {}
        
        # Create emotional arc document
        emotional_arc = {
            "user_id": user_id,
            "entry_id": entry_id,
            "date": entry_date,
            "primary_emotion": emotions.get("primary_emotion"),
            "secondary_emotion": emotions.get("secondary_emotion"),
            "intensity": emotions.get("intensity", 0.0),
            "emotions_detected": emotions.get("emotions_detected", []),
            "emotional_phrases": emotions.get("emotional_phrases", []),
            "targets": emotions.get("targets", {}),
            "created_at": datetime.now()
        }
        
        # Store in database
        self.mongo.emotional_arcs.insert_one(emotional_arc)
        
        # Check for emotion targets that match entities
        targets = emotions.get("targets", {})
        for emotion, target_list in targets.items():
            for target in target_list:
                # Find matching entity
                entity = self.mongo.get_entity_by_name(user_id, target)
                
                if entity:
                    # Update entity with emotion information
                    self.mongo.entities.update_one(
                        {"entity_id": entity["entity_id"]},
                        {
                            "$push": {
                                "emotional_mentions": {
                                    "emotion": emotion,
                                    "entry_id": entry_id,
                                    "date": entry_date
                                }
                            }
                        }
                    )
        
        # Analyze emotional continuity with previous entries
        emotional_continuity = self._analyze_emotional_continuity(
            user_id, 
            emotions.get("primary_emotion"), 
            entry_date
        )
        
        # Prepare response
        response = {
            "primary_emotion": emotions.get("primary_emotion"),
            "secondary_emotion": emotions.get("secondary_emotion"),
            "intensity": emotions.get("intensity", 0.0),
            "emotional_continuity": emotional_continuity
        }
        
        return response
    
    def _analyze_emotional_continuity(self, user_id: str, current_emotion: str, 
                                    entry_date: datetime) -> Dict[str, Any]:
        """
        Analyze emotional continuity with previous entries.
        
        Args:
            user_id: User identifier
            current_emotion: Current primary emotion
            entry_date: Date of the current entry
            
        Returns:
            Dictionary with emotional continuity analysis
        """
        if not current_emotion:
            return {}
        
        # Get recent emotional arcs (past 14 days)
        recent_date = entry_date - timedelta(days=14)
        recent_emotions = list(self.mongo.emotional_arcs.find({
            "user_id": user_id,
            "date": {"$gte": recent_date, "$lt": entry_date}
        }).sort("date", -1))
        
        if not recent_emotions:
            return {"status": "NEW_EMOTION", "previous_emotions": []}
        
        # Check if current emotion continues or changes
        recent_primary = [e.get("primary_emotion") for e in recent_emotions if "primary_emotion" in e]
        if not recent_primary:
            return {"status": "NEW_EMOTION", "previous_emotions": []}
        
        most_recent = recent_primary[0]
        
        if current_emotion == most_recent:
            consecutive_count = 1
            for emotion in recent_primary[1:]:
                if emotion == current_emotion:
                    consecutive_count += 1
                else:
                    break
            
            return {
                "status": "CONTINUING_EMOTION",
                "emotion": current_emotion,
                "consecutive_entries": consecutive_count + 1,  # Including current entry
                "previous_emotions": recent_primary[:5]
            }
        else:
            # Emotion has changed
            return {
                "status": "CHANGING_EMOTION",
                "from": most_recent,
                "to": current_emotion,
                "previous_emotions": recent_primary[:5]
            }
    
    def _process_narrative_elements(self, user_id: str, entry_id: str, 
                                  nlp_results: Dict[str, Any], entry_date: datetime) -> Dict[str, Any]:
        """
        Process narrative elements and story structure.
        
        Args:
            user_id: User identifier
            entry_id: Entry identifier
            nlp_results: Results from NLP processing
            entry_date: Date of the journal entry
            
        Returns:
            Dictionary with narrative analysis results
        """
        narrative_elements = nlp_results.get("enhanced", {}).get("narrative_elements", {})
        
        if not narrative_elements:
            return {}
        
        # Extract narrative arc, goals, conflicts, etc.
        narrative_id = str(uuid.uuid4())
        narrative_doc = {
            "narrative_id": narrative_id,
            "user_id": user_id,
            "entry_id": entry_id,
            "date": entry_date,
            "narrative_arc": narrative_elements.get("narrative_arc"),
            "goals": narrative_elements.get("goals", []),
            "conflicts": narrative_elements.get("conflicts", []),
            "learnings": narrative_elements.get("learnings", []),
            "reflections": narrative_elements.get("reflections", []),
            "emotional_arc": narrative_elements.get("emotional_arc", []),
            "narrative_roles": narrative_elements.get("narrative_roles", {}),
            "related_events": [],  # Will be populated later
            "created_at": datetime.now()
        }
        
        # Add any causal relationships
        if "causal_relationships" in narrative_elements:
            narrative_doc["causal_relationships"] = narrative_elements["causal_relationships"]
        
        # Generate narrative embedding
        narrative_embedding = None
        if "narrative_arc" in narrative_elements:
            # Create a narrative description
            narrative_desc = f"A {narrative_elements['narrative_arc']} narrative"
            
            if narrative_elements.get("goals"):
                goals_text = " ".join([g.get("text", "") for g in narrative_elements["goals"]])
                narrative_desc += f" involving goals like {goals_text}"
            
            if narrative_elements.get("conflicts"):
                conflicts_text = " ".join([c.get("text", "") for c in narrative_elements["conflicts"]])
                narrative_desc += f" with conflicts like {conflicts_text}"
            
            narrative_embedding = self.nlp_engine._generate_embedding(narrative_desc)
            
            # Store embedding
            nlp_results["narrative_embedding"] = narrative_embedding
        
        # Store narrative document
        self.mongo.narratives.insert_one(narrative_doc)
        
        # Check for connections to existing narratives
        connected_narratives = self._connect_to_existing_narratives(
            user_id, 
            narrative_id, 
            narrative_elements, 
            narrative_embedding
        )
        
        # Prepare response
        response = {
            "narrative_id": narrative_id,
            "narrative_arc": narrative_elements.get("narrative_arc"),
            "goals": len(narrative_elements.get("goals", [])),
            "conflicts": len(narrative_elements.get("conflicts", [])),
            "connected_narratives": connected_narratives
        }
        
        return response
    
    def _connect_to_existing_narratives(self, user_id: str, narrative_id: str, 
                                      narrative_elements: Dict[str, Any],
                                      embedding: Optional[List[float]]) -> List[Dict[str, Any]]:
        """
        Connect a new narrative to existing narratives.
        
        Args:
            user_id: User identifier
            narrative_id: ID of the new narrative
            narrative_elements: Narrative elements from NLP
            embedding: Vector embedding for the narrative
            
        Returns:
            List of connected narratives
        """
        connected = []
        
        # Get active narratives
        active_narratives = self.narrative_context.get("active_narratives", [])
        
        if not active_narratives:
            return connected
        
        # Extract entities from narrative roles
        narrative_entities = set()
        for role, entities in narrative_elements.get("narrative_roles", {}).items():
            for entity in entities:
                narrative_entities.add(entity)
        
        # Check each active narrative for connections
        for narrative in active_narratives:
            existing_id = narrative.get("narrative_id")
            if not existing_id or existing_id == narrative_id:
                continue
            
            connection_strength = 0
            connection_type = "unknown"
            
            # Check if narratives share the same arc type
            if narrative.get("narrative_arc") == narrative_elements.get("narrative_arc"):
                connection_strength += 0.5
                connection_type = "same_arc"
            
            # Check for shared entities
            existing_entities = set()
            for role, entities in narrative.get("narrative_roles", {}).items():
                for entity in entities:
                    existing_entities.add(entity)
            
            shared_entities = narrative_entities.intersection(existing_entities)
            if shared_entities:
                connection_strength += min(0.7, len(shared_entities) * 0.1)
                connection_type = "shared_entities"
            
            # Check semantic similarity if embeddings available
            if embedding and "embedding" in narrative:
                try:
                    narrative_embedding = np.array(embedding)
                    existing_embedding = np.array(narrative["embedding"])
                    
                    similarity = np.dot(narrative_embedding, existing_embedding) / (
                        np.linalg.norm(narrative_embedding) * np.linalg.norm(existing_embedding)
                    )
                    
                    connection_strength = max(connection_strength, float(similarity))
                    
                    if similarity > 0.7:
                        connection_type = "semantic_similarity"
                except Exception as e:
                    logger.error(f"Error calculating narrative embedding similarity: {e}")
            
            # If significant connection found, store it
            if connection_strength > 0.3:
                # Store connection in database
                self.mongo.narratives.update_one(
                    {"narrative_id": narrative_id},
                    {
                        "$push": {
                            "connected_narratives": {
                                "narrative_id": existing_id,
                                "strength": connection_strength,
                                "type": connection_type
                            }
                        }
                    }
                )
                
                self.mongo.narratives.update_one(
                    {"narrative_id": existing_id},
                    {
                        "$push": {
                            "connected_narratives": {
                                "narrative_id": narrative_id,
                                "strength": connection_strength,
                                "type": connection_type
                            }
                        }
                    }
                )
                
                # Add to result
                connected.append({
                    "narrative_id": existing_id,
                    "strength": connection_strength,
                    "type": connection_type
                })
        
        return connected
    
    def _process_temporal_sequences(self, user_id: str, entry_id: str, 
                                  nlp_results: Dict[str, Any], entry_date: datetime) -> Dict[str, Any]:
        """
        Process and analyze temporal sequences.
        
        Args:
            user_id: User identifier
            entry_id: Entry identifier
            nlp_results: Results from NLP processing
            entry_date: Date of the journal entry
            
        Returns:
            Dictionary with temporal sequence analysis
        """
        temporal_info = nlp_results.get("enhanced", {}).get("temporal_information", {})
        events = nlp_results.get("core", {}).get("events", [])
        
        if not temporal_info or not events:
            return {}
        
        # Check for ordered events that might form a sequence
        ordered_events = []
        for event in events:
            if event.get("temporality"):
                ordered_events.append({
                    "event_id": event.get("event_id", str(uuid.uuid4())),
                    "text": event.get("text", ""),
                    "type": event.get("type", "OTHER"),
                    "status": event.get("status", "UNKNOWN"),
                    "temporality": event.get("temporality", {})
                })
        
        if not ordered_events:
            return {}
        
        # Try to identify a sequence type
        sequence_type = self._identify_sequence_type(ordered_events)
        
        if not sequence_type:
            sequence_type = "GENERIC_SEQUENCE"
        
        # Create a new temporal sequence
        sequence_id = str(uuid.uuid4())
        sequence_doc = {
            "sequence_id": sequence_id,
            "user_id": user_id,
            "entry_id": entry_id,
            "sequence_type": sequence_type,
            "events": ordered_events,
            "start_date": entry_date,
            "created_at": datetime.now()
        }
        
        self.mongo.temporal_sequences.insert_one(sequence_doc)
        
        # Check if this sequence continues a previous one
        related_sequence = self._link_to_existing_sequences(user_id, sequence_id, sequence_type, ordered_events)
        
        # Prepare response
        response = {
            "sequence_id": sequence_id,
            "sequence_type": sequence_type,
            "event_count": len(ordered_events)
        }
        
        if related_sequence:
            response["related_sequence"] = related_sequence
        
        return response
    
    def _identify_sequence_type(self, events: List[Dict[str, Any]]) -> Optional[str]:
        """
        Identify the type of temporal sequence based on event patterns.
        
        Args:
            events: List of events in the sequence
            
        Returns:
            Sequence type or None
        """
        # Count event types
        type_counts = {}
        for event in events:
            event_type = event.get("type", "OTHER")
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
        
        # Find the dominant event type
        if type_counts:
            dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]
            return f"SEQUENCE_{dominant_type}"
        
        return None
    
    def _link_to_existing_sequences(self, user_id: str, sequence_id: str, 
                                  sequence_type: str, events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Link the new sequence to existing sequences.
        
        Args:
            user_id: User identifier
            sequence_id: ID of the new sequence
            sequence_type: Type of the new sequence
            events: Events in the new sequence
            
        Returns:
            Related sequence information or None
        """
        # Get recent sequences
        recent_sequences = self.narrative_context.get("temporal_sequences", [])
        
        if not recent_sequences:
            return None
        
        # For each recent sequence, check if the new sequence might be related
        for recent in recent_sequences:
            recent_id = recent.get("sequence_id")
            if not recent_id or recent_id == sequence_id:
                continue
            
            recent_type = recent.get("sequence_type", "")
            recent_events = recent.get("events", [])
            
            # Check if sequences are of the same type
            if recent_type != sequence_type:
                continue
            
            # Check for overlapping entities or actions
            recent_texts = [e.get("text", "").lower() for e in recent_events]
            current_texts = [e.get("text", "").lower() for e in events]
            
            is_related = False
            
            # Check for literal text matches
            for r_text in recent_texts:
                for c_text in current_texts:
                    if r_text in c_text or c_text in r_text:
                        is_related = True
                        break
                if is_related:
                    break
            
            if is_related:
                # Link sequences
                self.mongo.temporal_sequences.update_one(
                    {"sequence_id": sequence_id},
                    {"$set": {"related_sequence_id": recent_id}}
                )
                
                self.mongo.temporal_sequences.update_one(
                    {"sequence_id": recent_id},
                    {"$set": {"continued_by_sequence_id": sequence_id}}
                )
                
                return {
                    "sequence_id": recent_id,
                    "sequence_type": recent_type,
                    "event_count": len(recent_events)
                }
        
        return None
    
    def _generate_component_embeddings(self, nlp_results: Dict[str, Any]) -> Dict[str, Dict[str, List[float]]]:
        """
        Generate embeddings for different components (paragraphs, entities).
        
        Args:
            nlp_results: Results from NLP processing
            
        Returns:
            Dictionary with component embeddings
        """
        component_embeddings = {}
        
        # Already generated embeddings during processing
        if "entity_embeddings" in nlp_results:
            component_embeddings["entity_embeddings"] = nlp_results["entity_embeddings"]
        
        if "event_embeddings" in nlp_results:
            component_embeddings["event_embeddings"] = nlp_results["event_embeddings"]
        
        if "narrative_embedding" in nlp_results:
            component_embeddings["narrative_embedding"] = nlp_results["narrative_embedding"]
        
        # Generate paragraph embeddings if text can be split into paragraphs
        if "text" in nlp_results:
            text = nlp_results["text"]
            paragraphs = text.split("\n\n")
            
            if len(paragraphs) > 1:
                paragraph_embeddings = []
                
                for i, para in enumerate(paragraphs):
                    if para.strip():
                        embedding = self.nlp_engine._generate_embedding(para)
                        paragraph_embeddings.append({
                            "text": para,
                            "embedding": embedding,
                            "index": i
                        })
                
                component_embeddings["paragraphs"] = paragraph_embeddings
        
        return component_embeddings
    
    def _get_active_narratives(self, user_id: str, current_date: datetime) -> List[Dict[str, Any]]:
        """
        Get active narratives based on recency and context.
        
        Args:
            user_id: User identifier
            current_date: Current date
            
        Returns:
            List of active narratives
        """
        # Get narratives from the past 30 days
        recent_date = current_date - timedelta(days=30)
        
        recent_narratives = list(self.mongo.narratives.find({
            "user_id": user_id,
            "date": {"$gte": recent_date}
        }).sort("date", -1))
        
        active_narratives = []
        
        for narrative in recent_narratives:
            # Remove MongoDB _id
            narrative["_id"] = str(narrative["_id"])
            
            # Include if it's recent or has connected events
            if len(narrative.get("related_events", [])) > 0 or \
               (current_date - narrative["date"]).days <= 14:
                active_narratives.append(narrative)
        
        return active_narratives
    
    def get_ongoing_events(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all ongoing events for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of ongoing events
        """
        events = self.mongo.get_events(user_id, status="ongoing")
        
        # Sort by recency and importance
        events.sort(key=lambda e: (
            e.get("last_mentioned", datetime(1970, 1, 1)),
            {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(e.get("importance", "MEDIUM"), 0)
        ), reverse=True)
        
        return events
    
    def get_recent_events(self, user_id: str, days: int = 14) -> List[Dict[str, Any]]:
        """
        Get events from the past N days.
        
        Args:
            user_id: User identifier
            days: Number of days to look back
            
        Returns:
            List of recent events
        """
        return self.mongo.get_recent_events(user_id, days)
    
    def get_event_timeline(self, user_id: str, event_id: str) -> List[Dict[str, Any]]:
        """
        Get the timeline of an event's mentions with context.
        
        Args:
            user_id: User identifier
            event_id: Event identifier
            
        Returns:
            List of event mentions with context
        """
        event = self.mongo.get_event(event_id)
        if not event or event["user_id"] != user_id:
            return []
        
        timeline = []
        
        for mention in event.get("mentions", []):
            entry_id = mention.get("entry_id")
            if not entry_id:
                continue
            
            # Get entry context
            entry = self.mongo.get_journal_entry(entry_id)
            if entry:
                timeline_item = mention.copy()
                timeline_item["entry_date"] = entry.get("date")
                timeline_item["entry_summary"] = entry.get("summary", {})
                
                timeline.append(timeline_item)
        
        # Sort by date
        timeline.sort(key=lambda x: x.get("date", datetime(1970, 1, 1)))
        
        return timeline
    
    def mark_event_completed(self, user_id: str, event_id: str) -> Dict[str, Any]:
        """
        Mark an event as completed.
        
        Args:
            user_id: User identifier
            event_id: Event identifier
            
        Returns:
            Result dictionary
        """
        event = self.mongo.get_event(event_id)
        if not event or event["user_id"] != user_id:
            return {"success": False, "message": "Event not found"}
        
        completion_date = datetime.now()
        
        self.mongo.update_event(event_id, {
            "status": "completed",
            "completed_date": completion_date
        })
        
        # Check for related events that might also be completed
        related_events = self.get_related_events(user_id, event_id)
        potentially_completed = []
        
        for related in related_events:
            if related.get("status") == "ongoing" and \
               related.get("event_type") == event.get("event_type") and \
               related.get("similarity", 0) > 0.8:
                potentially_completed.append({
                    "event_id": related["event_id"],
                    "name": related["name"],
                    "similarity": related.get("similarity", 0)
                })
        
        return {
            "success": True,
            "message": "Event marked as completed",
            "event_id": event_id,
            "completed_date": completion_date,
            "potentially_completed": potentially_completed
        }
    
    def get_event_network(self, user_id: str, include_completed: bool = True) -> Dict[str, Any]:
        """
        Get the network of events and their relationships.
        
        Args:
            user_id: User identifier
            include_completed: Whether to include completed events
            
        Returns:
            Network of events and relationships
        """
        # Get events based on filter
        events = []
        if include_completed:
            events = self.mongo.get_events(user_id)
        else:
            events = self.mongo.get_events(user_id, status="ongoing")
        
        # Build network
        graph = nx.Graph()
        
        # Add nodes for events
        for event in events:
            # Generate a fallback name if 'name' is missing
            event_name = event.get("name")
            if not event_name:
                # Try to build a name from other fields
                if "description" in event:
                    event_name = event["description"][:50] + ("..." if len(event["description"]) > 50 else "")
                elif "text" in event:
                    event_name = event["text"][:50] + ("..." if len(event["text"]) > 50 else "")
                else:
                    # Last resort, use the event ID or "Unnamed Event"
                    event_name = f"Event {event.get('event_id', 'Unnamed')}"
            
            graph.add_node(
                event["event_id"],
                name=event_name,
                type=event.get("event_type", "OTHER"),
                status=event.get("status", "unknown"),
                first_mentioned=event.get("first_mentioned"),
                last_mentioned=event.get("last_mentioned")
            )
        
        # Add edges based on relationships
        self._add_event_relationships_to_graph(graph, events)
        
        # Convert to serializable format
        nodes = []
        links = []
        
        for node_id in graph.nodes():
            attrs = graph.nodes[node_id]
            nodes.append({
                "id": node_id,
                "name": attrs.get("name", ""),
                "type": attrs.get("type", "OTHER"),
                "status": attrs.get("status", "unknown"),
                "first_mentioned": attrs.get("first_mentioned"),
                "last_mentioned": attrs.get("last_mentioned")
            })
        
        for source, target, attrs in graph.edges(data=True):
            links.append({
                "source": source,
                "target": target,
                "type": attrs.get("type", "related"),
                "strength": attrs.get("strength", 0.5)
            })
        
        return {
            "nodes": nodes,
            "links": links
        }
    
    def _add_event_relationships_to_graph(self, graph: nx.Graph, events: List[Dict[str, Any]]):
        """
        Add relationships between events to a graph.
        
        Args:
            graph: NetworkX graph
            events: List of events
        """
        if len(events) <= 1:
            return
        
        # Create a map of event_id to index for quick lookup
        event_map = {event["event_id"]: i for i, event in enumerate(events)}
        
        for i, event1 in enumerate(events):
            event1_id = event1["event_id"]
            event1_participants = set(event1.get("participants", []))
            event1_location = event1.get("location", "")
            
            for j, event2 in enumerate(events[i+1:], i+1):
                # Skip if already processed or if events are the same
                if i == j or event1_id == event2["event_id"]:
                    continue
                
                event2_id = event2["event_id"]
                
                # Skip linking if events are very far apart in time
                if isinstance(event1.get("last_mentioned"), datetime) and \
                   isinstance(event2.get("last_mentioned"), datetime):
                    time_diff = abs((event1["last_mentioned"] - event2["last_mentioned"]).days)
                    if time_diff > 60:  # Skip if more than 60 days apart
                        continue
                
                # Determine if and how the events are related
                link_type = None
                link_strength = 0
                
                # 1. Check for shared participants
                event2_participants = set(event2.get("participants", []))
                shared_participants = event1_participants.intersection(event2_participants)
                
                if shared_participants:
                    link_type = "shared_participants"
                    link_strength = min(0.8, 0.5 + (len(shared_participants) * 0.1))
                
                # 2. Check for shared location
                event2_location = event2.get("location", "")
                if event1_location and event2_location and event1_location == event2_location:
                    if link_type:
                        link_type = "shared_participants_and_location"
                        link_strength = min(0.9, link_strength + 0.1)
                    else:
                        link_type = "shared_location"
                        link_strength = 0.7
                
                # 3. Check for semantic similarity
                if ("embedding" in event1 and "embedding" in event2 and 
                    (not link_type or link_strength < 0.7)):
                    try:
                        # Calculate embedding similarity
                        similarity = np.dot(np.array(event1["embedding"]), np.array(event2["embedding"])) / (
                            np.linalg.norm(np.array(event1["embedding"])) * 
                            np.linalg.norm(np.array(event2["embedding"]))
                        )
                        
                        if similarity > 0.7:  # High semantic similarity
                            if link_type:
                                # Semantic similarity confirms existing relationship
                                link_strength = max(link_strength, float(similarity))
                            else:
                                link_type = "semantic"
                                link_strength = float(similarity)
                    except Exception as e:
                        logger.error(f"Error calculating embedding similarity: {e}")
                
                # 4. Check for narrative connection
                if ("narrative_ids" in event1 and "narrative_ids" in event2 and 
                    (not link_type or link_strength < 0.8)):
                    event1_narratives = set(event1["narrative_ids"])
                    event2_narratives = set(event2["narrative_ids"])
                    
                    shared_narratives = event1_narratives.intersection(event2_narratives)
                    if shared_narratives:
                        if link_type:
                            link_type = f"{link_type}_and_narrative"
                            link_strength = max(link_strength, 0.8)
                        else:
                            link_type = "shared_narrative"
                            link_strength = 0.8
                
                # Add edge if events are related
                if link_type and link_strength > 0:
                    graph.add_edge(
                        event1_id,
                        event2_id,
                        type=link_type,
                        strength=link_strength
                    )
    
    def get_entity_relationships(self, user_id: str, min_strength: int = 1) -> Dict[str, Any]:
        """
        Get relationships between entities for visualization.
        
        Args:
            user_id: User identifier
            min_strength: Minimum relationship strength
            
        Returns:
            Entity relationship network
        """
        return self.mongo.get_entity_relationships(user_id, min_strength)
    
    def get_related_events(self, user_id: str, event_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get events related to a specific event.
        
        Args:
            user_id: User identifier
            event_id: Event identifier
            limit: Maximum number of related events to return
            
        Returns:
            List of related events
        """
        return self.mongo.find_related_events(user_id, event_id, limit)
    
    def get_emotional_trends(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get emotional trends over time.
        
        Args:
            user_id: User identifier
            days: Number of days to analyze
            
        Returns:
            Emotional trends analysis
        """
        return self.mongo.get_emotional_trends(user_id, days)
    
    def get_narratives(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get narrative arcs and story structures.
        
        Args:
            user_id: User identifier
            limit: Maximum number of narratives to return
            
        Returns:
            List of narratives
        """
        return self.mongo.get_narratives(user_id, limit=limit)
    
    def get_narrative(self, user_id: str, narrative_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific narrative by ID.
        
        Args:
            user_id: User identifier
            narrative_id: Narrative identifier
            
        Returns:
            Narrative data or None if not found
        """
        narrative = self.mongo.narratives.find_one({
            "user_id": user_id,
            "narrative_id": narrative_id
        })
        
        if not narrative:
            return None
        
        # Convert ObjectId to string
        narrative["_id"] = str(narrative["_id"])
        
        # Get related events
        related_events = []
        for event_id in narrative.get("events", []):
            event = self.mongo.get_event(event_id)
            if event:
                related_events.append({
                    "event_id": event["event_id"],
                    "name": event["name"],
                    "status": event["status"],
                    "date": event.get("last_mentioned")
                })
        
        narrative["related_events"] = related_events
        
        return narrative
    
    def get_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Generate insights based on journal data.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with various insights
        """
        return self.mongo.generate_insights(user_id)
    
    def search(self, user_id: str, query: str, search_type: str = "semantic") -> List[Dict[str, Any]]:
        """
        Search journal entries with advanced options.
        
        Args:
            user_id: User identifier
            query: Search query
            search_type: Type of search (semantic, text, or combined)
            
        Returns:
            List of search results
        """
        if search_type == "text":
            return self.mongo.text_search(user_id, query)
        elif search_type == "semantic":
            # Generate embedding for the query
            embedding = self.nlp_engine._generate_embedding(query)
            return self.mongo.semantic_search(user_id, query, embedding)
        else:  # Combined search
            embedding = self.nlp_engine._generate_embedding(query)
            
            # Get results from both methods
            text_results = self.mongo.text_search(user_id, query)
            semantic_results = self.mongo.semantic_search(user_id, query, embedding)
            
            # Combine and deduplicate
            combined = {}
            
            for result in text_results:
                entry_id = result["entry_id"]
                combined[entry_id] = result
                combined[entry_id]["text_score"] = result["relevance"]
                combined[entry_id]["semantic_score"] = 0
            
            for result in semantic_results:
                entry_id = result["entry_id"]
                if entry_id in combined:
                    combined[entry_id]["semantic_score"] = result["relevance"]
                    # Use the maximum of the two scores
                    combined[entry_id]["relevance"] = max(
                        combined[entry_id]["text_score"],
                        result["relevance"]
                    )
                else:
                    combined[entry_id] = result
                    combined[entry_id]["semantic_score"] = result["relevance"]
                    combined[entry_id]["text_score"] = 0
            
            # Convert to list and sort by relevance
            results = list(combined.values())
            results.sort(key=lambda x: x["relevance"], reverse=True)
            
            return results