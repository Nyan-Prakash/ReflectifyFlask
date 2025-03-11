# flask_reflectify.py
"""
Reflectify - An advanced journal app with sophisticated narrative understanding

This enhanced version incorporates advanced narrative understanding capabilities,
deep context awareness, relationship tracking, and comprehensive emotional intelligence.

Usage: python flask_reflectify.py
"""

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import check_password_hash
import pymongo
import datetime
import json
import uuid
import numpy as np
import pandas as pd
import networkx as nx
import re
import time
import base64
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO
import bcrypt
import torch
from sentence_transformers import SentenceTransformer, util
import spacy
import os
import tempfile
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import logging
import io
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg for non-interactive environments

# Import our enhanced components
from AdvancedNLPEngine import AdvancedNLPEngine
from MongoDBClient import MongoDBClient
from ContextAwareNarrativeTracker import ContextAwareNarrativeTracker  # Updated from ContextAwareEventTracker


# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Helper Functions
# =============================================================================

def calculate_streak(dates):
    """Calculate the current streak of consecutive days with entries."""
    if not dates:
        return 0
    
    # Extract just the dates (no time)
    date_set = set(d.date() for d in dates)
    
    # Get today's date
    today = datetime.now().date()
    
    # Check if there's an entry today
    if today not in date_set:
        return 0
    
    # Count consecutive days backwards from today
    streak = 1
    current_date = today - timedelta(days=1)
    
    while current_date in date_set:
        streak += 1
        current_date = current_date - timedelta(days=1)
    
    return streak

def format_date(date_obj):
    """Format date for display."""
    if isinstance(date_obj, str):
        try:
            date_obj = datetime.fromisoformat(date_obj.replace('Z', '+00:00'))
        except:
            return date_obj
            
    if isinstance(date_obj, datetime):
        return date_obj.strftime('%B %d, %Y')
    
    return "Unknown date"

def format_time_ago(date_obj):
    """Format date as time ago (e.g., '2 days ago')."""
    if not isinstance(date_obj, datetime):
        try:
            date_obj = datetime.fromisoformat(str(date_obj).replace('Z', '+00:00'))
        except:
            return "Unknown time"
    
    now = datetime.now()
    diff = now - date_obj
    
    if diff.days == 0:
        hours = diff.seconds // 3600
        if hours == 0:
            minutes = diff.seconds // 60
            if minutes == 0:
                return "Just now"
            elif minutes == 1:
                return "1 minute ago"
            else:
                return f"{minutes} minutes ago"
        elif hours == 1:
            return "1 hour ago"
        else:
            return f"{hours} hours ago"
    elif diff.days == 1:
        return "Yesterday"
    elif diff.days < 7:
        return f"{diff.days} days ago"
    elif diff.days < 30:
        weeks = diff.days // 7
        if weeks == 1:
            return "1 week ago"
        else:
            return f"{weeks} weeks ago"
    elif diff.days < 365:
        months = diff.days // 30
        if months == 1:
            return "1 month ago"
        else:
            return f"{months} months ago"
    else:
        years = diff.days // 365
        if years == 1:
            return "1 year ago"
        else:
            return f"{years} years ago"

def generate_plot_base64(fig):
    """Generate base64 encoded image from matplotlib figure."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def extract_similarity_components(new_event, existing_event):
    """Extract the components that contributed to similarity matching."""
    components = {}
    
    # Extract participants that matched
    new_participants = set(new_event.get("components", {}).get("participants", []))
    existing_participants = set(existing_event.get("participants", []))
    
    if new_participants and existing_participants:
        common_participants = list(new_participants.intersection(existing_participants))
        if common_participants:
            components["participants_match"] = common_participants
    
    # Extract location match
    new_location = new_event.get("components", {}).get("location", "").lower()
    existing_location = existing_event.get("location", "").lower()
    
    if new_location and existing_location and (new_location == existing_location or 
                                              new_location in existing_location or 
                                              existing_location in new_location):
        components["location_match"] = existing_location
    
    # Extract time references
    new_times = set(new_event.get("components", {}).get("time", []))
    existing_times = set(existing_event.get("time_mentions", []))
    
    if new_times and existing_times:
        common_times = list(new_times.intersection(existing_times))
        if common_times:
            components["time_match"] = common_times
    
    # Check if semantic similarity contributed
    if "embedding" in existing_event:
        components["semantic_match"] = True
    
    # Check if narrative context contributed
    if "narrative_ids" in existing_event:
        components["narrative_context"] = existing_event.get("narrative_ids", [])
    
    return components

def generate_network_graph(network_data, graph_type="event"):
    """Generate a network graph visualization."""
    G = nx.Graph()
    
    # Add nodes
    for node in network_data["nodes"]:
        G.add_node(node["id"], **node)
    
    # Add edges
    for link in network_data["links"]:
        G.add_edge(link["source"], link["target"], **link)
    
    # Generate positions
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    
    # Define color maps
    event_color_map = {
        "WORK": "#4285F4",
        "SOCIAL": "#EA4335",
        "HEALTH": "#34A853",
        "TRAVEL": "#FBBC05",
        "MILESTONE": "#9C27B0",
        "EDUCATION": "#FF6D01",
        "CREATIVE": "#00BCD4", 
        "PERSONAL": "#9E9E9E",
        "OTHER": "#7F7F7F"
    }
    
    entity_color_map = {
        "PERSON": "#4285F4",
        "PERSON_FAMILY": "#0F9D58",  # Dark green for family
        "PERSON_FRIEND": "#F4B400",  # Yellow for friends
        "PERSON_PROFESSIONAL": "#DB4437",  # Red for work contacts
        "LOCATION": "#34A853",
        "ORG": "#EA4335",
        "GPE": "#FBBC05",
        "FAC": "#9C27B0",
        "OTHER": "#7F7F7F"
    }
    
    color_map = event_color_map if graph_type == "event" else entity_color_map
    
    # Draw nodes by type
    for node_type in set(nx.get_node_attributes(G, 'type').values()):
        nodelist = [n for n, attr in G.nodes(data=True) if attr.get("type") == node_type]
        if nodelist:
            if graph_type == "entity":
                # Node size based on number of events or mentions for entity graph
                node_sizes = [300 + 100 * min(G.nodes[n].get("mentions", 1), 5) for n in nodelist]
            else:
                # Node size based on importance for event graph
                node_sizes = []
                for n in nodelist:
                    importance = G.nodes[n].get("importance", "MEDIUM")
                    if importance == "HIGH":
                        node_sizes.append(700)
                    elif importance == "MEDIUM":
                        node_sizes.append(500)
                    else:
                        node_sizes.append(300)
                
            nx.draw_networkx_nodes(
                G,
                pos,
                node_color=color_map.get(node_type, "#7F7F7F"),
                node_size=node_sizes,
                alpha=0.8,
                nodelist=nodelist
            )
    
    # Draw edges with different styles based on relationship type
    if graph_type == "event":
        # Different edge styles for event graph
        edge_types = set(nx.get_edge_attributes(G, 'type').values())
        
        for edge_type in edge_types:
            edgelist = [(u, v) for u, v, attr in G.edges(data=True) 
                       if attr.get('type') == edge_type]
            
            if edgelist:
                if "semantic" in edge_type:
                    # Dotted line for semantic connections
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=edgelist,
                        width=1.0,
                        alpha=0.6,
                        edge_color="#999999",
                        style="dotted"
                    )
                elif "participant" in edge_type:
                    # Solid line for shared participants
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=edgelist,
                        width=1.5,
                        alpha=0.7,
                        edge_color="#333333"
                    )
                elif "location" in edge_type:
                    # Dashed line for shared location
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=edgelist,
                        width=1.0,
                        alpha=0.7,
                        edge_color="#666666",
                        style="dashed"
                    )
                elif "narrative" in edge_type:
                    # Thick solid line for narrative connections
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=edgelist,
                        width=2.0,
                        alpha=0.8,
                        edge_color="#9C27B0"  # Purple for narrative connections
                    )
                else:
                    # Default edge style
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=edgelist,
                        width=1.0,
                        alpha=0.5,
                        edge_color="gray"
                    )
    else:
        # Different edge styles for entity graph
        edge_types = set(nx.get_edge_attributes(G, 'type').values())
        
        for edge_type in edge_types:
            edgelist = [(u, v) for u, v, attr in G.edges(data=True) 
                       if attr.get('type') == edge_type]
            
            if not edgelist:
                continue
                
            # Get weights for line thickness
            weights = [G.edges[u, v].get("strength", 1) for u, v in edgelist]
            widths = [max(0.5, min(3.0, w)) for w in weights]  # Scale between 0.5 and 3.0
            
            # Set different colors and styles based on relationship type
            if edge_type == "FAMILY":
                nx.draw_networkx_edges(
                    G, pos, edgelist=edgelist, width=widths, 
                    alpha=0.7, edge_color="#0F9D58", style="solid"  # Dark green for family
                )
            elif edge_type == "ROMANTIC":
                nx.draw_networkx_edges(
                    G, pos, edgelist=edgelist, width=widths, 
                    alpha=0.7, edge_color="#E91E63", style="solid"  # Pink for romantic
                )
            elif edge_type == "FRIENDSHIP":
                nx.draw_networkx_edges(
                    G, pos, edgelist=edgelist, width=widths, 
                    alpha=0.7, edge_color="#F4B400", style="solid"  # Yellow for friendship
                )
            elif edge_type == "PROFESSIONAL":
                nx.draw_networkx_edges(
                    G, pos, edgelist=edgelist, width=widths, 
                    alpha=0.7, edge_color="#DB4437", style="solid"  # Red for professional
                )
            elif edge_type == "LOCATION":
                nx.draw_networkx_edges(
                    G, pos, edgelist=edgelist, width=widths, 
                    alpha=0.7, edge_color="#34A853", style="dashed"  # Green dashed for location
                )
            else:
                nx.draw_networkx_edges(
                    G, pos, edgelist=edgelist, width=widths, 
                    alpha=0.5, edge_color="#999999", style="solid"  # Grey for other
                )
    
    # Draw labels
    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: G.nodes[n]["name"][:15] + "..." if len(G.nodes[n]["name"]) > 15 else G.nodes[n]["name"] for n in G.nodes()},
        font_size=8,
        font_family="sans-serif"
    )
    
    plt.axis("off")
    plt.tight_layout()
    
    # Convert to base64
    return generate_plot_base64(fig)

def generate_emotional_arc_chart(emotional_data, width=10, height=6):
    """Generate a chart visualizing emotional arcs over time."""
    if not emotional_data:
        return None
    
    # Create dataframe from emotional data
    df = pd.DataFrame(emotional_data)
    
    # Ensure we have date and primary_emotion columns
    if 'date' not in df.columns or 'primary_emotion' not in df.columns:
        return None
    
    # Sort by date
    df = df.sort_values('date')
    
    # Get unique emotions
    emotions = df['primary_emotion'].unique()
    
    # Create emotion map for plotting
    emotion_values = {
        'JOY': 1.0,
        'HAPPINESS': 1.0,
        'CONTENTMENT': 0.8,
        'TRUST': 0.7,
        'ANTICIPATION': 0.6,
        'SURPRISE': 0.4,
        'FEAR': -0.6,
        'ANGER': -0.8,
        'DISGUST': -0.9,
        'SADNESS': -1.0
    }
    
    # Map emotions to values (defaulting to 0 for unknown emotions)
    df['emotion_value'] = df['primary_emotion'].map(lambda x: emotion_values.get(x, 0))
    
    # Map emotions to colors
    emotion_colors = {
        'JOY': '#FFC107',  # Amber
        'HAPPINESS': '#FFEB3B',  # Yellow
        'CONTENTMENT': '#CDDC39',  # Lime
        'TRUST': '#8BC34A',  # Light Green
        'ANTICIPATION': '#4CAF50',  # Green
        'SURPRISE': '#009688',  # Teal
        'FEAR': '#7986CB',  # Indigo
        'ANGER': '#F44336',  # Red
        'DISGUST': '#9C27B0',  # Purple
        'SADNESS': '#673AB7'   # Deep Purple
    }
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Plot each emotion data point
    for emotion in emotions:
        emotion_df = df[df['primary_emotion'] == emotion]
        if len(emotion_df) > 0:
            color = emotion_colors.get(emotion, '#9E9E9E')  # Default to grey for unknown emotions
            ax.scatter(
                emotion_df['date'], 
                emotion_df['emotion_value'],
                color=color,
                label=emotion,
                alpha=0.7,
                s=50  # Point size
            )
    
    # Add a trend line if enough data points
    if len(df) > 2:
        try:
            from scipy import stats
            import matplotlib.dates as mdates
            
            # Convert dates to numbers for regression
            x = mdates.date2num(df['date'])
            y = df['emotion_value']
            
            # Calculate trend line
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Create line
            line_x = np.array([min(x), max(x)])
            line_y = intercept + slope * line_x
            
            # Convert back to datetime for plotting
            line_x_dates = mdates.num2date(line_x)
            
            # Plot trend line
            ax.plot(line_x_dates, line_y, 'k--', alpha=0.7)
            
            # Add annotation about trend
            trend_direction = "improving" if slope > 0 else "declining" if slope < 0 else "stable"
            ax.annotate(
                f"Emotional trend: {trend_direction}",
                xy=(0.05, 0.95), 
                xycoords='axes fraction',
                fontsize=10, 
                ha='left', 
                va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )
        except Exception as e:
            logger.error(f"Error generating trend line: {e}")
    
    # Add horizontal lines for reference
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add labels for the emotional regions
    ax.text(df['date'].iloc[0], 0.8, "Positive Emotions", fontsize=9, ha='left', va='center', alpha=0.7)
    ax.text(df['date'].iloc[0], -0.8, "Negative Emotions", fontsize=9, ha='left', va='center', alpha=0.7)
    
    # Set y-axis limits
    ax.set_ylim(-1.2, 1.2)
    
    # Format x-axis
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    # Add labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Emotional Valence")
    ax.set_title("Emotional Arc Over Time")
    
    # Add legend if we have multiple emotions
    if len(emotions) > 1:
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    return generate_plot_base64(fig)

def generate_narrative_graph(narrative_data, width=10, height=8):
    """Generate a graph visualization of narrative structure and connections."""
    if not narrative_data:
        return None
    
    G = nx.Graph()
    
    # Add narrative node
    G.add_node(
        narrative_data["narrative_id"],
        name=f"Narrative: {narrative_data.get('narrative_arc', 'Story')}",
        type="NARRATIVE",
        size=1000
    )
    
    # Add event nodes
    for event in narrative_data.get("related_events", []):
        G.add_node(
            event["event_id"],
            name=event["name"],
            type="EVENT",
            status=event.get("status", "unknown"),
            size=500
        )
        
        # Connect event to narrative
        G.add_edge(
            narrative_data["narrative_id"],
            event["event_id"],
            type="NARRATIVE_EVENT"
        )
    
    # Add entity nodes for narrative roles
    for role, entities in narrative_data.get("narrative_roles", {}).items():
        for entity in entities:
            entity_id = hashlib.md5(entity.lower().encode()).hexdigest()
            G.add_node(
                entity_id,
                name=entity,
                type="ENTITY",
                role=role,
                size=300
            )
            
            # Connect entity to narrative
            G.add_edge(
                narrative_data["narrative_id"],
                entity_id,
                type="NARRATIVE_ROLE",
                role=role
            )
    
    # Add connected narratives
    for connection in narrative_data.get("connected_narratives", []):
        connected_id = connection.get("narrative_id")
        if connected_id:
            G.add_node(
                connected_id,
                name=f"Related Narrative",
                type="CONNECTED_NARRATIVE",
                size=800
            )
            
            # Connect narratives
            G.add_edge(
                narrative_data["narrative_id"],
                connected_id,
                type="NARRATIVE_CONNECTION",
                strength=connection.get("strength", 0.5)
            )
    
    # Create plot
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Generate positions - emphasize the central narrative
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Define node colors
    color_map = {
        "NARRATIVE": "#9C27B0",  # Purple
        "CONNECTED_NARRATIVE": "#7B1FA2",  # Darker purple
        "EVENT": "#2196F3",  # Blue
        "ENTITY": "#FF9800"   # Orange
    }
    
    # Draw nodes by type
    for node_type in ["NARRATIVE", "CONNECTED_NARRATIVE", "EVENT", "ENTITY"]:
        nodelist = [n for n, attr in G.nodes(data=True) if attr.get("type") == node_type]
        
        if not nodelist:
            continue
            
        # Get sizes
        node_sizes = [G.nodes[n].get("size", 300) for n in nodelist]
        
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            node_color=color_map.get(node_type, "#7F7F7F"),
            node_size=node_sizes,
            alpha=0.8
        )
    
    # Draw edges by type
    edge_types = set(nx.get_edge_attributes(G, 'type').values())
    
    for edge_type in edge_types:
        edgelist = [(u, v) for u, v, attr in G.edges(data=True) 
                   if attr.get('type') == edge_type]
        
        if not edgelist:
            continue
            
        if edge_type == "NARRATIVE_EVENT":
            # Solid blue line for narrative to event connections
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edgelist,
                width=1.5,
                alpha=0.7,
                edge_color="#2196F3"
            )
        elif edge_type == "NARRATIVE_ROLE":
            # Dashed orange line for narrative to entity connections
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edgelist,
                width=1.5,
                alpha=0.7,
                edge_color="#FF9800",
                style="dashed"
            )
        elif edge_type == "NARRATIVE_CONNECTION":
            # Thick purple line for narrative connections
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edgelist,
                width=2.0,
                alpha=0.8,
                edge_color="#9C27B0"
            )
        else:
            # Default edge style
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edgelist,
                width=1.0,
                alpha=0.5,
                edge_color="gray"
            )
    
    # Draw labels
    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: G.nodes[n]["name"][:15] + "..." if len(G.nodes[n]["name"]) > 15 else G.nodes[n]["name"] for n in G.nodes()},
        font_size=8,
        font_family="sans-serif"
    )
    
    plt.axis("off")
    plt.tight_layout()
    
    return generate_plot_base64(fig)

# =============================================================================
# Flask Web Application
# =============================================================================

app = Flask(__name__, template_folder='templates')
app.secret_key = 'reflectify_secret_key'  # Change this in production!

# Initialize global objects
mongo_client = MongoDBClient()  # Enhanced MongoDB client
nlp_engine = AdvancedNLPEngine()  # Enhanced NLP engine
narrative_tracker = ContextAwareNarrativeTracker(nlp_engine, mongo_client)  # Enhanced narrative tracker

@app.route('/')
def home():
    """Home route - redirects to login or journal page."""
    if 'user_id' in session:
        return redirect(url_for('journal'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username and password:
            user_id = mongo_client.verify_user(username, password)
            if user_id:
                session['user_id'] = user_id
                session['username'] = username
                flash('Login successful!', 'success')
                return redirect(url_for('journal'))
            else:
                flash('Invalid username or password', 'danger')
        else:
            flash('Please enter both username and password', 'warning')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if username and password and confirm_password:
            if password != confirm_password:
                flash('Passwords don\'t match', 'danger')
            else:
                user_id = mongo_client.add_user(username, password)
                if user_id:
                    flash('Registration successful! You can now log in.', 'success')
                    return redirect(url_for('login'))
                else:
                    flash('Username already exists', 'danger')
        else:
            flash('Please fill in all fields', 'warning')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout route."""
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/journal')
def journal():
    """Journal page with enhanced narrative understanding."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get last 5 entries
    entries = mongo_client.get_journal_entries(session['user_id'], limit=5)
    
    # Get ongoing narratives
    active_narratives = mongo_client.get_narratives(session['user_id'], limit=3)
    
    # Get recent events
    recent_events = narrative_tracker.get_recent_events(session['user_id'])
    
    # Get emotional trends
    emotional_data = mongo_client.get_emotional_trends(session['user_id'], days=14)
    
    return render_template('journal.html', 
                          entries=entries, 
                          events=recent_events,
                          active_narratives=active_narratives,
                          emotional_data=emotional_data,
                          format_date=format_date,
                          format_time_ago=format_time_ago, datetime=datetime)

@app.route('/add_journal_entry', methods=['POST'])
def add_journal_entry():
    """Add a new journal entry with comprehensive narrative processing."""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"})
    
    text = request.form.get('text')
    if not text:
        return jsonify({"success": False, "message": "No entry text provided"})
    
    # Process the entry with our advanced narrative tracker
    entry_date = datetime.now()
    entry_id = str(uuid.uuid4())
    
    processing_result = narrative_tracker.process_journal_entry(
        user_id=session['user_id'],
        entry_text=text,
        entry_id=entry_id,
        entry_date=entry_date
    )
    
    # Extract the processed events
    processed_events = processing_result.get("events", [])
    
    # Extract entities
    processed_entities = processing_result.get("entities", [])
    
    # Extract narrative analysis
    narrative_analysis = processing_result.get("narrative_analysis", {})
    
    # Extract emotional analysis
    emotional_analysis = processing_result.get("emotional_analysis", {})
    
    # Prepare event data for response
    event_data = []
    for event in processed_events:
        event_data.append({
            "event_id": event.get("event_id", ""),
            "name": event.get("name", ""),
            "type": event.get("event_type", "OTHER"),
            "status": event.get("status", "ongoing"),
            "is_new": event.get("is_new", True),
            "description": event.get("description", ""),
            "location": event.get("location", ""),
            "participants": event.get("participants", []),
            "metadata": event.get("metadata", {})
        })
    
    # Prepare entity data for response
    entity_data = []
    for entity in processed_entities[:5]:  # Limit to top 5 entities
        entity_data.append({
            "entity_id": entity.get("entity_id", ""),
            "text": entity.get("text", ""),
            "type": entity.get("type", "UNKNOWN"),
            "is_new": entity.get("is_new", True),
            "mention_count": entity.get("mention_count", 1)
        })
    
    # Prepare narrative data
    narrative_data = {
        "narrative_id": narrative_analysis.get("narrative_id", ""),
        "narrative_arc": narrative_analysis.get("narrative_arc", ""),
        "connected_narratives": narrative_analysis.get("connected_narratives", [])
    }
    
    # Prepare emotional data
    emotion_data = {
        "primary_emotion": emotional_analysis.get("primary_emotion", ""),
        "intensity": emotional_analysis.get("intensity", 0.0),
        "emotional_continuity": emotional_analysis.get("emotional_continuity", {})
    }
    
    return jsonify({
        "success": True,
        "message": "Journal entry saved with enhanced narrative understanding",
        "entry_id": entry_id,
        "events": event_data,
        "entities": entity_data,
        "narrative": narrative_data,
        "emotion": emotion_data
    })

@app.route('/events')
def events():
    """Enhanced events page with narrative understanding."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get filter parameters
    status_filter = request.args.get('status', 'All')
    type_filter = request.args.get('type', 'All')
    timeframe = request.args.get('timeframe', 'All Time')
    narrative_filter = request.args.get('narrative', 'All')
    
    # Get narratives for filtering
    narratives = mongo_client.get_narratives(session['user_id'])
    narrative_options = [{"id": n.get("narrative_id", ""), "name": n.get("narrative_arc", "Story")} for n in narratives]
    
    # Get all events
    all_events = mongo_client.get_events(session['user_id'])
    
    # Apply filters
    filtered_events = all_events
    
    if status_filter != "All":
        status = status_filter.lower()
        filtered_events = [e for e in filtered_events if e.get("status", "").lower() == status]
    
    if type_filter != "All":
        filtered_events = [e for e in filtered_events if e.get("event_type", "") == type_filter]
    
    if timeframe != "All Time":
        today = datetime.now()
        if timeframe == "Last 30 Days":
            cutoff = today - timedelta(days=30)
        elif timeframe == "Last 90 Days":
            cutoff = today - timedelta(days=90)
        elif timeframe == "This Year":
            cutoff = datetime(today.year, 1, 1)
        
        filtered_events = [e for e in filtered_events 
                          if isinstance(e.get("last_mentioned"), datetime) and 
                          e["last_mentioned"] >= cutoff]
    
    if narrative_filter != "All":
        # Filter events by narrative
        filtered_events = [e for e in filtered_events 
                          if "narrative_ids" in e and 
                          narrative_filter in e["narrative_ids"]]
    
    # Add similarity data to events for display
    for event in filtered_events:
        # Calculate similarity information to display
        similarity_data = {}
        
        # Check for mentions to determine if this is an event recognized across multiple entries
        if "mentions" in event and len(event["mentions"]) > 1:
            similarity_data["match_count"] = len(event["mentions"])
            
            # Check for shared participants
            if "participants" in event and len(event["participants"]) > 0:
                similarity_data["participants_match"] = event["participants"]
            
            # Check for location
            if "location" in event and event["location"]:
                similarity_data["location_match"] = event["location"]
                
            # Check for time references
            if "time_mentions" in event and len(event["time_mentions"]) > 0:
                similarity_data["time_match"] = True
                
            # Assume semantic matching if the event has multiple mentions
            similarity_data["semantic_match"] = True
            
            # Check for narrative connections
            if "narrative_ids" in event and event["narrative_ids"]:
                similarity_data["narrative_connections"] = event["narrative_ids"]
        
        # Add narrative connections
        if "narrative_ids" in event:
            # Get narrative names
            narrative_names = []
            for narrative_id in event.get("narrative_ids", []):
                for n in narratives:
                    if n.get("narrative_id") == narrative_id:
                        narrative_names.append(n.get("narrative_arc", "Story"))
                        break
            
            if narrative_names:
                similarity_data["narratives"] = narrative_names
        
        event["similarity_data"] = similarity_data if similarity_data else None
    
    # Group events by type
    events_by_type = {}
    for event in filtered_events:
        event_type = event.get("event_type", "OTHER")
        if event_type not in events_by_type:
            events_by_type[event_type] = []
        events_by_type[event_type].append(event)
    
    # Color map for event types
    color_map = {
        "WORK": "#4285F4",
        "SOCIAL": "#EA4335", 
        "HEALTH": "#34A853",
        "TRAVEL": "#FBBC05",
        "MILESTONE": "#9C27B0",
        "EDUCATION": "#FF6D01",
        "CREATIVE": "#00BCD4",
        "PERSONAL": "#9E9E9E",
        "OTHER": "#7F7F7F"
    }
    
    return render_template('events.html', 
                          events=filtered_events,
                          events_by_type=events_by_type,
                          color_map=color_map,
                          format_date=format_date,
                          status_filter=status_filter,
                          type_filter=type_filter,
                          timeframe=timeframe,
                          narrative_filter=narrative_filter,
                          narrative_options=narrative_options)

@app.route('/event/<event_id>')
def event_detail(event_id):
    """Event detail page with narrative context."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get the event
    event = mongo_client.get_event(event_id)
    
    if not event or event.get("user_id") != session['user_id']:
        flash('Event not found', 'warning')
        return redirect(url_for('events'))
    
    # Get event timeline/mentions
    timeline = narrative_tracker.get_event_timeline(session['user_id'], event_id)
    
    # Get related events
    related_events = narrative_tracker.get_related_events(session['user_id'], event_id)
    
    # Get narrative information if event is part of narratives
    narratives = []
    if "narrative_ids" in event:
        for narrative_id in event["narrative_ids"]:
            narrative = mongo_client.get_narrative(session['user_id'], narrative_id)
            if narrative:
                narratives.append(narrative)
    print(event)
    return render_template('event_detail.html',
                          event=event,
                          timeline=timeline,
                          related_events=related_events,
                          narratives=narratives,
                          format_date=format_date)

@app.route('/mark_complete/<event_id>', methods=['POST'])
def mark_complete(event_id):
    """Mark an event as completed."""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"})
    
    result = narrative_tracker.mark_event_completed(session['user_id'], event_id)
    return jsonify(result)

@app.route('/timeline')
def timeline():
    """Enhanced timeline page with narrative visualization."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get date range parameters
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')
    
    # Default to last 90 days if not specified
    today = datetime.now()
    if not end_date_str:
        end_date = today
    else:
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        except:
            end_date = today
    
    if not start_date_str:
        start_date = today - timedelta(days=90)
    else:
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        except:
            start_date = today - timedelta(days=90)
    
    # Convert to datetime
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    # Get journal entries within date range
    entries = mongo_client.get_journal_entries(
        session['user_id'], 
        limit=100,
        start_date=start_datetime,
        end_date=end_datetime
    )
    
    # Group entries by month and year
    entries_by_month = {}
    for entry in entries:
        entry_date = entry["date"]
        if isinstance(entry_date, str):
            try:
                entry_date = datetime.fromisoformat(entry_date.replace('Z', '+00:00'))
            except:
                # Handle potential date format issues
                continue
        
        month_year = entry_date.strftime("%B %Y")
        
        if month_year not in entries_by_month:
            entries_by_month[month_year] = []
            
        entries_by_month[month_year].append(entry)
    
    # Sort months chronologically
    sorted_months = sorted(
        entries_by_month.keys(),
        key=lambda x: datetime.strptime(x, "%B %Y"),
        reverse=True
    )
    
    # Get all events for entry-event mapping
    all_events = mongo_client.get_events(session['user_id'])
    
    # Create mapping of entry_id to events
    entry_events = {}
    for event in all_events:
        for mention in event.get("mentions", []):
            entry_id = mention.get("entry_id")
            if entry_id:
                if entry_id not in entry_events:
                    entry_events[entry_id] = []
                entry_events[entry_id].append(event)
    
    # Get emotional arcs for the time period
    emotional_arcs = list(mongo_client.emotional_arcs.find({
        "user_id": session['user_id'],
        "date": {"$gte": start_datetime, "$lte": end_datetime}
    }).sort("date", 1))
    
    # Generate emotional arc visualization if we have enough data
    emotional_arc_chart = None
    if len(emotional_arcs) >= 3:
        emotional_arc_chart = generate_emotional_arc_chart(emotional_arcs)
    
    # Get narratives active in this time period
    narratives = list(mongo_client.narratives.find({
        "user_id": session['user_id'],
        "date": {"$gte": start_datetime, "$lte": end_datetime}
    }).sort("date", 1))
    
    
    return render_template('timeline.html',
                          entries_by_month=entries_by_month,
                          sorted_months=sorted_months,
                          entry_events=entry_events,
                          start_date=start_date.strftime('%Y-%m-%d'),
                          end_date=end_date.strftime('%Y-%m-%d'),
                          format_date=format_date,
                          emotional_arc_chart=emotional_arc_chart,
                          narratives=narratives,
                          format_time_ago=format_time_ago)

@app.route('/narratives')
def narratives():
    """Narratives overview page."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get narratives
    all_narratives = mongo_client.get_narratives(session['user_id'])
    
    # Group narratives by arc type
    narratives_by_type = {}
    for narrative in all_narratives:
        arc_type = narrative.get("narrative_arc", "General")
        if arc_type not in narratives_by_type:
            narratives_by_type[arc_type] = []
        narratives_by_type[arc_type].append(narrative)
    
    # Get concepts and themes
    concepts = mongo_client.get_concepts(session['user_id'], min_mentions=2)
    
    # Get temporal sequences
    sequences = mongo_client.get_temporal_sequences(session['user_id'])
    
    return render_template('narratives.html',
                          narratives=all_narratives,
                          narratives_by_type=narratives_by_type,
                          concepts=concepts,
                          sequences=sequences,
                          format_date=format_date)

@app.route('/narrative/<narrative_id>')
def narrative_detail(narrative_id):
    """Narrative detail page."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get the narrative
    narrative = mongo_client.get_narrative(session['user_id'], narrative_id)
    
    if not narrative:
        flash('Narrative not found', 'warning')
        return redirect(url_for('narratives'))
    
    # Generate narrative graph visualization
    narrative_graph = generate_narrative_graph(narrative)
    
    # Get emotional arc for this narrative
    entry_id = narrative.get("entry_id")
    emotional_arc = None
    if entry_id:
        emotional_arc = mongo_client.get_emotional_arc_for_entry(session['user_id'], entry_id)
    
    return render_template('narrative_detail.html',
                          narrative=narrative,
                          narrative_graph=narrative_graph,
                          emotional_arc=emotional_arc,
                          format_date=format_date)

@app.route('/mindmap')
def mindmap():
    """Mind map visualization page with enhanced options."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Visualization type
    viz_type = request.args.get('type', 'event')
    
    if viz_type == 'event':
        # Get event network with narrative connections
        network = narrative_tracker.get_event_network(session['user_id'])
        graph_type = "event"
    elif viz_type == 'entity':
        # Get entity relationships
        min_strength = int(request.args.get('min_strength', '1'))
        network = mongo_client.get_entity_relationships(session['user_id'], min_strength)
        graph_type = "entity"
    elif viz_type == 'narrative':
        # Get narrative network - reconstructed from narratives and their connections
        narratives = mongo_client.get_narratives(session['user_id'])
        network = {"nodes": [], "links": []}
        
        # Add narrative nodes
        for narrative in narratives:
            network["nodes"].append({
                "id": narrative.get("narrative_id", ""),
                "name": narrative.get("narrative_arc", "Story"),
                "type": "NARRATIVE"
            })
            
            # Add connections between narratives
            for connection in narrative.get("connected_narratives", []):
                if "narrative_id" in connection:
                    network["links"].append({
                        "source": narrative.get("narrative_id", ""),
                        "target": connection.get("narrative_id", ""),
                        "type": connection.get("type", "connected"),
                        "strength": connection.get("strength", 0.5)
                    })
        
        graph_type = "narrative"
    else:
        # Default to entity view
        network = mongo_client.get_entity_relationships(session['user_id'])
        graph_type = "entity"
    
    graph_img = None
    if network["nodes"]:
        try:
            graph_img = generate_network_graph(network, graph_type)
        except Exception as e:
            logger.error(f"Error generating network graph: {e}")
    
    # Color maps for legend
    event_color_map = {
        "WORK": "#4285F4",
        "SOCIAL": "#EA4335",
        "HEALTH": "#34A853",
        "TRAVEL": "#FBBC05",
        "MILESTONE": "#9C27B0",
        "EDUCATION": "#FF6D01",
        "CREATIVE": "#00BCD4",
        "PERSONAL": "#9E9E9E",
        "OTHER": "#7F7F7F"
    }
    
    entity_color_map = {
        "PERSON": "#4285F4",
        "PERSON_FAMILY": "#0F9D58",  # Dark green for family
        "PERSON_FRIEND": "#F4B400",  # Yellow for friends
        "PERSON_PROFESSIONAL": "#DB4437",  # Red for work contacts
        "LOCATION": "#34A853",
        "ORG": "#EA4335",
        "GPE": "#FBBC05",
        "FAC": "#9C27B0",
        "OTHER": "#7F7F7F"
    }
    
    narrative_color_map = {
        "BEGINNING": "#4285F4",  # Blue
        "MIDDLE": "#34A853",     # Green
        "CLIMAX": "#FBBC05",     # Yellow
        "RESOLUTION": "#EA4335"  # Red
    }
    
    return render_template('mindmap.html',
                          graph_img=graph_img,
                          viz_type=viz_type,
                          event_color_map=event_color_map,
                          entity_color_map=entity_color_map,
                          narrative_color_map=narrative_color_map)

@app.route('/entities')
def entities():
    """Entities page with relationship data."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get filter parameter
    entity_type = request.args.get('type', 'All')
    min_mentions = int(request.args.get('min_mentions', '1'))
    
    # Get entities
    all_entities = mongo_client.get_entities(session['user_id'], min_mentions=min_mentions)
    
    # Apply filter
    if entity_type != "All":
        filtered_entities = [e for e in all_entities if e.get("type", "") == entity_type]
    else:
        filtered_entities = all_entities
    
    # Group entities by type
    entities_by_type = {}
    for entity in filtered_entities:
        entity_type = entity.get("type", "UNKNOWN")
        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = []
        entities_by_type[entity_type].append(entity)
    
    # Color map for entity types
    color_map = {
        "PERSON": "#4285F4",
        "PERSON_FAMILY": "#0F9D58",
        "PERSON_FRIEND": "#F4B400",
        "PERSON_PROFESSIONAL": "#DB4437",
        "LOCATION": "#34A853",
        "ORG": "#EA4335",
        "GPE": "#FBBC05",
        "FAC": "#9C27B0",
        "OTHER": "#7F7F7F"
    }
    
    return render_template('entities.html',
                          entities=filtered_entities,
                          entities_by_type=entities_by_type,
                          color_map=color_map,
                          entity_type=entity_type,
                          min_mentions=min_mentions,
                          format_date=format_date)

@app.route('/entity/<entity_id>')
def entity_detail(entity_id):
    """Entity detail page with relationship data."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get the entity
    entity = mongo_client.get_entity(session['user_id'], entity_id)
    
    if not entity:
        flash('Entity not found', 'warning')
        return redirect(url_for('entities'))
    
    # Get emotional mentions
    emotional_mentions = entity.get("emotional_mentions", [])
    
    # Group emotional mentions by emotion
    emotions_by_type = {}
    for mention in emotional_mentions:
        emotion = mention.get("emotion", "UNKNOWN")
        if emotion not in emotions_by_type:
            emotions_by_type[emotion] = []
        emotions_by_type[emotion].append(mention)
    
    return render_template('entity_detail.html',
                          entity=entity,
                          emotions_by_type=emotions_by_type,
                          format_date=format_date)

@app.route('/search')
def search():
    """Enhanced search page with semantic, text, and multi-faceted search."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    query = request.args.get('query', '')
    search_type = request.args.get('type', 'smart')
    
    results = []
    if query:
        if search_type == 'semantic' and nlp_engine.embedding_model:
            # Generate embedding for query
            query_embedding = nlp_engine.embedding_model.encode(query).tolist()
            
            # Perform semantic search
            results = mongo_client.semantic_search(
                session['user_id'],
                query,
                query_embedding,
                limit=10,
                threshold=0.5  # Lower threshold for more results
            )
        elif search_type == 'text':
            # Text search
            results = mongo_client.text_search(
                session['user_id'],
                query,
                limit=10
            )
        else:  # smart/combined search
            # Generate embedding for query
            query_embedding = None
            if nlp_engine.embedding_model:
                query_embedding = nlp_engine.embedding_model.encode(query).tolist()
            
            # Create multi-facet query
            faceted_query = {
                "text": query,
                "dates": {}  # No date filter by default
            }
            
            # Check for emotional terms in query
            emotion_terms = {
                "happy": "JOY", "joy": "JOY", "excited": "JOY", 
                "sad": "SADNESS", "sadness": "SADNESS", "depressed": "SADNESS",
                "angry": "ANGER", "anger": "ANGER", "frustrated": "ANGER",
                "fear": "FEAR", "afraid": "FEAR", "scared": "FEAR",
                "surprised": "SURPRISE", "surprise": "SURPRISE", "shocked": "SURPRISE",
                "disgusted": "DISGUST", "disgust": "DISGUST", "repulsed": "DISGUST"
            }
            
            emotions = []
            for term, emotion in emotion_terms.items():
                if term in query.lower() and emotion not in emotions:
                    emotions.append(emotion)
            
            if emotions:
                faceted_query["emotions"] = emotions
            
            # Perform multi-faceted search
            results = mongo_client.multi_facet_search(
                session['user_id'],
                faceted_query,
                query_embedding,
                limit=10
            )
    
    # Get all events for entry-event mapping
    all_events = mongo_client.get_events(session['user_id'])
    
    # Create mapping of entry_id to events
    entry_events = {}
    for event in all_events:
        for mention in event.get("mentions", []):
            entry_id = mention.get("entry_id")
            if entry_id:
                if entry_id not in entry_events:
                    entry_events[entry_id] = []
                entry_events[entry_id].append(event)
    
    # Get all entities for enriching results
    all_entities = mongo_client.get_entities(session['user_id'])
    
    # Create mapping of entry_id to entities
    entry_entities = {}
    for entity in all_entities:
        for entry_id in entity.get("entry_ids", []):
            if entry_id not in entry_entities:
                entry_entities[entry_id] = []
            entry_entities[entry_id].append(entity)
    
    return render_template('search.html',
                          query=query,
                          search_type=search_type,
                          results=results,
                          entry_events=entry_events,
                          entry_entities=entry_entities,
                          format_date=format_date)

@app.route('/insights')
def insights():
    """Enhanced insights page with narrative understanding."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get all entries for the user
    entries = mongo_client.get_journal_entries(
        session['user_id'],
        limit=500  # Get more entries for better analysis
    )
    
    if not entries:
        return render_template('insights.html', has_data=False)
    
    # Get insights from MongoDB client
    insights_data = mongo_client.generate_insights(session['user_id'])
    
    # Process entries for analysis
    entry_dates = []
    sentiments = []
    word_counts = []
    topics = []
    
    for entry in entries:
        # Process date
        entry_date = entry.get("date")
        if isinstance(entry_date, str):
            try:
                entry_date = datetime.fromisoformat(entry_date.replace('Z', '+00:00'))
            except:
                continue
        
        if not isinstance(entry_date, datetime):
            continue
            
        entry_dates.append(entry_date)
        
        # Process sentiment
        sentiment = None
        if "nlp_data" in entry and "sentiment" in entry["nlp_data"].get("core", {}):
            sentiment = entry["nlp_data"]["core"]["sentiment"]
            sentiments.append({
                "date": entry_date,
                "label": sentiment["label"],
                "score": sentiment["score"]
            })
        
        # Process word count
        text = entry.get("text", "")
        word_counts.append({
            "date": entry_date,
            "count": len(text.split())
        })
        
        # Process topics
        if "nlp_data" in entry and "topics" in entry["nlp_data"].get("core", {}):
            for topic in entry["nlp_data"]["core"]["topics"]:
                topics.append(topic)
    
    # Calculate stats
    total_entries = len(entries)
    active_days = len(set([d.date() for d in entry_dates])) if entry_dates else 0
    streak = calculate_streak(entry_dates) if entry_dates else 0
    
    # Create date ranges for filtering
    today = datetime.now().date()
    week_ago = (datetime.now() - timedelta(days=7)).date()
    month_ago = (datetime.now() - timedelta(days=30)).date()
    
    # Filter entries by date ranges
    week_entries = [d for d in entry_dates if d.date() >= week_ago]
    month_entries = [d for d in entry_dates if d.date() >= month_ago]
    
    # Journal frequency chart
    freq_chart = None
    if entry_dates:
        # Group entries by day
        entries_by_day = {}
        for date in entry_dates:
            day = date.date()
            if day not in entries_by_day:
                entries_by_day[day] = 0
            entries_by_day[day] += 1
        
        # Convert to list for plotting
        freq_data = []
        for day, count in entries_by_day.items():
            freq_data.append({
                "date": day,
                "entries": count
            })
        
        # Sort by date
        freq_data.sort(key=lambda x: x["date"])
        
        # Create dataframe
        import pandas as pd
        freq_df = pd.DataFrame(freq_data)
        
        # Only show last 30 days
        cutoff_date = today - timedelta(days=30)
        recent_freq_df = freq_df[freq_df["date"] >= cutoff_date]
        
        if not recent_freq_df.empty:
            # Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(recent_freq_df["date"], recent_freq_df["entries"], color="#4285F4")
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Entries")
            ax.set_title("Journal Entries by Day (Last 30 Days)")
            
            # Format x-axis to show dates
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            freq_chart = generate_plot_base64(fig)
    
    # Sentiment stats
    sentiment_stats = None
    sentiment_chart = None
    if sentiments:
        positive_count = sum(1 for s in sentiments if s["label"] == "POSITIVE")
        negative_count = sum(1 for s in sentiments if s["label"] == "NEGATIVE")
        neutral_count = sum(1 for s in sentiments if s["label"] == "NEUTRAL")
        
        total_sentiments = len(sentiments)
        
        positive_percent = round((positive_count / total_sentiments) * 100 if total_sentiments > 0 else 0)
        neutral_percent = round((neutral_count / total_sentiments) * 100 if total_sentiments > 0 else 0)
        negative_percent = round((negative_count / total_sentiments) * 100 if total_sentiments > 0 else 0)
        
        sentiment_stats = {
            "positive": positive_percent,
            "neutral": neutral_percent,
            "negative": negative_percent
        }
        
        # Sentiment trend chart
        sentiment_data = []
        for s in sentiments:
            score = s["score"]
            # Normalize score: 0 = negative, 0.5 = neutral, 1 = positive
            if s["label"] == "NEGATIVE":
                normalized_score = 1 - score  # Invert negative scores
            else:
                normalized_score = score
                
            sentiment_data.append({
                "date": s["date"],
                "sentiment": normalized_score
            })
        
        # Sort by date
        sentiment_data.sort(key=lambda x: x["date"])
        
        # Create dataframe
        sentiment_df = pd.DataFrame(sentiment_data)
        
        # Only show last 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        recent_sentiment_df = sentiment_df[sentiment_df["date"] >= cutoff_date]
        
        if not recent_sentiment_df.empty:
            # Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Set colors based on sentiment
            sentiment_colors = []
            for score in recent_sentiment_df["sentiment"]:
                if score > 0.6:
                    sentiment_colors.append("#34A853")  # Green for positive
                elif score < 0.4:
                    sentiment_colors.append("#EA4335")  # Red for negative
                else:
                    sentiment_colors.append("#FBBC05")  # Yellow for neutral
            
            ax.scatter(recent_sentiment_df["date"], recent_sentiment_df["sentiment"], 
                      color=sentiment_colors, alpha=0.7)
            
            # Add trend line
            try:
                from scipy import stats
                
                # Convert dates to numbers for regression
                dates_num = mdates.date2num(recent_sentiment_df["date"])
                
                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    dates_num, recent_sentiment_df["sentiment"]
                )
                
                # Create line based on regression
                line_x = np.array([min(dates_num), max(dates_num)])
                line_y = intercept + slope * line_x
                
                # Convert back to datetime for plotting
                line_x_dates = mdates.num2date(line_x)
                
                # Plot trend line
                ax.plot(line_x_dates, line_y, 'b--', alpha=0.7)
                
                # Add annotation about trend
                if slope > 0.01:
                    trend_text = "Your mood is improving!"
                elif slope < -0.01:
                    trend_text = "Your mood is declining."
                else:
                    trend_text = "Your mood is stable."
                    
                ax.annotate(trend_text, xy=(0.05, 0.95), xycoords='axes fraction',
                           fontsize=10, ha='left', va='top')
            except:
                # Skip trend line if error occurs
                pass
            
            # Add reference lines
            ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
            ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.3)
            ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.3)
            
            ax.set_ylim(0, 1)
            ax.set_xlabel("Date")
            ax.set_ylabel("Mood Score")
            ax.set_title("Mood Trends (Last 30 Days)")
            
            # Format x-axis to show dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            sentiment_chart = generate_plot_base64(fig)
    
    # Narrative insights
    narrative_stats = None
    narrative_chart = None
    narratives = mongo_client.get_narratives(session['user_id'])
    
    if narratives:
        # Count narratives by arc type
        arc_types = {}
        for narrative in narratives:
            arc_type = narrative.get("narrative_arc", "OTHER")
            if arc_type not in arc_types:
                arc_types[arc_type] = 0
            arc_types[arc_type] += 1
        
        # Create pie chart of narrative arc types
        if arc_types:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Custom colors for narrative arcs
            color_map = {
                "BEGINNING": "#4285F4",  # Blue
                "MIDDLE": "#34A853",     # Green
                "CLIMAX": "#FBBC05",     # Yellow
                "RESOLUTION": "#EA4335"  # Red
            }
            
            colors = [color_map.get(arc, "#7F7F7F") for arc in arc_types.keys()]
            
            wedges, texts, autotexts = ax.pie(
                arc_types.values(), 
                labels=arc_types.keys(),
                autopct='%1.1f%%',
                startangle=90,
                colors=colors
            )
            
            # Style the chart
            ax.axis('equal')
            plt.setp(autotexts, size=10, weight="bold", color="white")
            plt.setp(texts, size=12)
            
            # Add title
            ax.set_title("Narratives by Arc Type", size=14)
            
            plt.tight_layout()
            narrative_chart = generate_plot_base64(fig)
        
        # Calculate narrative stats
        total_narratives = len(narratives)
        complete_narratives = sum(1 for n in narratives if n.get("narrative_arc") == "RESOLUTION")
        ongoing_narratives = total_narratives - complete_narratives
        
        narrative_stats = {
            "total": total_narratives,
            "complete": complete_narratives,
            "ongoing": ongoing_narratives,
            "completion_rate": round((complete_narratives / total_narratives) * 100 if total_narratives > 0 else 0)
        }
    
    # Topic chart
    topic_chart = None
    if topics:
        # Count topic frequency
        from collections import Counter
        topic_counter = Counter(topics)
        
        # Get top 10 topics
        top_topics = topic_counter.most_common(10)
        
        if top_topics:
            # Create dataframe
            topics_df = pd.DataFrame(top_topics, columns=["Topic", "Count"])
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(topics_df["Topic"], topics_df["Count"], color="#4285F4")
            
            # Add count labels
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 0.5
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, str(int(width)),
                       va='center', fontsize=8)
            
            ax.set_title("Most Mentioned Topics")
            ax.set_xlabel("Frequency")
            
            plt.tight_layout()
            topic_chart = generate_plot_base64(fig)
    
    # Event stats
    event_stats = None
    event_chart = None
    events = mongo_client.get_events(session['user_id'])
    
    if events:
        # Count events by type
        event_types = {}
        for event in events:
            event_type = event.get("event_type", "OTHER")
            if event_type not in event_types:
                event_types[event_type] = 0
            event_types[event_type] += 1
        
        # Create pie chart of event types
        if event_types:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Custom colors
            color_map = {
                "WORK": "#4285F4",
                "SOCIAL": "#EA4335",
                "HEALTH": "#34A853",
                "TRAVEL": "#FBBC05",
                "MILESTONE": "#9C27B0",
                "EDUCATION": "#FF6D01",
                "CREATIVE": "#00BCD4",
                "PERSONAL": "#9E9E9E",
                "OTHER": "#7F7F7F"
            }
            
            colors = [color_map.get(et, "#7F7F7F") for et in event_types.keys()]
            
            wedges, texts, autotexts = ax.pie(
                event_types.values(), 
                labels=event_types.keys(),
                autopct='%1.1f%%',
                startangle=90,
                colors=colors
            )
            
            # Style the chart
            ax.axis('equal')
            plt.setp(autotexts, size=10, weight="bold", color="white")
            plt.setp(texts, size=12)
            
            # Add title
            ax.set_title("Events by Type", size=14)
            
            plt.tight_layout()
            event_chart = generate_plot_base64(fig)
        
        # Event completion stats
        ongoing_events = sum(1 for e in events if e.get("status") == "ongoing")
        completed_events = sum(1 for e in events if e.get("status") == "completed")
        total_events = ongoing_events + completed_events
        
        if total_events > 0:
            completion_rate = round((completed_events / total_events) * 100 if total_events > 0 else 0)
            
            event_stats = {
                "total": total_events,
                "ongoing": ongoing_events,
                "completed": completed_events,
                "completion_rate": completion_rate
            }
    
    # Emotional intelligence insights
    emotion_stats = None
    emotion_chart = None
    emotional_arcs = list(mongo_client.emotional_arcs.find({"user_id": session['user_id']}))
    
    if emotional_arcs:
        # Count emotions by type
        emotion_types = {}
        for arc in emotional_arcs:
            emotion = arc.get("primary_emotion")
            if emotion and emotion != "NEUTRAL":
                if emotion not in emotion_types:
                    emotion_types[emotion] = 0
                emotion_types[emotion] += 1
        
        # Create bar chart of emotion types
        if emotion_types:
            # Sort emotions by frequency
            sorted_emotions = sorted(emotion_types.items(), key=lambda x: x[1], reverse=True)
            emotions = [e[0] for e in sorted_emotions]
            counts = [e[1] for e in sorted_emotions]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Custom colors for emotions
            color_map = {
                "JOY": "#FFC107",        # Amber
                "HAPPINESS": "#FFEB3B",   # Yellow
                "CONTENTMENT": "#CDDC39", # Lime
                "TRUST": "#8BC34A",       # Light Green
                "ANTICIPATION": "#4CAF50", # Green
                "SURPRISE": "#009688",    # Teal
                "FEAR": "#7986CB",        # Indigo
                "ANGER": "#F44336",       # Red
                "DISGUST": "#9C27B0",     # Purple
                "SADNESS": "#673AB7"      # Deep Purple
            }
            
            colors = [color_map.get(e, "#9E9E9E") for e in emotions]
            
            # Create horizontal bar chart
            ax.barh(emotions, counts, color=colors)
            
            # Add count labels
            for i, v in enumerate(counts):
                ax.text(v + 0.5, i, str(v), va='center')
            
            # Style the chart
            ax.set_title("Most Common Emotions", size=14)
            ax.set_xlabel("Frequency")
            
            plt.tight_layout()
            emotion_chart = generate_plot_base64(fig)
        
        # Calculate emotion stats
        positive_emotions = ["JOY", "HAPPINESS", "CONTENTMENT", "TRUST", "ANTICIPATION"]
        negative_emotions = ["FEAR", "ANGER", "DISGUST", "SADNESS"]
        
        positive_count = sum(emotion_types.get(e, 0) for e in positive_emotions)
        negative_count = sum(emotion_types.get(e, 0) for e in negative_emotions)
        neutral_count = emotion_types.get("NEUTRAL", 0)
        surprise_count = emotion_types.get("SURPRISE", 0)
        
        total_emotions = positive_count + negative_count + neutral_count + surprise_count
        
        if total_emotions > 0:
            emotion_stats = {
                "positive": round((positive_count / total_emotions) * 100),
                "negative": round((negative_count / total_emotions) * 100),
                "neutral": round((neutral_count / total_emotions) * 100),
                "surprise": round((surprise_count / total_emotions) * 100)
            }
    
    # Relationship insights
    relationship_stats = None
    relationship_chart = None
    relationships = mongo_client.entity_relationships.find({"user_id": session['user_id']})
    relationships = list(relationships)
    
    if relationships:
        # Count relationships by type
        relationship_types = {}
        for rel in relationships:
            rel_type = rel.get("relationship_type", "ASSOCIATED")
            if rel_type not in relationship_types:
                relationship_types[rel_type] = 0
            relationship_types[rel_type] += 1
        
        # Create pie chart of relationship types
        if relationship_types:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Custom colors
            color_map = {
                "FAMILY": "#0F9D58",       # Dark green
                "ROMANTIC": "#E91E63",     # Pink
                "FRIENDSHIP": "#F4B400",   # Yellow
                "PROFESSIONAL": "#DB4437", # Red
                "LOCATION": "#34A853",     # Green
                "ASSOCIATED": "#9E9E9E"    # Grey
            }
            
            labels = []
            sizes = []
            colors = []
            
            for rel_type, count in relationship_types.items():
                # Format label
                label = rel_type.title().replace('_', ' ')
                labels.append(label)
                sizes.append(count)
                colors.append(color_map.get(rel_type, "#7F7F7F"))
            
            wedges, texts, autotexts = ax.pie(
                sizes, 
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors
            )
            
            # Style the chart
            ax.axis('equal')
            plt.setp(autotexts, size=10, weight="bold", color="white")
            plt.setp(texts, size=12)
            
            # Add title
            ax.set_title("Relationship Types", size=14)
            
            plt.tight_layout()
            relationship_chart = generate_plot_base64(fig)
        
        # Calculate relationship stats
        total_relationships = len(relationships)
        
        # Count entities with relationships
        entity_ids = set()
        for rel in relationships:
            entity_ids.add(rel.get("source_id", ""))
            entity_ids.add(rel.get("target_id", ""))
        
        entity_count = len(entity_ids)
        
        # Calculate average relationships per entity
        avg_relationships = round(total_relationships / entity_count, 1) if entity_count > 0 else 0
        
        relationship_stats = {
            "total": total_relationships,
            "entities": entity_count,
            "avg_per_entity": avg_relationships
        }
    
    return render_template('insights.html',
                         has_data=True,
                         total_entries=total_entries,
                         active_days=active_days,
                         streak=streak,
                         week_entries=len(week_entries),
                         month_entries=len(month_entries),
                         freq_chart=freq_chart,
                         sentiment_stats=sentiment_stats,
                         sentiment_chart=sentiment_chart,
                         topic_chart=topic_chart,
                         event_stats=event_stats,
                         event_chart=event_chart,
                         narrative_stats=narrative_stats,
                         narrative_chart=narrative_chart,
                         emotion_stats=emotion_stats,
                         emotion_chart=emotion_chart,
                         relationship_stats=relationship_stats,
                         relationship_chart=relationship_chart,
                         insights_data=insights_data)

# =============================================================================
# Main Application
# =============================================================================

if __name__ == '__main__':
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # Print startup message
    print("=" * 80)
    print("Enhanced Reflectify Flask Application with Advanced Narrative Understanding")
    print("=" * 80)
    print("Creating templates and setting up the application...")
    
    # Start the Flask server
    print("Starting server on http://127.0.0.1:5000")
    print("You can access the application by opening this URL in your browser.")
    print("=" * 80)
    
    # Run the application
    app.run(debug=False, use_reloader=False)
