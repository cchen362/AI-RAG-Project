"""
Agent Memory Management - Conversation State and Context

This module manages conversation history, context preservation, and knowledge
accumulation across multi-turn interactions in the agentic RAG system.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib

@dataclass
class ConversationTurn:
    """Single conversation turn with user query and agent response"""
    turn_id: str
    user_query: str
    agent_response: str
    reasoning_chain: List[Dict]
    sources_used: List[str]
    confidence_score: float
    timestamp: float
    execution_time: float

@dataclass
class KnowledgeFragment:
    """Reusable knowledge fragment from previous interactions"""
    fragment_id: str
    content: str
    source: str
    relevance_score: float
    last_used: float
    usage_count: int
    related_queries: List[str]

class AgentMemory:
    """
    Manages conversation state, context preservation, and knowledge accumulation
    for the agentic RAG system.
    """
    
    def __init__(self, max_conversation_length: int = 10, 
                 memory_persistence_file: Optional[str] = None):
        """
        Initialize agent memory system.
        
        Args:
            max_conversation_length: Maximum turns to keep in active memory
            memory_persistence_file: Optional file to persist memory across sessions
        """
        self.max_conversation_length = max_conversation_length
        self.memory_file = memory_persistence_file
        
        # Active conversation state
        self.conversation_history: List[ConversationTurn] = []
        self.current_context: Dict[str, Any] = {}
        
        # Knowledge management
        self.knowledge_fragments: Dict[str, KnowledgeFragment] = {}
        self.query_patterns: Dict[str, List[str]] = defaultdict(list)
        
        # Session metadata
        self.session_start = time.time()
        self.total_interactions = 0
        
        # Load persistent memory if available
        if self.memory_file:
            self._load_memory()
    
    def add_conversation_turn(self, user_query: str, agent_response: str,
                            reasoning_chain: List[Dict], sources_used: List[str],
                            confidence_score: float, execution_time: float) -> str:
        """
        Add a new conversation turn to memory.
        
        Returns:
            turn_id: Unique identifier for this turn
        """
        turn_id = self._generate_turn_id(user_query)
        
        turn = ConversationTurn(
            turn_id=turn_id,
            user_query=user_query,
            agent_response=agent_response,
            reasoning_chain=reasoning_chain,
            sources_used=sources_used,
            confidence_score=confidence_score,
            timestamp=time.time(),
            execution_time=execution_time
        )
        
        self.conversation_history.append(turn)
        
        # Maintain conversation length limit
        if len(self.conversation_history) > self.max_conversation_length:
            old_turn = self.conversation_history.pop(0)
            self._archive_turn_knowledge(old_turn)
        
        # Update context and patterns
        self._update_context(turn)
        self._update_query_patterns(user_query, sources_used)
        
        self.total_interactions += 1
        
        # Persist if configured
        if self.memory_file:
            self._save_memory()
            
        return turn_id
    
    def get_relevant_context(self, current_query: str, max_turns: int = 3) -> Dict[str, Any]:
        """
        Get relevant context from conversation history for current query.
        
        Args:
            current_query: Current user query
            max_turns: Maximum previous turns to consider
            
        Returns:
            Context dictionary with relevant information
        """
        context = {
            "previous_queries": [],
            "related_knowledge": [],
            "conversation_patterns": {},
            "suggested_sources": []
        }
        
        # Get recent conversation history
        recent_turns = self.conversation_history[-max_turns:] if self.conversation_history else []
        
        for turn in recent_turns:
            context["previous_queries"].append({
                "query": turn.user_query,
                "confidence": turn.confidence_score,
                "sources": turn.sources_used
            })
        
        # Find related knowledge fragments
        related_fragments = self._find_related_knowledge(current_query)
        context["related_knowledge"] = [
            {
                "content": fragment.content,
                "source": fragment.source,
                "relevance": fragment.relevance_score
            }
            for fragment in related_fragments[:3]  # Top 3 most relevant
        ]
        
        # Suggest optimal sources based on query patterns
        context["suggested_sources"] = self._suggest_sources(current_query)
        
        # Add conversation patterns
        context["conversation_patterns"] = self._analyze_conversation_patterns()
        
        return context
    
    def update_knowledge_fragments(self, source: str, content: str, 
                                 query_context: str, relevance_score: float = 0.7):
        """
        Add or update knowledge fragments from successful retrievals.
        
        Args:
            source: Knowledge source (text_rag, colpali_visual, salesforce)
            content: Content to store
            query_context: Query that led to this knowledge
            relevance_score: Relevance score for this fragment
        """
        fragment_id = self._generate_fragment_id(content, source)
        
        if fragment_id in self.knowledge_fragments:
            # Update existing fragment
            fragment = self.knowledge_fragments[fragment_id]
            fragment.last_used = time.time()
            fragment.usage_count += 1
            if query_context not in fragment.related_queries:
                fragment.related_queries.append(query_context)
        else:
            # Create new fragment
            fragment = KnowledgeFragment(
                fragment_id=fragment_id,
                content=content,
                source=source,
                relevance_score=relevance_score,
                last_used=time.time(),
                usage_count=1,
                related_queries=[query_context]
            )
            self.knowledge_fragments[fragment_id] = fragment
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation session."""
        if not self.conversation_history:
            return {"status": "No conversation history"}
        
        # Calculate statistics
        avg_confidence = sum(turn.confidence_score for turn in self.conversation_history) / len(self.conversation_history)
        avg_execution_time = sum(turn.execution_time for turn in self.conversation_history) / len(self.conversation_history)
        
        # Count source usage
        source_usage = defaultdict(int)
        for turn in self.conversation_history:
            for source in turn.sources_used:
                source_usage[source] += 1
        
        # Recent query topics
        recent_topics = [turn.user_query for turn in self.conversation_history[-3:]]
        
        return {
            "session_duration": time.time() - self.session_start,
            "total_turns": len(self.conversation_history),
            "avg_confidence": avg_confidence,
            "avg_execution_time": avg_execution_time,
            "source_usage": dict(source_usage),
            "recent_topics": recent_topics,
            "knowledge_fragments": len(self.knowledge_fragments)
        }
    
    def clear_session(self, preserve_knowledge: bool = True):
        """
        Clear current session while optionally preserving learned knowledge.
        
        Args:
            preserve_knowledge: Whether to keep knowledge fragments
        """
        # Archive current conversation if valuable
        for turn in self.conversation_history:
            if turn.confidence_score > 0.7:  # Only archive high-confidence turns
                self._archive_turn_knowledge(turn)
        
        # Clear conversation state
        self.conversation_history.clear()
        self.current_context.clear()
        
        # Optionally clear knowledge
        if not preserve_knowledge:
            self.knowledge_fragments.clear()
            self.query_patterns.clear()
        
        # Reset session metadata
        self.session_start = time.time()
        self.total_interactions = 0
    
    def _generate_turn_id(self, query: str) -> str:
        """Generate unique ID for conversation turn."""
        timestamp = str(int(time.time() * 1000))
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        return f"turn_{timestamp}_{query_hash}"
    
    def _generate_fragment_id(self, content: str, source: str) -> str:
        """Generate unique ID for knowledge fragment."""
        content_hash = hashlib.md5(f"{source}_{content}".encode()).hexdigest()[:12]
        return f"frag_{content_hash}"
    
    def _update_context(self, turn: ConversationTurn):
        """Update current context based on new conversation turn."""
        self.current_context.update({
            "last_query": turn.user_query,
            "last_sources": turn.sources_used,
            "last_confidence": turn.confidence_score,
            "conversation_length": len(self.conversation_history)
        })
    
    def _update_query_patterns(self, query: str, sources_used: List[str]):
        """Update query patterns for source recommendation."""
        query_type = self._classify_query_type(query)
        self.query_patterns[query_type].extend(sources_used)
    
    def _classify_query_type(self, query: str) -> str:
        """Simple query classification for pattern tracking."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['chart', 'graph', 'figure', 'visual']):
            return "visual"
        elif any(word in query_lower for word in ['business', 'project', 'salesforce']):
            return "business"
        elif any(word in query_lower for word in ['technical', 'model', 'algorithm']):
            return "technical"
        else:
            return "general"
    
    def _find_related_knowledge(self, query: str) -> List[KnowledgeFragment]:
        """Find knowledge fragments related to current query."""
        # Simple keyword-based matching (can be enhanced with embeddings)
        query_words = set(query.lower().split())
        
        scored_fragments = []
        for fragment in self.knowledge_fragments.values():
            content_words = set(fragment.content.lower().split())
            
            # Calculate similarity based on word overlap
            overlap = len(query_words.intersection(content_words))
            if overlap > 0:
                similarity = overlap / len(query_words.union(content_words))
                fragment.relevance_score = similarity
                scored_fragments.append(fragment)
        
        # Sort by relevance and recency
        scored_fragments.sort(key=lambda x: (x.relevance_score, x.last_used), reverse=True)
        return scored_fragments
    
    def _suggest_sources(self, query: str) -> List[str]:
        """Suggest optimal sources based on query patterns."""
        query_type = self._classify_query_type(query)
        
        if query_type in self.query_patterns:
            # Count source usage for this query type
            source_counts = defaultdict(int)
            for source in self.query_patterns[query_type]:
                source_counts[source] += 1
            
            # Return sources sorted by usage frequency
            return sorted(source_counts.keys(), key=lambda x: source_counts[x], reverse=True)
        
        # Default suggestions based on query type
        return self._get_default_sources(query_type)
    
    def _get_default_sources(self, query_type: str) -> List[str]:
        """Get default source suggestions for query type."""
        defaults = {
            "visual": ["colpali_visual", "text_rag"],
            "business": ["salesforce", "text_rag"],
            "technical": ["text_rag", "colpali_visual"],
            "general": ["text_rag", "salesforce", "colpali_visual"]
        }
        return defaults.get(query_type, ["text_rag"])
    
    def _analyze_conversation_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in conversation flow."""
        if len(self.conversation_history) < 2:
            return {}
        
        patterns = {
            "avg_confidence_trend": [],
            "source_switching_patterns": [],
            "query_complexity_trend": []
        }
        
        # Confidence trend
        confidences = [turn.confidence_score for turn in self.conversation_history]
        patterns["avg_confidence_trend"] = confidences[-3:] if len(confidences) >= 3 else confidences
        
        # Source switching patterns
        if len(self.conversation_history) >= 2:
            for i in range(1, len(self.conversation_history)):
                prev_sources = set(self.conversation_history[i-1].sources_used)
                curr_sources = set(self.conversation_history[i].sources_used)
                if prev_sources != curr_sources:
                    patterns["source_switching_patterns"].append({
                        "from": list(prev_sources),
                        "to": list(curr_sources)
                    })
        
        return patterns
    
    def _archive_turn_knowledge(self, turn: ConversationTurn):
        """Archive knowledge from conversation turn before removing from active memory."""
        # Extract valuable knowledge from high-confidence turns
        if turn.confidence_score > 0.7:
            for source in turn.sources_used:
                self.update_knowledge_fragments(
                    source=source,
                    content=turn.agent_response[:500],  # First 500 chars
                    query_context=turn.user_query,
                    relevance_score=turn.confidence_score
                )
    
    def _save_memory(self):
        """Save memory state to file."""
        if not self.memory_file:
            return
        
        memory_data = {
            "conversation_history": [asdict(turn) for turn in self.conversation_history],
            "knowledge_fragments": {k: asdict(v) for k, v in self.knowledge_fragments.items()},
            "query_patterns": dict(self.query_patterns),
            "session_metadata": {
                "session_start": self.session_start,
                "total_interactions": self.total_interactions
            }
        }
        
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save memory to {self.memory_file}: {e}")
    
    def _load_memory(self):
        """Load memory state from file."""
        if not self.memory_file:
            return
        
        try:
            with open(self.memory_file, 'r') as f:
                memory_data = json.load(f)
            
            # Restore conversation history
            self.conversation_history = [
                ConversationTurn(**turn_data) 
                for turn_data in memory_data.get("conversation_history", [])
            ]
            
            # Restore knowledge fragments
            fragment_data = memory_data.get("knowledge_fragments", {})
            self.knowledge_fragments = {
                k: KnowledgeFragment(**v) for k, v in fragment_data.items()
            }
            
            # Restore query patterns
            self.query_patterns = defaultdict(list, memory_data.get("query_patterns", {}))
            
            # Restore session metadata
            metadata = memory_data.get("session_metadata", {})
            self.session_start = metadata.get("session_start", time.time())
            self.total_interactions = metadata.get("total_interactions", 0)
            
        except FileNotFoundError:
            # File doesn't exist yet, start fresh
            pass
        except Exception as e:
            print(f"Warning: Could not load memory from {self.memory_file}: {e}")
            # Continue with fresh memory