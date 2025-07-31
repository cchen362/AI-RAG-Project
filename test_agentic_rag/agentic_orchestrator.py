"""
Agentic RAG Orchestrator - Graph-R1 Inspired Multi-Turn Reasoning

This module implements a lightweight agentic approach to RAG that follows the 
Graph-R1 "think-retrieve-rethink-generate" paradigm, building on the existing
production components from the main RAG system.
"""

import sys
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import time

# Add parent directory to path to import existing components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_system import RAGSystem
from src.colpali_retriever import ColPaliRetriever  
from src.salesforce_connector import SalesforceConnector
from src.cross_encoder_reranker import CrossEncoderReRanker

class AgentStep(Enum):
    """Agent reasoning steps following Graph-R1 paradigm"""
    THINK = "think"
    RETRIEVE = "retrieve" 
    RETHINK = "rethink"
    GENERATE = "generate"

class SourceType(Enum):
    """Available knowledge sources"""
    TEXT_RAG = "text_rag"
    COLPALI_VISUAL = "colpali_visual"
    SALESFORCE = "salesforce"

@dataclass
class AgentAction:
    """Single agent action in the reasoning chain"""
    step: AgentStep
    source: Optional[SourceType] = None
    query: str = ""
    reasoning: str = ""
    result: Any = None
    confidence: float = 0.0
    timestamp: float = 0.0

@dataclass 
class AgentResponse:
    """Complete agent response with reasoning chain"""
    final_answer: str
    reasoning_chain: List[AgentAction]
    total_steps: int
    execution_time: float
    sources_used: List[SourceType]
    confidence_score: float

class AgenticOrchestrator:
    """
    Graph-R1 inspired agentic orchestrator that performs multi-turn reasoning
    over multiple knowledge sources using existing production components.
    """
    
    def __init__(self, 
                 rag_system: Optional[RAGSystem] = None,
                 colpali_retriever: Optional[ColPaliRetriever] = None,
                 salesforce_connector: Optional[SalesforceConnector] = None,
                 reranker: Optional[CrossEncoderReRanker] = None,
                 max_steps: int = 5,
                 confidence_threshold: float = 0.8,
                 enable_logging: bool = True):
        """
        Initialize the agentic orchestrator with existing components.
        
        Args:
            rag_system: Existing RAG system for text processing
            colpali_retriever: Visual document retriever
            salesforce_connector: Business data connector
            reranker: Cross-encoder for result ranking
            max_steps: Maximum reasoning steps per query
            confidence_threshold: Minimum confidence to stop reasoning
            enable_logging: Enable detailed logging
        """
        self.rag_system = rag_system
        self.colpali_retriever = colpali_retriever
        self.salesforce_connector = salesforce_connector
        self.reranker = reranker
        
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold
        
        # Initialize logging
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.WARNING)
            
        # Track conversation state
        self.conversation_history = []
        self.agent_memory = {}
        
    def query(self, user_query: str, context: Optional[Dict] = None) -> AgentResponse:
        """
        Main entry point for agentic multi-turn reasoning.
        
        Args:
            user_query: User's question/request
            context: Optional context from previous interactions
            
        Returns:
            AgentResponse with complete reasoning chain and final answer
        """
        start_time = time.time()
        reasoning_chain = []
        
        self.logger.info(f"Starting agentic reasoning for query: {user_query}")
        
        # Step 1: Initial THINK - analyze query and plan approach
        think_action = self._think_step(user_query, context)
        reasoning_chain.append(think_action)
        
        current_knowledge = {}
        step_count = 1
        
        while step_count < self.max_steps:
            # Step 2: RETRIEVE - get information from selected sources
            retrieve_action = self._retrieve_step(
                user_query, think_action.reasoning, current_knowledge
            )
            reasoning_chain.append(retrieve_action)
            
            if retrieve_action.result:
                current_knowledge[retrieve_action.source.value] = retrieve_action.result
            
            # Step 3: RETHINK - analyze results and decide next action
            rethink_action = self._rethink_step(
                user_query, current_knowledge, reasoning_chain
            )
            reasoning_chain.append(rethink_action)
            
            # Check if we have sufficient information
            if rethink_action.confidence >= self.confidence_threshold:
                self.logger.info(f"Confidence threshold reached: {rethink_action.confidence}")
                break
                
            # Check if we need more information
            if "CONTINUE" not in rethink_action.reasoning.upper():
                break
                
            step_count += 1
        
        # Step 4: GENERATE - synthesize final answer
        generate_action = self._generate_step(user_query, current_knowledge, reasoning_chain)
        reasoning_chain.append(generate_action)
        
        execution_time = time.time() - start_time
        sources_used = [action.source for action in reasoning_chain 
                       if action.source and action.result]
        
        response = AgentResponse(
            final_answer=generate_action.result,
            reasoning_chain=reasoning_chain,
            total_steps=len(reasoning_chain),
            execution_time=execution_time,
            sources_used=sources_used,
            confidence_score=generate_action.confidence
        )
        
        self.logger.info(f"Agentic reasoning completed in {execution_time:.2f}s with {len(reasoning_chain)} steps")
        return response
    
    def _think_step(self, query: str, context: Optional[Dict] = None) -> AgentAction:
        """
        Initial thinking step - analyze query and plan retrieval strategy.
        This implements the "think" phase of Graph-R1.
        """
        self.logger.info("THINK: Analyzing query and planning approach")
        
        # Simple heuristic-based query analysis (can be enhanced with LLM)
        reasoning = self._analyze_query_intent(query)
        
        return AgentAction(
            step=AgentStep.THINK,
            query=query,
            reasoning=reasoning,
            confidence=0.7,  # Moderate confidence in initial analysis
            timestamp=time.time()
        )
    
    def _retrieve_step(self, query: str, plan: str, current_knowledge: Dict) -> AgentAction:
        """
        Retrieval step - get information from the most appropriate source.
        This implements the "retrieve" phase of Graph-R1.
        """
        # Determine best source based on plan and current knowledge
        source = self._select_source(plan, current_knowledge)
        
        self.logger.info(f"RETRIEVE: Querying {source.value}")
        
        result = None
        confidence = 0.0
        
        try:
            if source == SourceType.TEXT_RAG and self.rag_system:
                result = self._query_text_rag(query)
                confidence = 0.8
                
            elif source == SourceType.COLPALI_VISUAL and self.colpali_retriever:
                result = self._query_colpali(query)
                confidence = 0.75
                
            elif source == SourceType.SALESFORCE and self.salesforce_connector:
                result = self._query_salesforce(query)
                confidence = 0.7
                
        except Exception as e:
            self.logger.error(f"Error querying {source.value}: {str(e)}")
            result = None
            confidence = 0.0
        
        return AgentAction(
            step=AgentStep.RETRIEVE,
            source=source,
            query=query,
            reasoning=f"Retrieved information from {source.value}",
            result=result,
            confidence=confidence,
            timestamp=time.time()
        )
    
    def _rethink_step(self, query: str, knowledge: Dict, chain: List[AgentAction]) -> AgentAction:
        """
        Rethinking step - analyze retrieved information and decide next action.
        This implements the "rethink" phase of Graph-R1.
        """
        self.logger.info("RETHINK: Analyzing retrieved information")
        
        # Analyze completeness of current knowledge
        reasoning, confidence = self._assess_knowledge_completeness(query, knowledge, chain)
        
        return AgentAction(
            step=AgentStep.RETHINK,
            query=query,
            reasoning=reasoning,
            confidence=confidence,
            timestamp=time.time()
        )
    
    def _generate_step(self, query: str, knowledge: Dict, chain: List[AgentAction]) -> AgentAction:
        """
        Generation step - synthesize final answer from all retrieved information.
        This implements the "generate" phase of Graph-R1.
        """
        self.logger.info("GENERATE: Synthesizing final answer")
        
        # Combine all knowledge sources into coherent answer
        final_answer = self._synthesize_answer(query, knowledge)
        
        # Calculate confidence based on available information
        confidence = self._calculate_final_confidence(knowledge, chain)
        
        return AgentAction(
            step=AgentStep.GENERATE,
            query=query,
            reasoning="Synthesized final answer from available knowledge",
            result=final_answer,
            confidence=confidence,
            timestamp=time.time()
        )
    
    def _analyze_query_intent(self, query: str) -> str:
        """
        Analyze query to determine optimal retrieval strategy.
        Simple heuristic-based approach (can be enhanced with LLM).
        """
        query_lower = query.lower()
        
        # Visual/chart-related queries
        if any(word in query_lower for word in ['chart', 'graph', 'figure', 'image', 'visual', 'diagram']):
            return "VISUAL_QUERY: Start with ColPali for visual content, then supplement with text if needed"
        
        # Business/organizational queries  
        if any(word in query_lower for word in ['business', 'project', 'team', 'organization', 'salesforce']):
            return "BUSINESS_QUERY: Check Salesforce first, then technical sources if needed"
        
        # Technical queries
        if any(word in query_lower for word in ['model', 'algorithm', 'technical', 'implementation', 'code']):
            return "TECHNICAL_QUERY: Start with Text RAG, use ColPali for technical diagrams if needed"
        
        # Complex multi-part queries
        if '?' in query and len(query.split()) > 15:
            return "COMPLEX_QUERY: Multi-step approach required, start with Text RAG for overview"
            
        return "GENERAL_QUERY: Use Text RAG as primary source, adapt based on results"
    
    def _select_source(self, plan: str, current_knowledge: Dict) -> SourceType:
        """
        Select the next best source to query based on plan and current knowledge.
        Only selects available sources.
        """
        plan_upper = plan.upper()
        
        # If we haven't queried a recommended source yet (and it's available)
        if "VISUAL_QUERY" in plan_upper and SourceType.COLPALI_VISUAL.value not in current_knowledge:
            if self.colpali_retriever:
                return SourceType.COLPALI_VISUAL
        
        if "BUSINESS_QUERY" in plan_upper and SourceType.SALESFORCE.value not in current_knowledge:
            if self.salesforce_connector:
                return SourceType.SALESFORCE
            
        if "TECHNICAL_QUERY" in plan_upper and SourceType.TEXT_RAG.value not in current_knowledge:
            if self.rag_system:
                return SourceType.TEXT_RAG
        
        # Default fallback logic - only try available sources
        if SourceType.TEXT_RAG.value not in current_knowledge and self.rag_system:
            return SourceType.TEXT_RAG
        elif SourceType.COLPALI_VISUAL.value not in current_knowledge and self.colpali_retriever:
            return SourceType.COLPALI_VISUAL
        elif SourceType.SALESFORCE.value not in current_knowledge and self.salesforce_connector:
            return SourceType.SALESFORCE
        else:
            # If all available sources have been tried, default to best available
            if self.rag_system:
                return SourceType.TEXT_RAG
            elif self.salesforce_connector:
                return SourceType.SALESFORCE
            elif self.colpali_retriever:
                return SourceType.COLPALI_VISUAL
    
    def _query_text_rag(self, query: str) -> Dict:
        """Query the text RAG system and return structured result."""
        if not self.rag_system:
            return {"error": "Text RAG system not available"}
            
        try:
            result = self.rag_system.query(query)
            return {
                "source": "text_rag",
                "answer": result.get("answer", ""),
                "relevant_chunks": result.get("relevant_chunks", []),
                "confidence": result.get("confidence", 0.0)
            }
        except Exception as e:
            return {"error": f"Text RAG error: {str(e)}"}
    
    def _query_colpali(self, query: str) -> Dict:
        """Query ColPali visual retriever and return structured result."""
        if not self.colpali_retriever:
            return {"error": "ColPali system not available"}
            
        try:
            # Use the correct ColPali method name
            results, metrics = self.colpali_retriever.retrieve(query, top_k=5)
            return {
                "source": "colpali_visual",
                "visual_matches": [{
                    "document_name": r.document_name,
                    "page_number": r.page_number,
                    "score": r.score,
                    "content": getattr(r, 'content', 'Visual content')
                } for r in results],
                "confidence": min(results[0].score, 1.0) if results else 0.0,
                "metrics": {
                    "total_results": metrics.total_results,
                    "search_time": metrics.search_time
                }
            }
        except Exception as e:
            return {"error": f"ColPali error: {str(e)}"}
    
    def _query_salesforce(self, query: str) -> Dict:
        """Query Salesforce connector and return structured result."""
        if not self.salesforce_connector:
            return {"error": "Salesforce system not available"}
            
        try:
            # Use the correct Salesforce method name
            records = self.salesforce_connector.search_knowledge_with_intent(query, limit=5)
            return {
                "source": "salesforce",
                "records": records,
                "summary": f"Found {len(records)} relevant business records",
                "confidence": 0.8 if records else 0.2
            }
        except Exception as e:
            return {"error": f"Salesforce error: {str(e)}"}
    
    def _assess_knowledge_completeness(self, query: str, knowledge: Dict, chain: List[AgentAction]) -> Tuple[str, float]:
        """
        Assess whether we have sufficient information to answer the query.
        Returns reasoning and confidence score.
        """
        sources_queried = len([k for k in knowledge.keys() if not knowledge[k].get("error")])
        successful_results = len([k for k in knowledge.keys() if knowledge[k].get("answer") or knowledge[k].get("records")])
        
        # Count available sources
        available_sources = 0
        if self.rag_system:
            available_sources += 1
        if self.colpali_retriever:
            available_sources += 1
        if self.salesforce_connector:
            available_sources += 1
        
        # More intelligent stopping criteria
        if successful_results >= 1 and sources_queried >= available_sources:
            return "SUFFICIENT: Tried all available sources", 0.85
        elif successful_results >= 2:
            return "SUFFICIENT: Multiple sources provide good coverage", 0.85
        elif successful_results == 1 and sources_queried >= 2:
            return "ADEQUATE: One good source, others failed", 0.75
        elif successful_results == 1 and sources_queried == 1:
            # Check if we have more sources to try
            if sources_queried < available_sources:
                return "CONTINUE: Try additional available sources", 0.5
            else:
                return "SUFFICIENT: Only one source available", 0.8
        else:
            return "INSUFFICIENT: No successful retrievals", 0.2
    
    def _synthesize_answer(self, query: str, knowledge: Dict) -> str:
        """
        Synthesize final answer from all available knowledge sources.
        """
        answer_parts = []
        
        # Collect information from each source
        for source, data in knowledge.items():
            if data.get("error"):
                continue
                
            if source == "text_rag" and data.get("answer"):
                answer_parts.append(f"**Technical Information**: {data['answer']}")
                
            elif source == "colpali_visual" and data.get("visual_matches"):
                visual_info = f"Found {len(data['visual_matches'])} relevant visual elements"
                answer_parts.append(f"**Visual Analysis**: {visual_info}")
                
            elif source == "salesforce" and data.get("summary"):
                answer_parts.append(f"**Business Context**: {data['summary']}")
        
        if not answer_parts:
            return "I apologize, but I couldn't retrieve sufficient information to answer your query. Please try rephrasing or check if the relevant documents are available."
        
        # Combine all parts into coherent answer
        final_answer = f"Based on my multi-source analysis:\n\n" + "\n\n".join(answer_parts)
        
        return final_answer
    
    def _calculate_final_confidence(self, knowledge: Dict, chain: List[AgentAction]) -> float:
        """Calculate overall confidence in the final answer."""
        if not knowledge:
            return 0.0
        
        # Average confidence from successful retrievals
        confidences = []
        for data in knowledge.values():
            if not data.get("error") and data.get("confidence"):
                confidences.append(data["confidence"])
        
        if not confidences:
            return 0.3  # Low confidence if no successful retrievals
        
        # Boost confidence for multiple sources
        base_confidence = sum(confidences) / len(confidences)
        source_bonus = min(0.2, len(confidences) * 0.05)  # Up to 20% bonus for multiple sources
        
        return min(1.0, base_confidence + source_bonus)
    
    def get_reasoning_summary(self, response: AgentResponse) -> str:
        """Get a human-readable summary of the agent's reasoning process."""
        summary_parts = [
            f"**Agent Reasoning Summary** ({response.total_steps} steps, {response.execution_time:.2f}s)",
            f"**Sources Used**: {', '.join([s.value for s in response.sources_used])}",
            f"**Final Confidence**: {response.confidence_score:.2f}",
            "",
            "**Reasoning Chain**:"
        ]
        
        for i, action in enumerate(response.reasoning_chain, 1):
            step_desc = f"{i}. **{action.step.value.upper()}**"
            if action.source:
                step_desc += f" ({action.source.value})"
            if action.reasoning:
                step_desc += f": {action.reasoning}"
            summary_parts.append(step_desc)
        
        return "\n".join(summary_parts)