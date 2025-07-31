"""
True LLM Reasoning Agent - Real Agentic Intelligence

This module implements genuine agentic reasoning using GPT-4o Mini as the decision-making
engine, following the ReAct (Reasoning + Acting) framework. Replaces the pseudo-agentic
fixed pipeline with true LLM-driven dynamic reasoning.

Key Features:
- LLM-powered decision making at each reasoning step
- Dynamic source selection based on query analysis
- Multi-source intelligence with context-aware orchestration
- Cost-efficient selective retrieval
- Reasoning transparency and explainability
"""

import sys
import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import openai

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing components
from src.rag_system import RAGSystem
from src.colpali_retriever import ColPaliRetriever
from src.salesforce_connector import SalesforceConnector
from src.cross_encoder_reranker import CrossEncoderReRanker

class ReasoningStep(Enum):
    """ReAct framework reasoning steps"""
    THOUGHT = "thought"      # LLM analyzes situation and plans
    ACTION = "action"        # LLM decides which source to query
    OBSERVATION = "observation"  # System provides retrieval results
    REFLECTION = "reflection"    # LLM evaluates completeness

class SourceType(Enum):
    """Available knowledge sources with capabilities"""
    TEXT_RAG = "text_rag"
    COLPALI_VISUAL = "colpali_visual" 
    SALESFORCE = "salesforce"

@dataclass
class SourceCapability:
    """Source capability description for LLM understanding"""
    name: str
    description: str
    best_for: List[str]
    typical_response_time: str
    cost_level: str  # low/medium/high
    availability_status: str  # available/unavailable/limited

@dataclass 
class ReasoningAction:
    """Single reasoning action in the ReAct framework"""
    step: ReasoningStep
    content: str
    source: Optional[SourceType] = None
    confidence: float = 0.0
    reasoning: str = ""
    result: Any = None
    tokens_used: int = 0
    timestamp: float = 0.0

@dataclass
class AgenticResponse:
    """Complete agentic response with reasoning transparency"""
    final_answer: str
    reasoning_chain: List[ReasoningAction]
    total_reasoning_steps: int
    sources_queried: List[SourceType]
    total_execution_time: float
    total_llm_tokens: int
    confidence_score: float
    reasoning_quality: str
    cost_breakdown: Dict[str, float]

class LLMReasoningAgent:
    """
    True agentic reasoning engine using GPT-4o Mini for dynamic decision making.
    Implements ReAct framework for genuine intelligence-driven source orchestration.
    """
    
    def __init__(self, 
                 rag_system: Optional[RAGSystem] = None,
                 colpali_retriever: Optional[ColPaliRetriever] = None,
                 salesforce_connector: Optional[SalesforceConnector] = None,
                 reranker: Optional[CrossEncoderReRanker] = None,
                 model_name: str = "gpt-4o-mini",
                 max_reasoning_steps: int = 10,
                 confidence_threshold: float = 0.75,
                 cost_threshold: float = 0.10,  # Maximum cost per query in USD
                 enable_logging: bool = True):
        """
        Initialize the true LLM reasoning agent.
        
        Args:
            rag_system: Text RAG system for document retrieval
            colpali_retriever: Visual document processing system
            salesforce_connector: Business knowledge base connector
            reranker: Cross-encoder for result ranking
            model_name: LLM model for reasoning (GPT-4o Mini recommended)
            max_reasoning_steps: Maximum reasoning iterations per query
            confidence_threshold: Minimum confidence to stop reasoning
            cost_threshold: Maximum cost per query (safety limit)
            enable_logging: Enable detailed reasoning logging
        """
        # Store source systems
        self.rag_system = rag_system
        self.colpali_retriever = colpali_retriever  
        self.salesforce_connector = salesforce_connector
        self.reranker = reranker
        
        # LLM configuration
        self.model_name = model_name
        self.max_reasoning_steps = max_reasoning_steps
        self.confidence_threshold = confidence_threshold
        self.cost_threshold = cost_threshold
        
        # Initialize OpenAI client (uses existing environment variables)
        self.openai_client = openai.OpenAI()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if enable_logging:
            self.logger.setLevel(logging.INFO)
        
        # Source capability mapping for LLM understanding
        self.source_capabilities = self._initialize_source_capabilities()
        
        # Cost tracking
        self.total_cost = 0.0
        self.total_tokens = 0
        
        self.logger.info(f"True LLM Reasoning Agent initialized with {model_name}")
        self.logger.info(f"Available sources: {list(self.get_available_sources().keys())}")
    
    def _initialize_source_capabilities(self) -> Dict[SourceType, SourceCapability]:
        """Initialize source capability descriptions for LLM reasoning."""
        return {
            SourceType.TEXT_RAG: SourceCapability(
                name="Text RAG",
                description="Document chunks with semantic search, technical explanations, definitions",
                best_for=["technical queries", "definitions", "explanations", "how-to guides", "academic content"],
                typical_response_time="1-2 seconds",
                cost_level="low",
                availability_status="available" if self.rag_system else "unavailable"
            ),
            SourceType.COLPALI_VISUAL: SourceCapability(
                name="ColPali Visual",
                description="Visual document analysis, diagrams, charts, images, PDF-as-image processing",
                best_for=["visual content", "diagrams", "charts", "image analysis", "document layouts"],
                typical_response_time="3-5 seconds",
                cost_level="medium",
                availability_status="available" if self.colpali_retriever else "unavailable"
            ),
            SourceType.SALESFORCE: SourceCapability(
                name="Salesforce Knowledge",
                description="Business knowledge base, CRM data, enterprise trends, industry insights",
                best_for=["business queries", "industry trends", "enterprise applications", "CRM insights"],
                typical_response_time="2-3 seconds", 
                cost_level="low",
                availability_status="available" if self.salesforce_connector else "unavailable"
            )
        }
    
    def get_available_sources(self) -> Dict[SourceType, SourceCapability]:
        """Get currently available sources with their capabilities."""
        return {
            source_type: capability 
            for source_type, capability in self.source_capabilities.items()
            if capability.availability_status == "available"
        }
    
    def query(self, user_query: str, context: Optional[Dict] = None) -> AgenticResponse:
        """
        Main entry point for true agentic reasoning using ReAct framework.
        
        Args:
            user_query: User's question or request
            context: Optional context from previous interactions
            
        Returns:
            AgenticResponse with complete reasoning chain and final answer
        """
        start_time = time.time()
        reasoning_chain = []
        total_llm_tokens = 0
        
        self.logger.info(f"\n🚀 STARTING TRUE AGENTIC REASONING")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Query: {user_query}")
        self.logger.info(f"Available sources: {list(self.get_available_sources().keys())}")
        self.logger.info(f"Max steps: {self.max_reasoning_steps}, Confidence threshold: {self.confidence_threshold}")
        
        # Initialize reasoning context
        reasoning_context = {
            "query": user_query,
            "available_sources": self.get_available_sources(),
            "results_gathered": {},
            "reasoning_history": [],
            "cost_so_far": 0.0
        }
        
        # ReAct Framework Implementation
        step_count = 0
        task_complete = False
        
        while step_count < self.max_reasoning_steps and not task_complete:
            step_count += 1
            
            # THOUGHT: LLM analyzes current state and plans next action
            thought_action = self._llm_thought_step(reasoning_context, step_count)
            reasoning_chain.append(thought_action)
            total_llm_tokens += thought_action.tokens_used
            
            # Check if LLM decided task is complete
            if "COMPLETE" in thought_action.content.upper():
                self.logger.info("LLM determined task complete, generating final answer")
                break
            
            # ACTION: LLM decides which source to query
            action_decision = self._llm_action_step(reasoning_context, thought_action)
            reasoning_chain.append(action_decision)
            total_llm_tokens += action_decision.tokens_used
            
            # OBSERVATION: Query the selected source
            if action_decision.source:
                observation = self._execute_source_query(action_decision.source, user_query, reasoning_context)
                reasoning_chain.append(observation)
                
                # Update reasoning context with results
                reasoning_context["results_gathered"][action_decision.source.value] = observation.result
            
            # REFLECTION: LLM evaluates completeness and next steps
            reflection = self._llm_reflection_step(reasoning_context, reasoning_chain)
            reasoning_chain.append(reflection)
            total_llm_tokens += reflection.tokens_used
            
            # Check if reflection indicates completion
            if reflection.confidence >= self.confidence_threshold:
                self.logger.info(f"Confidence threshold reached: {reflection.confidence}")
                task_complete = True
            elif "SUFFICIENT" in reflection.content.upper():
                task_complete = True
            
            # Cost safety check
            if reasoning_context["cost_so_far"] > self.cost_threshold:
                self.logger.warning(f"Cost threshold exceeded: ${reasoning_context['cost_so_far']:.4f}")
                break
        
        # Generate final answer using gathered information
        final_answer = self._generate_final_answer(user_query, reasoning_context, reasoning_chain)
        total_llm_tokens += final_answer.tokens_used
        reasoning_chain.append(final_answer)
        
        # Log the final generated response for debugging
        self.logger.info(f"\n🎯 FINAL GENERATED RESPONSE:")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Query: {user_query}")
        self.logger.info(f"Final Answer: {final_answer.result}")
        self.logger.info(f"Confidence: {final_answer.confidence:.2f}")
        self.logger.info(f"{'='*60}")
        
        # Build comprehensive response
        execution_time = time.time() - start_time
        sources_queried = [action.source for action in reasoning_chain if action.source and action.result]
        
        response = AgenticResponse(
            final_answer=final_answer.result,
            reasoning_chain=reasoning_chain,
            total_reasoning_steps=len(reasoning_chain),
            sources_queried=sources_queried,
            total_execution_time=execution_time,
            total_llm_tokens=total_llm_tokens,
            confidence_score=final_answer.confidence,
            reasoning_quality=self._assess_reasoning_quality(reasoning_chain),
            cost_breakdown=self._calculate_cost_breakdown(reasoning_chain, total_llm_tokens)
        )
        
        # COMPREHENSIVE DEBUG SUMMARY
        self.logger.info(f"\n🎉 TRUE AGENTIC REASONING COMPLETE")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Total steps: {len(reasoning_chain)}")
        self.logger.info(f"Execution time: {execution_time:.2f}s")
        self.logger.info(f"LLM tokens: {total_llm_tokens}")
        self.logger.info(f"Sources queried: {[s.value for s in sources_queried]}")
        self.logger.info(f"Final confidence: {final_answer.confidence:.2f}")
        self.logger.info(f"Total cost: ${self.total_cost:.4f}")
        
        # Step-by-step summary
        self.logger.info(f"\n📋 REASONING CHAIN SUMMARY:")
        for i, action in enumerate(reasoning_chain, 1):
            if action.step == ReasoningStep.THOUGHT:
                decision = "CONTINUE" if "CONTINUE" in action.content.upper() else "COMPLETE"
                self.logger.info(f"  {i}. THOUGHT → {decision}")
            elif action.step == ReasoningStep.ACTION:
                source = action.source.value if action.source else "none"
                self.logger.info(f"  {i}. ACTION → {source}")
            elif action.step == ReasoningStep.OBSERVATION:
                if action.source:
                    success = "✅" if action.result else "❌"
                    self.logger.info(f"  {i}. OBSERVATION → {action.source.value} {success}")
                else:
                    self.logger.info(f"  {i}. GENERATE → Final answer")
            elif action.step == ReasoningStep.REFLECTION:
                assessment = "CONTINUE" if "CONTINUE" in action.content.upper() else "SUFFICIENT"
                self.logger.info(f"  {i}. REFLECTION → {assessment} (conf: {action.confidence:.2f})")
        
        self.logger.info(f"{'='*80}")
        
        return response
    
    def _llm_thought_step(self, context: Dict, step_number: int) -> ReasoningAction:
        """
        THOUGHT step: LLM analyzes current situation and plans next action.
        """
        available_sources_desc = self._format_sources_for_prompt(context["available_sources"])
        results_summary = self._format_results_summary(context["results_gathered"])
        
        prompt = f"""You are an intelligent reasoning agent orchestrating a multi-source RAG system.

Query: "{context['query']}"
Reasoning Step: {step_number}

Available Sources:
{available_sources_desc}

Results Gathered So Far:
{results_summary}

Previous Reasoning:
{self._format_reasoning_history(context.get('reasoning_history', []))}

THOUGHT: Analyze the current situation and plan your next action.

Consider:
1. What information do we have so far?
2. What information is still needed to answer the query well?
3. Which source (if any) would be most valuable to query next?
4. Is the current information sufficient to provide a good answer?

Respond with either:
- CONTINUE: [reasoning for why more information is needed and which source to try]
- COMPLETE: [reasoning for why current information is sufficient]

Be cost-conscious and efficient. Only continue if additional sources would significantly improve the answer."""

        # DEBUG LOGGING
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🧠 LLM THOUGHT STEP {step_number}")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Query: {context['query']}")
        self.logger.info(f"Available sources: {list(context['available_sources'].keys())}")
        self.logger.info(f"Results so far: {list(context['results_gathered'].keys())}")
        self.logger.info(f"\n📝 PROMPT TO LLM:")
        self.logger.info(f"{prompt}")

        try:
            response = self._call_llm(prompt)
            
            # DEBUG LOGGING - LLM RESPONSE
            self.logger.info(f"\n🤖 LLM RESPONSE:")
            self.logger.info(f"{response['content']}")
            self.logger.info(f"Tokens used: {response['tokens']}")
            self.logger.info(f"Cost: ${response.get('cost', 0):.4f}")
            
            # Analyze decision
            decision = "CONTINUE" if "CONTINUE" in response["content"].upper() else "COMPLETE"
            self.logger.info(f"\n🎯 LLM DECISION: {decision}")
            
            return ReasoningAction(
                step=ReasoningStep.THOUGHT,
                content=response["content"],
                confidence=0.7,  # Moderate confidence for planning
                reasoning="LLM analysis of current state and planning",
                tokens_used=response["tokens"],
                timestamp=time.time()
            )
        except Exception as e:
            self.logger.error(f"❌ LLM thought step failed: {e}")
            return ReasoningAction(
                step=ReasoningStep.THOUGHT,
                content="COMPLETE: Error in reasoning, proceeding with available information",
                confidence=0.3,
                reasoning=f"Error fallback: {str(e)}",
                timestamp=time.time()
            )
    
    def _llm_action_step(self, context: Dict, thought: ReasoningAction) -> ReasoningAction:
        """
        ACTION step: LLM decides which specific source to query next.
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"⚡ LLM ACTION STEP")
        self.logger.info(f"{'='*60}")
        
        if "COMPLETE" in thought.content.upper():
            self.logger.info(f"🛑 THOUGHT decided COMPLETE - no action needed")
            return ReasoningAction(
                step=ReasoningStep.ACTION,
                content="No action needed - proceeding to final answer generation",
                confidence=thought.confidence,
                reasoning="Task marked complete by thought step",
                timestamp=time.time()
            )
        
        available_sources = [
            source_type for source_type, capability in context["available_sources"].items()
            if source_type.value not in context["results_gathered"]
        ]
        
        self.logger.info(f"Available unqueried sources: {[s.value for s in available_sources]}")
        
        if not available_sources:
            self.logger.info(f"🚫 No more sources available - all have been queried")
            return ReasoningAction(
                step=ReasoningStep.ACTION,
                content="No more sources available",
                confidence=0.8,
                reasoning="All available sources have been queried",
                timestamp=time.time()
            )
        
        sources_desc = "\n".join([
            f"- {source.value}: {context['available_sources'][source].description}"
            for source in available_sources
        ])
        
        prompt = f"""Based on the thought analysis, select the best source to query next.

Query: "{context['query']}"
Thought Analysis: {thought.content}

Available Unqueried Sources:
{sources_desc}

Which source would be most valuable for this specific query?

Respond with:
SOURCE: [source_name]
REASONING: [why this source is optimal for the query]

Choose based on:
1. Relevance to the query type and content
2. Likelihood of providing new valuable information
3. Cost-effectiveness

Source options: {[s.value for s in available_sources]}"""

        self.logger.info(f"\n📝 PROMPT TO LLM:")
        self.logger.info(f"{prompt}")

        try:
            response = self._call_llm(prompt)
            content = response["content"]
            
            # DEBUG LOGGING - LLM RESPONSE
            self.logger.info(f"\n🤖 LLM RESPONSE:")
            self.logger.info(f"{content}")
            self.logger.info(f"Tokens used: {response['tokens']}")
            self.logger.info(f"Cost: ${response.get('cost', 0):.4f}")
            
            # Extract selected source
            selected_source = None
            for source in available_sources:
                if source.value in content.lower():
                    selected_source = source
                    break
            
            self.logger.info(f"\n🎯 LLM SOURCE SELECTION: {selected_source.value if selected_source else 'NONE DETECTED'}")
            if not selected_source:
                self.logger.warning(f"⚠️ Could not extract source from LLM response!")
            
            return ReasoningAction(
                step=ReasoningStep.ACTION,
                content=content,
                source=selected_source,
                confidence=0.8,
                reasoning="LLM source selection based on query analysis",
                tokens_used=response["tokens"],
                timestamp=time.time()
            )
        except Exception as e:
            # Fallback to first available source
            fallback_source = available_sources[0] if available_sources else None
            self.logger.error(f"❌ LLM action step failed: {e}")
            self.logger.info(f"🔄 Falling back to: {fallback_source.value if fallback_source else 'none'}")
            
            return ReasoningAction(
                step=ReasoningStep.ACTION,
                content=f"Fallback to {fallback_source.value if fallback_source else 'none'}",
                source=fallback_source,
                confidence=0.4,
                reasoning=f"Error fallback: {str(e)}",
                timestamp=time.time()
            )
    
    def _execute_source_query(self, source: SourceType, query: str, context: Dict) -> ReasoningAction:
        """
        OBSERVATION step: Execute query against selected source.
        """
        start_time = time.time()
        
        try:
            if source == SourceType.TEXT_RAG and self.rag_system:
                result = self.rag_system.query(query)
                observation_content = f"Text RAG returned: {len(result.get('chunks', []))} chunks"
                
            elif source == SourceType.COLPALI_VISUAL and self.colpali_retriever:
                result = self.colpali_retriever.query(query)
                observation_content = f"ColPali returned visual analysis results"
                
            elif source == SourceType.SALESFORCE and self.salesforce_connector:
                # Use correct Salesforce connector method
                try:
                    result = self.salesforce_connector.search_knowledge_realtime(query, limit=5)
                    observation_content = f"Salesforce returned: {len(result)} knowledge articles"
                except Exception as sf_error:
                    self.logger.error(f"Salesforce query failed: {sf_error}")
                    result = []
                    observation_content = f"Salesforce query failed: {str(sf_error)}"
                
            else:
                result = None
                observation_content = f"Source {source.value} not available"
            
            execution_time = time.time() - start_time
            
            return ReasoningAction(
                step=ReasoningStep.OBSERVATION,
                content=observation_content,
                source=source,
                result=result,
                confidence=0.8 if result else 0.1,
                reasoning=f"Retrieved information from {source.value} in {execution_time:.2f}s",
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Source query failed for {source.value}: {e}")
            return ReasoningAction(
                step=ReasoningStep.OBSERVATION,
                content=f"Error querying {source.value}: {str(e)}",
                source=source,
                result=None,
                confidence=0.1,
                reasoning=f"Query execution failed: {str(e)}",
                timestamp=time.time()
            )
    
    def _llm_reflection_step(self, context: Dict, reasoning_chain: List[ReasoningAction]) -> ReasoningAction:
        """
        REFLECTION step: LLM evaluates completeness and decides next steps.
        """
        results_summary = self._format_results_summary(context["results_gathered"])
        recent_actions = reasoning_chain[-3:] if len(reasoning_chain) >= 3 else reasoning_chain
        remaining_sources = [s.value for s in context['available_sources'].keys() if s.value not in context['results_gathered']]
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🤔 LLM REFLECTION STEP")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Results gathered: {list(context['results_gathered'].keys())}")
        self.logger.info(f"Remaining sources: {remaining_sources}")
        
        prompt = f"""Evaluate the information gathered so far and decide if it's sufficient.

Original Query: "{context['query']}"

Information Gathered:
{results_summary}

Recent Actions:
{self._format_actions_for_prompt(recent_actions)}

Available Remaining Sources: {remaining_sources}

REFLECTION: Evaluate the completeness of information for answering the query.

Consider:
1. Can the query be answered well with current information?
2. Would additional sources add significant value?
3. Is the information comprehensive enough for the user's needs?
4. What's the confidence level in providing a good answer?

Respond with:
ASSESSMENT: SUFFICIENT/CONTINUE
CONFIDENCE: [0.0 to 1.0]
REASONING: [detailed explanation of your assessment]

Be efficient - only continue if additional sources would substantially improve the answer quality."""

        self.logger.info(f"\n📝 PROMPT TO LLM:")
        self.logger.info(f"{prompt}")

        try:
            response = self._call_llm(prompt)
            content = response["content"]
            
            # DEBUG LOGGING - LLM RESPONSE
            self.logger.info(f"\n🤖 LLM RESPONSE:")
            self.logger.info(f"{content}")
            self.logger.info(f"Tokens used: {response['tokens']}")
            self.logger.info(f"Cost: ${response.get('cost', 0):.4f}")
            
            # Extract confidence score and assessment
            confidence = 0.5  # default
            assessment = "SUFFICIENT"  # default
            
            if "CONFIDENCE:" in content:
                try:
                    conf_line = [line for line in content.split('\n') if 'CONFIDENCE:' in line][0]
                    confidence = float(conf_line.split(':')[1].strip())
                except:
                    pass
            
            if "ASSESSMENT:" in content:
                try:
                    assess_line = [line for line in content.split('\n') if 'ASSESSMENT:' in line][0]
                    assessment = assess_line.split(':')[1].strip().upper()
                except:
                    pass
            elif "CONTINUE" in content.upper():
                assessment = "CONTINUE"
            
            self.logger.info(f"\n🎯 LLM ASSESSMENT: {assessment}")
            self.logger.info(f"🎯 LLM CONFIDENCE: {confidence:.2f}")
            
            return ReasoningAction(
                step=ReasoningStep.REFLECTION,
                content=content,
                confidence=confidence,
                reasoning="LLM evaluation of information completeness",
                tokens_used=response["tokens"],
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"❌ LLM reflection step failed: {e}")
            return ReasoningAction(
                step=ReasoningStep.REFLECTION,
                content="SUFFICIENT: Error in reflection, proceeding with available information",
                confidence=0.5,
                reasoning=f"Error fallback: {str(e)}",
                timestamp=time.time()
            )
    
    def _generate_final_answer(self, query: str, context: Dict, reasoning_chain: List[ReasoningAction]) -> ReasoningAction:
        """
        Generate final answer by synthesizing all gathered information.
        """
        results_summary = self._format_results_summary(context["results_gathered"])
        reasoning_summary = self._format_reasoning_chain(reasoning_chain)
        
        prompt = f"""Synthesize a comprehensive answer using all gathered information.

Original Query: "{query}"

Information Sources Used:
{results_summary}

Reasoning Process:
{reasoning_summary}

Generate a clear, comprehensive answer that:
1. Directly addresses the user's query
2. Synthesizes information from all relevant sources
3. Provides context and explanations where helpful
4. Acknowledges any limitations or uncertainty

Write a natural, helpful response that makes the best use of the gathered information."""

        try:
            response = self._call_llm(prompt)
            
            return ReasoningAction(
                step=ReasoningStep.OBSERVATION,  # Final generation step
                content="Final answer generated",
                result=response["content"], 
                confidence=0.8,
                reasoning="Synthesized final answer from all gathered information",
                tokens_used=response["tokens"],
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Final answer generation failed: {e}")
            return ReasoningAction(
                step=ReasoningStep.OBSERVATION,
                content="Final answer generation failed",
                result="I apologize, but I encountered an error generating the response. Please try again.",
                confidence=0.1,
                reasoning=f"Generation error: {str(e)}",
                timestamp=time.time()
            )
    
    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Make API call to LLM with cost and token tracking.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an intelligent reasoning agent for a multi-source RAG system. Be analytical, efficient, and transparent in your reasoning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent reasoning
                max_tokens=1000   # Reasonable limit for reasoning steps
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # Update cost tracking (GPT-4o Mini pricing)
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens * 0.00015 / 1000) + (output_tokens * 0.0006 / 1000)  # GPT-4o Mini pricing
            
            self.total_cost += cost
            self.total_tokens += tokens_used
            
            return {
                "content": content,
                "tokens": tokens_used,
                "cost": cost
            }
            
        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            raise
    
    def _format_sources_for_prompt(self, sources: Dict[SourceType, SourceCapability]) -> str:
        """Format source capabilities for LLM prompt."""
        if not sources:
            return "No sources available"
        
        formatted = []
        for source_type, capability in sources.items():
            formatted.append(f"""- {capability.name} ({source_type.value}):
  Description: {capability.description}
  Best for: {', '.join(capability.best_for)}
  Response time: {capability.typical_response_time}
  Cost: {capability.cost_level}""")
        
        return "\n".join(formatted)
    
    def _format_results_summary(self, results: Dict[str, Any]) -> str:
        """Format gathered results for LLM prompt."""
        if not results:
            return "No information gathered yet"
        
        formatted = []
        for source, result in results.items():
            if result:
                summary = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                formatted.append(f"- {source}: {summary}")
            else:
                formatted.append(f"- {source}: No results")
        
        return "\n".join(formatted)
    
    def _format_reasoning_history(self, history: List[str]) -> str:
        """Format reasoning history for context."""
        if not history:
            return "No previous reasoning steps"
        return "\n".join([f"{i+1}. {step}" for i, step in enumerate(history)])
    
    def _format_actions_for_prompt(self, actions: List[ReasoningAction]) -> str:
        """Format recent actions for prompt context."""
        formatted = []
        for action in actions:
            formatted.append(f"- {action.step.value}: {action.content[:100]}...")
        return "\n".join(formatted)
    
    def _format_reasoning_chain(self, chain: List[ReasoningAction]) -> str:
        """Format complete reasoning chain for final synthesis."""
        formatted = []
        for i, action in enumerate(chain, 1):
            formatted.append(f"{i}. {action.step.value.upper()}: {action.content[:150]}...")
        return "\n".join(formatted)
    
    def _assess_reasoning_quality(self, chain: List[ReasoningAction]) -> str:
        """Assess the quality of the reasoning process."""
        if len(chain) < 3:
            return "insufficient"
        elif len(chain) > 15:
            return "excessive"
        else:
            return "good"
    
    def _calculate_cost_breakdown(self, chain: List[ReasoningAction], total_llm_tokens: int) -> Dict[str, float]:
        """Calculate detailed cost breakdown."""
        llm_cost = sum(getattr(action, 'cost', 0) for action in chain)
        return {
            "llm_reasoning": llm_cost,
            "source_queries": 0.01 * len([a for a in chain if a.source]),  # Estimated
            "total": llm_cost + 0.01 * len([a for a in chain if a.source])
        }
    
    def get_cost_stats(self) -> Dict[str, Any]:
        """Get cumulative cost statistics."""
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "average_cost_per_query": self.total_cost / max(1, len(getattr(self, '_query_count', 1))),
            "cost_threshold": self.cost_threshold
        }

# Example usage and testing
if __name__ == "__main__":
    # Example initialization
    logging.basicConfig(level=logging.INFO)
    
    # This would be initialized with actual components in real usage
    agent = LLMReasoningAgent(
        model_name="gpt-4o-mini",
        max_reasoning_steps=8,
        confidence_threshold=0.75
    )
    
    print("🧠 True LLM Reasoning Agent initialized successfully!")
    print(f"Model: {agent.model_name}")
    print(f"Available sources: {list(agent.get_available_sources().keys())}")
    print("Ready for genuine agentic reasoning!")