"""
True Agentic RAG Orchestrator using ReAct Framework

This implements true agentic behavior where an LLM agent reasons through
retrieval strategy dynamically, rather than using brittle keyword-based routing.

The agent follows the ReAct (Reasoning and Acting) pattern:
1. Think - Reason about the query and plan approach
2. Act - Execute specific retrieval actions  
3. Observe - Evaluate results and decide next steps
4. Repeat until sufficient information is gathered
5. Synthesize - Generate final response

Example agent reasoning flow:
<think>
User is asking "what's the retrieval time in ColPali pipeline?"
This could refer to performance benchmarks (visual charts) or technical docs (text).
I should start with visual search since performance data often appears in charts.
</think>
<action>search_colpali</action>
<observation>Found visual content with performance metrics, confidence: 0.85</observation>
<think>
Good visual match. Let me check text sources for detailed technical specs.
</think>
<action>search_text</action>
<observation>Found technical documentation, confidence: 0.72</observation>
<think>
I have both visual and text sources. I can now synthesize a comprehensive answer.
</think>
"""

import os
import sys
import asyncio
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import openai
from pathlib import Path

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import existing components
try:
    from src.rag_system import RAGSystem
    from src.colpali_retriever import ColPaliRetriever  
    from src.salesforce_connector import SalesforceConnector
except ImportError as e:
    logging.error(f"Failed to import components: {e}")
    raise

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgenticRAGOrchestrator:
    """
    True agentic RAG system using ReAct framework.
    
    Unlike rule-based routing, this uses LLM reasoning to:
    - Analyze each query individually
    - Plan optimal retrieval strategy  
    - Execute retrieval actions
    - Evaluate results and adapt approach
    - Generate transparent reasoning traces
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize agentic orchestrator with existing high-quality components."""
        self.config = config
        
        # Initialize OpenAI for agent reasoning (separate from retrieval synthesis)
        self.reasoning_llm = self._init_reasoning_llm()
        
        # Initialize existing high-quality retrievers
        logger.info("🧠 Initializing agentic RAG components...")
        try:
            # Keep proven text RAG with LLM synthesis
            self.text_rag = RAGSystem(config)
            
            # Enhanced ColPali retriever with proper MaxSim scoring
            from src.colpali_maxsim_retriever import ColPaliMaxSimRetriever
            self.colpali_retriever = ColPaliMaxSimRetriever(config)
            
            # Enhanced Salesforce connector
            self.salesforce_connector = SalesforceConnector()
            
            logger.info("✅ Agentic RAG components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise
        
        # Agentic reasoning settings
        self.max_reasoning_steps = config.get('max_reasoning_steps', 5)
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.7)
        
        # Track reasoning history for transparency
        self.reasoning_history = []
        
        logger.info(f"🤖 Agentic RAG orchestrator ready (max steps: {self.max_reasoning_steps})")
    
    def _init_reasoning_llm(self) -> openai.OpenAI:
        """Initialize OpenAI client for agent reasoning."""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found in environment")
            
            client = openai.OpenAI(api_key=api_key)
            logger.info("✅ Reasoning LLM initialized (GPT-4o-mini)")
            return client
        except Exception as e:
            logger.error(f"❌ Failed to initialize reasoning LLM: {e}")
            raise
    
    async def process_query_agentically(self, query: str) -> Dict[str, Any]:
        """
        Main agentic processing using ReAct framework.
        
        This is where the magic happens - the LLM agent reasons through
        the retrieval strategy step by step, adapting based on results.
        """
        logger.info(f"🧠 Starting agentic processing for: '{query}'")
        
        # Clear previous reasoning history
        self.reasoning_history = []
        
        try:
            # Step 1: Initial reasoning about the query
            logger.info("💭 Step 1: Agent analyzing query and planning approach...")
            initial_reasoning = await self._think_about_query(query)
            self.reasoning_history.append(initial_reasoning)
            
            # Step 2-N: Iterative reasoning and retrieval
            logger.info("🔄 Step 2+: Agent executing retrieval strategy...")
            for step_num in range(self.max_reasoning_steps):
                logger.info(f"   Reasoning step {step_num + 1}/{self.max_reasoning_steps}")
                
                # Agent decides what action to take next
                action_decision = await self._decide_next_action(query, self.reasoning_history)
                self.reasoning_history.append({
                    'step_type': 'action_decision',
                    'step_number': step_num + 1,
                    **action_decision
                })
                
                if action_decision['action'] == 'synthesize':
                    logger.info("✅ Agent decided to synthesize - sufficient information gathered")
                    break
                
                # Execute retrieval action
                logger.info(f"🔍 Executing action: {action_decision['action']}")
                retrieval_results = await self._execute_retrieval_action(
                    query, action_decision
                )
                
                # Agent observes and evaluates results
                observation = await self._observe_results(query, retrieval_results)
                self.reasoning_history.append(observation)
                
                # Check if we have sufficient information
                if observation.get('confidence_score', 0) >= self.min_confidence_threshold:
                    logger.info(f"✅ Confidence threshold met: {observation['confidence_score']:.2f}")
                    break
            
            # Final synthesis using existing high-quality approach
            logger.info("🎯 Synthesizing final response...")
            final_answer = await self._synthesize_agentic_response(query, self.reasoning_history)
            
            logger.info(f"✅ Agentic processing completed with {len(self.reasoning_history)} reasoning steps")
            
            return {
                'success': True,
                'answer': final_answer['content'],
                'reasoning_trace': self.reasoning_history,
                'sources_used': final_answer.get('sources', []),
                'agent_decisions': [step.get('decision_summary') for step in self.reasoning_history if 'decision_summary' in step],
                'final_confidence': final_answer.get('confidence', 0.0),
                'total_reasoning_steps': len(self.reasoning_history)
            }
            
        except Exception as e:
            logger.error(f"❌ Agentic processing failed: {e}")
            return {
                'success': False,
                'error': f"Agentic processing failed: {str(e)}",
                'reasoning_trace': self.reasoning_history
            }
    
    async def _think_about_query(self, query: str) -> Dict[str, Any]:
        """
        Agent's initial reasoning about the query and retrieval strategy.
        
        This is like the <think> step in your example - the agent analyzes
        what the user is asking and plans the best approach.
        """
        thinking_prompt = f"""You are an intelligent RAG agent analyzing a user query to plan an optimal retrieval strategy.

<query>{query}</query>

<available_sources>
1. TEXT_RAG: High-quality text documents with proven LLM synthesis for detailed content extraction
2. COLPALI_VISUAL: Visual document analysis using ColPali for charts, diagrams, tables, figures  
3. SALESFORCE: Business knowledge base with policies, procedures, workflows, customer service info
</available_sources>

<task>
Analyze this query and reason about the best retrieval strategy. Consider:
1. What type of information is the user seeking?
2. Which sources would most likely contain this information?
3. Are there multiple aspects to this question requiring different sources?
4. What's the logical order to search for information?
5. How can I determine if I have sufficient information to answer?
</task>

Respond EXACTLY in this format:
<think>
[Your detailed reasoning about the query, what information is needed, and why]
</think>
<plan>
[Specific plan for which sources to query, in what order, and success criteria]
</plan>
<confidence>
[Your initial confidence (0.0-1.0) in being able to answer this query with available sources]
</confidence>"""

        try:
            response = await self._call_reasoning_llm(thinking_prompt)
            
            return {
                'step_type': 'initial_reasoning',
                'query': query,
                'reasoning': self._extract_section(response, 'think'),
                'plan': self._extract_section(response, 'plan'),
                'initial_confidence': float(self._extract_section(response, 'confidence') or '0.5'),
                'timestamp': datetime.now().isoformat(),
                'decision_summary': f"Analyzed query and planned retrieval approach"
            }
        except Exception as e:
            logger.error(f"❌ Initial reasoning failed: {e}")
            # Fallback to basic reasoning
            return {
                'step_type': 'initial_reasoning',
                'query': query,
                'reasoning': f"Failed to generate detailed reasoning: {e}. Will try all sources.",
                'plan': "Query text, visual, and business sources systematically.",
                'initial_confidence': 0.5,
                'timestamp': datetime.now().isoformat(),
                'decision_summary': "Fallback to comprehensive search"
            }
    
    async def _decide_next_action(self, query: str, history: List[Dict]) -> Dict[str, Any]:
        """
        Agent decides what retrieval action to take next based on current knowledge.
        
        This implements the decision-making part of the ReAct cycle.
        """
        context = self._build_context_from_history(history)
        
        decision_prompt = f"""You are an intelligent RAG agent deciding the next retrieval action.

<query>{query}</query>
<current_context>
{context}
</current_context>

<available_actions>
1. search_text - Query text documents for detailed textual information
2. search_visual - Query visual documents (PDFs) for charts, diagrams, tables
3. search_salesforce - Query business knowledge base for policies, procedures
4. search_multiple - Query multiple sources simultaneously for comprehensive coverage
5. synthesize - Sufficient information gathered, ready to generate final answer
</available_actions>

<task>
Based on the query and what we've learned so far, decide the next action:
- What information gaps still exist?
- Which sources haven't been tried that might help?
- Is the current information sufficient for a comprehensive answer?
- What's the most logical next step?
</task>

Respond EXACTLY in this format:
<think>
[Your reasoning about what to do next and why]
</think>
<action>search_text|search_visual|search_salesforce|search_multiple|synthesize</action>
<target_sources>
[If action is search_*, specify which specific aspects to search for]
</target_sources>
<reasoning>
[Why this action will help answer the query and address information gaps]
</reasoning>
<expected_confidence>
[Expected confidence level (0.0-1.0) after this action]
</expected_confidence>"""

        try:
            response = await self._call_reasoning_llm(decision_prompt)
            
            return {
                'thinking': self._extract_section(response, 'think'),
                'action': self._extract_section(response, 'action'),
                'target_sources': self._extract_section(response, 'target_sources'),
                'reasoning': self._extract_section(response, 'reasoning'),
                'expected_confidence': float(self._extract_section(response, 'expected_confidence') or '0.5'),
                'timestamp': datetime.now().isoformat(),
                'decision_summary': f"Decided to {self._extract_section(response, 'action')}"
            }
        except Exception as e:
            logger.error(f"❌ Action decision failed: {e}")
            # Fallback to systematic search
            return {
                'thinking': f"Decision making failed: {e}. Defaulting to text search.",
                'action': 'search_text',
                'target_sources': 'General text search',
                'reasoning': 'Fallback action due to reasoning failure',
                'expected_confidence': 0.5,
                'timestamp': datetime.now().isoformat(),
                'decision_summary': "Fallback to text search"
            }
    
    async def _execute_retrieval_action(self, query: str, action_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the retrieval action decided by the agent.
        
        This calls the appropriate retriever(s) based on the agent's decision.
        """
        action = action_decision['action']
        target_sources = action_decision.get('target_sources', '')
        
        try:
            if action == 'search_text':
                logger.info("📝 Executing text search...")
                # Use existing proven text RAG with context
                enhanced_query = f"{query}\n\nContext: {target_sources}"
                results = self.text_rag.query(enhanced_query)
                
                return {
                    'action_executed': 'search_text',
                    'results': results,
                    'source_type': 'text',
                    'success': results.get('success', False)
                }
            
            elif action == 'search_visual':
                logger.info("🖼️ Executing visual search...")
                # Use ColPali for visual search (will need MaxSim fixes)
                results = await self._search_visual_with_context(query, target_sources)
                
                return {
                    'action_executed': 'search_visual', 
                    'results': results,
                    'source_type': 'visual',
                    'success': len(results.get('documents', [])) > 0
                }
            
            elif action == 'search_salesforce':
                logger.info("🏢 Executing Salesforce search...")
                # Use enhanced Salesforce connector
                results = await self._search_salesforce_with_context(query, target_sources)
                
                return {
                    'action_executed': 'search_salesforce',
                    'results': results,
                    'source_type': 'salesforce', 
                    'success': len(results.get('articles', [])) > 0
                }
            
            elif action == 'search_multiple':
                logger.info("🔍 Executing multi-source search...")
                # Search multiple sources in parallel
                text_task = asyncio.create_task(self._search_text_async(query, target_sources))
                visual_task = asyncio.create_task(self._search_visual_with_context(query, target_sources))
                sf_task = asyncio.create_task(self._search_salesforce_with_context(query, target_sources))
                
                text_results, visual_results, sf_results = await asyncio.gather(
                    text_task, visual_task, sf_task, return_exceptions=True
                )
                
                return {
                    'action_executed': 'search_multiple',
                    'results': {
                        'text': text_results if not isinstance(text_results, Exception) else None,
                        'visual': visual_results if not isinstance(visual_results, Exception) else None,
                        'salesforce': sf_results if not isinstance(sf_results, Exception) else None
                    },
                    'source_type': 'multiple',
                    'success': True
                }
            
            else:
                return {
                    'action_executed': action,
                    'results': {},
                    'source_type': 'none',
                    'success': False,
                    'error': f"Unknown action: {action}"
                }
                
        except Exception as e:
            logger.error(f"❌ Retrieval action failed: {e}")
            return {
                'action_executed': action,
                'results': {},
                'source_type': action.replace('search_', ''),
                'success': False,
                'error': str(e)
            }
    
    async def _observe_results(self, query: str, retrieval_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agent evaluates retrieval results and decides if they're sufficient.
        
        This is the observation step in the ReAct cycle where the agent
        examines what it found and decides whether to continue searching.
        """
        results_summary = self._format_results_for_evaluation(retrieval_results)
        
        observation_prompt = f"""You are an intelligent RAG agent evaluating retrieval results.

<query>{query}</query>
<retrieval_results>
{results_summary}
</retrieval_results>

<task>
Evaluate these retrieval results:
1. How well do they address the original query?
2. What specific information was found?
3. What information is still missing or unclear?
4. Are the results high quality and relevant?
5. Should we search additional sources or do we have enough?
</task>

Respond EXACTLY in this format:
<think>
[Your detailed evaluation of the results and what they tell us]
</think>
<quality>low|medium|high</quality>
<information_found>
[Specific information that addresses the query]
</information_found>
<information_gaps>
[What information is still missing to fully answer the query]
</information_gaps>
<confidence>
[Your confidence (0.0-1.0) in being able to answer the query with current information]
</confidence>
<recommendation>
continue_search|sufficient_info
</recommendation>"""

        try:
            response = await self._call_reasoning_llm(observation_prompt)
            
            confidence_score = float(self._extract_section(response, 'confidence') or '0.5')
            
            return {
                'step_type': 'observation',
                'results_quality': self._extract_section(response, 'quality'),
                'information_found': self._extract_section(response, 'information_found'),
                'information_gaps': self._extract_section(response, 'information_gaps'),
                'confidence_score': confidence_score,
                'recommendation': self._extract_section(response, 'recommendation'),
                'thinking': self._extract_section(response, 'think'),
                'raw_results': retrieval_results,
                'timestamp': datetime.now().isoformat(),
                'decision_summary': f"Evaluated results - confidence: {confidence_score:.2f}"
            }
        except Exception as e:
            logger.error(f"❌ Result observation failed: {e}")
            return {
                'step_type': 'observation',
                'results_quality': 'unknown',
                'information_found': 'Unable to evaluate due to error',
                'information_gaps': 'Unknown gaps due to evaluation failure',
                'confidence_score': 0.3,
                'recommendation': 'continue_search',
                'thinking': f"Observation failed: {e}",
                'raw_results': retrieval_results,
                'timestamp': datetime.now().isoformat(),
                'decision_summary': "Evaluation failed - continuing search"
            }
    
    async def _synthesize_agentic_response(self, query: str, reasoning_history: List[Dict]) -> Dict[str, Any]:
        """
        Generate final response using agent's gathered information.
        
        This preserves the existing high-quality response generation while
        incorporating the agent's reasoning context.
        """
        # Extract all information gathered by the agent
        all_sources = []
        agent_insights = []
        
        for step in reasoning_history:
            if step.get('step_type') == 'observation':
                info_found = step.get('information_found', '')
                if info_found and info_found != 'Unable to evaluate due to error':
                    agent_insights.append(info_found)
                
                # Extract actual retrieval results
                raw_results = step.get('raw_results', {})
                if raw_results.get('success'):
                    all_sources.append({
                        'content': raw_results.get('results', {}),
                        'source_type': raw_results.get('source_type', 'unknown'),
                        'confidence': step.get('confidence_score', 0.5)
                    })
        
        # Create synthesis prompt incorporating agent reasoning
        agent_context = "\n".join([f"- {insight}" for insight in agent_insights if insight])
        
        synthesis_prompt = f"""You are synthesizing a comprehensive answer based on agentic retrieval.

<query>{query}</query>

<agent_insights>
{agent_context}
</agent_insights>

<retrieved_sources>
{self._format_sources_for_synthesis(all_sources)}
</retrieved_sources>

<task>
Generate a comprehensive, accurate answer that:
1. Directly addresses the user's query
2. Incorporates insights from the agent's reasoning process
3. Uses information from multiple sources when available
4. Maintains high quality and accuracy
5. Provides specific details rather than generic responses
</task>

Generate a clear, detailed answer based on the available information."""

        try:
            response = await self._call_reasoning_llm(synthesis_prompt)
            
            return {
                'content': response,
                'sources': all_sources,
                'confidence': max([src.get('confidence', 0.0) for src in all_sources] + [0.5]),
                'synthesis_type': 'agentic'
            }
        except Exception as e:
            logger.error(f"❌ Response synthesis failed: {e}")
            return {
                'content': f"I gathered information from multiple sources but encountered an error during synthesis: {e}",
                'sources': all_sources,
                'confidence': 0.3,
                'synthesis_type': 'fallback'
            }
    
    # Helper methods
    
    async def _call_reasoning_llm(self, prompt: str) -> str:
        """Call the reasoning LLM with error handling."""
        try:
            response = self.reasoning_llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intelligent RAG agent that reasons step-by-step about information retrieval. Always follow the exact format requested."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=800,
                temperature=0.1  # Low temperature for consistent reasoning
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ Reasoning LLM call failed: {e}")
            raise
    
    def _extract_section(self, text: str, section: str) -> str:
        """Extract content between XML-like tags."""
        pattern = f"<{section}>(.*?)</{section}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    def _build_context_from_history(self, history: List[Dict]) -> str:
        """Build context string from reasoning history."""
        context_parts = []
        for step in history:
            if step.get('step_type') == 'initial_reasoning':
                context_parts.append(f"Initial Plan: {step.get('plan', '')}")
            elif step.get('step_type') == 'observation':
                context_parts.append(f"Found: {step.get('information_found', '')}")
                context_parts.append(f"Still Missing: {step.get('information_gaps', '')}")
        
        return "\n".join(context_parts)
    
    def _format_results_for_evaluation(self, retrieval_results: Dict[str, Any]) -> str:
        """Format retrieval results for agent evaluation."""
        if not retrieval_results.get('success', False):
            return f"No results found. Error: {retrieval_results.get('error', 'Unknown error')}"
        
        source_type = retrieval_results.get('source_type', 'unknown')
        results = retrieval_results.get('results', {})
        
        if source_type == 'text':
            return f"Text Results: {results.get('answer', 'No answer')[:200]}..."
        elif source_type == 'visual':
            docs = results.get('documents', [])
            return f"Visual Results: Found {len(docs)} relevant documents"
        elif source_type == 'salesforce':
            articles = results.get('articles', [])
            return f"Salesforce Results: Found {len(articles)} knowledge articles"
        elif source_type == 'multiple':
            summary = []
            if results.get('text'):
                summary.append("Text: Found relevant content")
            if results.get('visual'):
                summary.append("Visual: Found relevant documents")
            if results.get('salesforce'):
                summary.append("Salesforce: Found knowledge articles")
            return "Multiple Sources: " + ", ".join(summary)
        
        return f"Results from {source_type}: {str(results)[:200]}..."
    
    def _format_sources_for_synthesis(self, sources: List[Dict]) -> str:
        """Format sources for synthesis prompt."""
        if not sources:
            return "No sources available"
        
        formatted = []
        for i, source in enumerate(sources, 1):
            source_type = source.get('source_type', 'unknown')
            confidence = source.get('confidence', 0.0)
            content = str(source.get('content', ''))[:300]
            
            formatted.append(f"Source {i} ({source_type}, confidence: {confidence:.2f}):\n{content}")
        
        return "\n\n".join(formatted)
    
    # Placeholder methods for retriever integration
    
    async def _search_visual_with_context(self, query: str, context: str) -> Dict[str, Any]:
        """Search visual documents with agent context using proper MaxSim scoring."""
        try:
            # Enhanced query with agent context
            enhanced_query = f"{query} {context}".strip()
            
            # Use ColPali MaxSim retriever
            maxsim_results = await self.colpali_retriever.query_with_maxsim(enhanced_query, top_k=5)
            
            if maxsim_results:
                # Transform MaxSim results to expected format
                documents = []
                for result in maxsim_results:
                    documents.append({
                        'content': result['content'],
                        'score': result['score'],
                        'metadata': result['metadata'],
                        'source_type': 'visual_maxsim'
                    })
                
                return {
                    'documents': documents,
                    'success': True,
                    'total_found': len(documents),
                    'scoring_method': 'maxsim'
                }
            else:
                return {'documents': [], 'success': False, 'error': 'No visual matches found'}
                
        except Exception as e:
            logger.error(f"❌ Visual search with MaxSim failed: {e}")
            return {'documents': [], 'success': False, 'error': str(e)}
    
    async def _search_salesforce_with_context(self, query: str, context: str) -> Dict[str, Any]:
        """Search Salesforce with agent context using enhanced business term extraction."""
        try:
            # Enhanced query with agent context
            enhanced_query = f"{query} {context}".strip()
            
            # Use Salesforce connector with business term extraction
            sf_results = self.salesforce_connector.search(enhanced_query)
            
            if sf_results and sf_results.get('success', False):
                # Transform Salesforce results to expected format
                articles = []
                for article in sf_results.get('articles', []):
                    articles.append({
                        'content': article.get('content', ''),
                        'title': article.get('title', ''),
                        'score': article.get('score', 0.8),  # Default confidence
                        'metadata': {
                            'source': 'salesforce',
                            'article_id': article.get('id', ''),
                            'url': article.get('url', ''),
                            'type': 'business_knowledge'
                        }
                    })
                
                return {
                    'articles': articles,
                    'success': True,
                    'total_found': len(articles),
                    'search_method': 'business_terms'
                }
            else:
                error_msg = sf_results.get('error', 'No business articles found')
                return {'articles': [], 'success': False, 'error': error_msg}
                
        except Exception as e:
            logger.error(f"❌ Salesforce search with context failed: {e}")
            return {'articles': [], 'success': False, 'error': str(e)}
    
    async def _search_text_async(self, query: str, context: str) -> Dict[str, Any]:
        """Async wrapper for text search."""
        enhanced_query = f"{query}\n\nContext: {context}"
        return self.text_rag.query(enhanced_query)
    
    def generate_reasoning_report(self) -> str:
        """
        Generate complete audit trail of agent decisions.
        
        This provides full transparency into how the agent reasoned
        through the retrieval process.
        """
        if not self.reasoning_history:
            return "No reasoning history available."
        
        report = "# Agent Reasoning Trace\n\n"
        
        for i, step in enumerate(self.reasoning_history, 1):
            step_type = step.get('step_type', 'unknown')
            
            report += f"## Step {i}: {step_type.replace('_', ' ').title()}\n"
            
            if step_type == 'initial_reasoning':
                report += f"**Query Analysis**: {step.get('reasoning', '')}\n\n"
                report += f"**Planned Approach**: {step.get('plan', '')}\n\n"
                report += f"**Initial Confidence**: {step.get('initial_confidence', 0.0):.2f}\n\n"
                
            elif step_type == 'action_decision':
                report += f"**Agent Thinking**: {step.get('thinking', '')}\n\n"
                report += f"**Action Chosen**: {step.get('action', '')}\n\n"
                report += f"**Target Sources**: {step.get('target_sources', '')}\n\n"
                report += f"**Reasoning**: {step.get('reasoning', '')}\n\n"
                
            elif step_type == 'observation':
                report += f"**Result Evaluation**: {step.get('thinking', '')}\n\n"
                report += f"**Quality Assessment**: {step.get('results_quality', '')}\n\n"
                report += f"**Information Found**: {step.get('information_found', '')}\n\n"
                report += f"**Information Gaps**: {step.get('information_gaps', '')}\n\n"
                report += f"**Confidence Score**: {step.get('confidence_score', 0.0):.2f}\n\n"
                report += f"**Recommendation**: {step.get('recommendation', '')}\n\n"
            
            report += "---\n\n"
        
        return report
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            'orchestrator_type': 'agentic_react',
            'max_reasoning_steps': self.max_reasoning_steps,
            'min_confidence_threshold': self.min_confidence_threshold,
            'components': {
                'text_rag': hasattr(self, 'text_rag') and self.text_rag is not None,
                'colpali_retriever': hasattr(self, 'colpali_retriever') and self.colpali_retriever is not None,
                'salesforce_connector': hasattr(self, 'salesforce_connector') and self.salesforce_connector is not None,
                'reasoning_llm': hasattr(self, 'reasoning_llm') and self.reasoning_llm is not None
            },
            'last_reasoning_steps': len(self.reasoning_history),
            'config': self.config
        }


# Factory function for easy creation
def create_agentic_rag_orchestrator(config: Dict[str, Any]) -> AgenticRAGOrchestrator:
    """
    Factory function to create agentic RAG orchestrator.
    
    Example config:
    {
        'chunk_size': 800,
        'chunk_overlap': 150,
        'embedding_model': 'local',
        'max_retrieved_chunks': 5,
        'max_reasoning_steps': 5,
        'min_confidence_threshold': 0.7
    }
    """
    logger.info("🏗️ Creating agentic RAG orchestrator...")
    
    # Validate configuration
    required_keys = ['chunk_size', 'chunk_overlap', 'embedding_model']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Set agentic defaults
    config.setdefault('max_reasoning_steps', 5)
    config.setdefault('min_confidence_threshold', 0.7)
    
    orchestrator = AgenticRAGOrchestrator(config)
    logger.info("🎉 Agentic RAG orchestrator ready!")
    
    return orchestrator


# Example usage for testing
if __name__ == "__main__":
    """Test the agentic orchestrator with example config."""
    test_config = {
        'chunk_size': 800,
        'chunk_overlap': 150,
        'embedding_model': 'local',
        'max_retrieved_chunks': 5,
        'max_reasoning_steps': 3,
        'min_confidence_threshold': 0.6
    }
    
    async def test_agentic_orchestrator():
        try:
            orchestrator = create_agentic_rag_orchestrator(test_config)
            
            # Test query
            test_query = "What is the retrieval time in a ColPali pipeline?"
            
            print(f"🧪 Testing agentic orchestrator with query: '{test_query}'")
            result = await orchestrator.process_query_agentically(test_query)
            
            print(f"✅ Success: {result['success']}")
            print(f"📊 Reasoning steps: {result['total_reasoning_steps']}")
            print(f"🎯 Final confidence: {result['final_confidence']:.2f}")
            print(f"📋 Answer: {result['answer'][:200]}...")
            
            # Show reasoning trace
            print("\n📝 Reasoning Report:")
            print(orchestrator.generate_reasoning_report())
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    # Run test if OpenAI API key is available
    if os.getenv('OPENAI_API_KEY'):
        asyncio.run(test_agentic_orchestrator())
    else:
        print("⚠️ OpenAI API key not found - skipping agentic test")
        print("✅ Agentic orchestrator module loaded successfully")