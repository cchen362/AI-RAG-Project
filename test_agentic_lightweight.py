"""
Lightweight Agentic RAG Test App
Token-efficient testing without emojis or heavy UI
"""

import sys
import os
import time
import logging
from typing import Dict, List, Optional, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_system import RAGSystem
from colpali_retriever import ColPaliRetriever
from salesforce_connector import SalesforceConnector
from cross_encoder_reranker import CrossEncoderReRanker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class AgenticOrchestrator:
    """
    Core agentic orchestrator implementing Graph-R1 concepts:
    - Dynamic source selection based on query analysis
    - Confidence-based stopping criteria
    - Multi-turn think-query-retrieve-rethink loops
    """
    
    def __init__(self,
                 rag_system: Optional[RAGSystem] = None,
                 colpali_retriever: Optional[ColPaliRetriever] = None,
                 salesforce_connector: Optional[SalesforceConnector] = None,
                 reranker: Optional[CrossEncoderReRanker] = None,
                 max_reasoning_steps: int = 6,
                 confidence_threshold: float = 0.8):
        
        self.rag_system = rag_system
        self.colpali_retriever = colpali_retriever
        self.salesforce_connector = salesforce_connector
        self.reranker = reranker
        self.max_reasoning_steps = max_reasoning_steps
        self.confidence_threshold = confidence_threshold
        
        # Track available sources
        self.available_sources = self._identify_available_sources()
        logger.info(f"Available sources: {self.available_sources}")
    
    def _identify_available_sources(self) -> List[str]:
        """Identify which sources are actually available"""
        sources = []
        if self.rag_system:
            sources.append("text_rag")
        if self.colpali_retriever:
            sources.append("colpali_visual")
        if self.salesforce_connector:
            sources.append("salesforce")
        return sources
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Main agentic query processing implementing Graph-R1 reasoning loop
        """
        start_time = time.time()
        reasoning_chain = []
        accumulated_knowledge = {}
        step_count = 0
        
        logger.info(f"Starting agentic query: {user_query}")
        
        # STEP 1: THINK - Analyze query and plan approach
        think_result = self._think_step(user_query)
        reasoning_chain.append(think_result)
        logger.info(f"THINK: {think_result['reasoning']}")
        
        # Multi-turn reasoning loop
        while step_count < self.max_reasoning_steps:
            step_count += 1
            
            # STEP 2: RETRIEVE - Get information from selected source
            retrieve_result = self._retrieve_step(user_query, think_result['plan'], accumulated_knowledge)
            reasoning_chain.append(retrieve_result)
            
            if retrieve_result['success']:
                accumulated_knowledge[retrieve_result['source']] = retrieve_result['content']
                logger.info(f"RETRIEVE #{step_count}: {retrieve_result['source']} - {retrieve_result['content'][:100]}...")
            else:
                logger.info(f"RETRIEVE #{step_count}: {retrieve_result['source']} - FAILED")
            
            # STEP 3: RETHINK - Assess completeness and decide next action
            rethink_result = self._rethink_step(user_query, accumulated_knowledge, reasoning_chain)
            reasoning_chain.append(rethink_result)
            logger.info(f"RETHINK #{step_count}: {rethink_result['assessment']} (confidence: {rethink_result['confidence']:.2f})")
            
            # Check stopping criteria
            if rethink_result['confidence'] >= self.confidence_threshold:
                logger.info(f"Stopping: High confidence reached ({rethink_result['confidence']:.2f})")
                break
            elif rethink_result['should_stop']:
                logger.info(f"Stopping: {rethink_result['stop_reason']}")
                break
        
        # STEP 4: GENERATE - Synthesize final answer
        generate_result = self._generate_step(user_query, accumulated_knowledge, reasoning_chain)
        reasoning_chain.append(generate_result)
        
        execution_time = time.time() - start_time
        
        return {
            'final_answer': generate_result['answer'],
            'reasoning_chain': reasoning_chain,
            'execution_time': execution_time,
            'sources_used': list(accumulated_knowledge.keys()),
            'confidence_score': generate_result['confidence'],
            'total_steps': len(reasoning_chain),
            'approach': 'agentic'
        }
    
    def _think_step(self, query: str) -> Dict[str, Any]:
        """Analyze query and plan retrieval approach"""
        
        # Simple but effective query analysis
        query_lower = query.lower()
        
        # Determine query characteristics
        is_visual_query = any(term in query_lower for term in 
                             ['chart', 'graph', 'diagram', 'figure', 'visualization', 'image'])
        is_business_query = any(term in query_lower for term in 
                               ['sales', 'business', 'company', 'customer', 'revenue', 'profit'])
        is_technical_query = any(term in query_lower for term in 
                                ['algorithm', 'model', 'architecture', 'implementation', 'code'])
        
        # Create plan based on analysis
        plan_components = []
        if is_visual_query and "colpali_visual" in self.available_sources:
            plan_components.append("VISUAL_QUERY")
        if is_business_query and "salesforce" in self.available_sources:
            plan_components.append("BUSINESS_QUERY")
        if is_technical_query or not plan_components:  # Default to text for technical or unclear queries
            if "text_rag" in self.available_sources:
                plan_components.append("TEXT_QUERY")
        
        plan = " + ".join(plan_components) if plan_components else "GENERAL_QUERY"
        
        return {
            'step_type': 'THINK',
            'reasoning': f'Query analysis: visual={is_visual_query}, business={is_business_query}, technical={is_technical_query}',
            'plan': plan,
            'query_characteristics': {
                'visual': is_visual_query,
                'business': is_business_query,
                'technical': is_technical_query
            }
        }
    
    def _retrieve_step(self, query: str, plan: str, current_knowledge: Dict) -> Dict[str, Any]:
        """Retrieve information from selected source based on plan"""
        
        # Select source based on plan and availability
        selected_source = self._select_source_from_plan(plan, current_knowledge)
        
        if not selected_source:
            return {
                'step_type': 'RETRIEVE',
                'source': 'none',
                'success': False,
                'content': 'No available sources for this query type',
                'reasoning': 'All relevant sources already queried or unavailable'
            }
        
        # Execute retrieval
        try:
            if selected_source == 'text_rag' and self.rag_system:
                results = self.rag_system.query(query, max_chunks=3)
                # DEBUG: Print actual response format
                logger.debug(f"RAG response keys: {list(results.keys())}")
                logger.debug(f"RAG success: {results.get('success')}")
                
                if results.get('success') and results.get('answer'):
                    content = results['answer'][:500]  # Use 'answer' not 'chunks'
                    sources_count = len(results.get('sources', []))
                    return {
                        'step_type': 'RETRIEVE',
                        'source': 'text_rag',
                        'success': True,
                        'content': content,
                        'reasoning': f'Retrieved answer from {sources_count} sources, confidence: {results.get("confidence", 0):.2f}'
                    }
                else:
                    return {
                        'step_type': 'RETRIEVE',
                        'source': 'text_rag',
                        'success': False,
                        'content': results.get('error', 'No answer generated'),
                        'reasoning': f'RAG query failed: {results.get("error", "Unknown error")}'
                    }
            
            elif selected_source == 'salesforce' and self.salesforce_connector:
                # Simulate Salesforce query (replace with actual implementation)
                return {
                    'step_type': 'RETRIEVE',
                    'source': 'salesforce',
                    'success': False,
                    'content': 'Salesforce query not implemented in test',
                    'reasoning': 'Placeholder for Salesforce integration'
                }
            
            elif selected_source == 'colpali_visual' and self.colpali_retriever:
                # Simulate ColPali query (replace with actual implementation)
                return {
                    'step_type': 'RETRIEVE',
                    'source': 'colpali_visual',
                    'success': False,
                    'content': 'ColPali query not implemented in test',
                    'reasoning': 'Placeholder for visual document processing'
                }
            
            # Source not available
            return {
                'step_type': 'RETRIEVE',
                'source': selected_source,
                'success': False,
                'content': f'{selected_source} not available',
                'reasoning': f'Source {selected_source} is not initialized'
            }
            
        except Exception as e:
            return {
                'step_type': 'RETRIEVE',
                'source': selected_source,
                'success': False,
                'content': f'Error: {str(e)}',
                'reasoning': f'Exception during {selected_source} retrieval'
            }
    
    def _select_source_from_plan(self, plan: str, current_knowledge: Dict) -> Optional[str]:
        """Intelligent source selection based on plan and current knowledge"""
        
        # Priority order based on plan
        if "VISUAL_QUERY" in plan and "colpali_visual" not in current_knowledge:
            if "colpali_visual" in self.available_sources:
                return "colpali_visual"
        
        if "BUSINESS_QUERY" in plan and "salesforce" not in current_knowledge:
            if "salesforce" in self.available_sources:
                return "salesforce"
        
        if "TEXT_QUERY" in plan and "text_rag" not in current_knowledge:
            if "text_rag" in self.available_sources:
                return "text_rag"
        
        # Fallback: try any unused available source
        for source in self.available_sources:
            if source not in current_knowledge:
                return source
        
        return None  # All sources exhausted
    
    def _rethink_step(self, query: str, knowledge: Dict, chain: List) -> Dict[str, Any]:
        """Assess knowledge completeness and determine next action"""
        
        successful_retrievals = sum(1 for step in chain 
                                  if step.get('step_type') == 'RETRIEVE' and step.get('success'))
        
        sources_attempted = len([step for step in chain if step.get('step_type') == 'RETRIEVE'])
        
        # Calculate confidence based on successful retrievals and coverage
        base_confidence = min(0.9, successful_retrievals * 0.3)
        
        # Boost confidence if we have good coverage
        if successful_retrievals >= 1 and sources_attempted >= len(self.available_sources):
            base_confidence = max(base_confidence, 0.8)
        
        # Assess if we should continue
        should_stop = False
        stop_reason = ""
        
        if successful_retrievals >= 2:
            should_stop = True
            stop_reason = "Multiple successful retrievals completed"
        elif successful_retrievals >= 1 and sources_attempted >= len(self.available_sources):
            should_stop = True
            stop_reason = "All available sources attempted"
        elif sources_attempted >= len(self.available_sources) and successful_retrievals == 0:
            should_stop = True
            stop_reason = "No successful retrievals from any source"
        
        assessment = f"Success: {successful_retrievals}/{sources_attempted}, Knowledge: {len(knowledge)} sources"
        
        return {
            'step_type': 'RETHINK',
            'assessment': assessment,
            'confidence': base_confidence,
            'should_stop': should_stop,
            'stop_reason': stop_reason,
            'successful_retrievals': successful_retrievals,
            'sources_attempted': sources_attempted
        }
    
    def _generate_step(self, query: str, knowledge: Dict, chain: List) -> Dict[str, Any]:
        """Generate final answer from accumulated knowledge"""
        
        if not knowledge:
            return {
                'step_type': 'GENERATE',
                'answer': f'I could not find relevant information to answer "{query}". Please ensure documents are loaded into the system.',
                'confidence': 0.2,
                'reasoning': 'No knowledge retrieved from any source'
            }
        
        # Simple synthesis (replace with GPT-4V in production)
        answer_parts = []
        for source, content in knowledge.items():
            if content and content != 'No relevant information found':
                answer_parts.append(f"From {source}: {content[:200]}")
        
        if answer_parts:
            final_answer = f"Based on the available information:\n\n" + "\n\n".join(answer_parts)
            confidence = 0.7
        else:
            final_answer = f'The sources were queried but did not contain relevant information for "{query}".'
            confidence = 0.3
        
        return {
            'step_type': 'GENERATE',
            'answer': final_answer,
            'confidence': confidence,
            'reasoning': f'Synthesized from {len(knowledge)} sources',
            'sources_used': list(knowledge.keys())
        }

class BaselineOrchestrator:
    """
    Baseline single-turn orchestrator for comparison
    """
    
    def __init__(self, rag_system: Optional[RAGSystem] = None,
                 colpali_retriever: Optional[ColPaliRetriever] = None,
                 salesforce_connector: Optional[SalesforceConnector] = None,
                 reranker: Optional[CrossEncoderReRanker] = None):
        
        self.rag_system = rag_system
        self.colpali_retriever = colpali_retriever
        self.salesforce_connector = salesforce_connector
        self.reranker = reranker
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """Simple single-turn query"""
        start_time = time.time()
        
        # Just use text RAG for baseline
        if self.rag_system:
            try:
                results = self.rag_system.query(user_query, max_chunks=3)
                if results.get('success') and results.get('answer'):
                    answer = f"Based on the documents: {results['answer'][:300]}"
                    confidence = results.get('confidence', 0.6)
                    success = True
                else:
                    answer = f"No relevant information found: {results.get('error', 'Unknown error')}"
                    confidence = 0.2
                    success = False
            except Exception as e:
                answer = f"Error processing query: {str(e)}"
                confidence = 0.1
                success = False
        else:
            answer = "Text RAG system not available."
            confidence = 0.1
            success = False
        
        execution_time = time.time() - start_time
        
        return {
            'final_answer': answer,
            'reasoning_chain': [{'step_type': 'SINGLE_RETRIEVAL', 'success': success}],
            'execution_time': execution_time,
            'sources_used': ['text_rag'] if success else [],
            'confidence_score': confidence,
            'total_steps': 1,
            'approach': 'baseline'
        }

def initialize_test_system():
    """Initialize minimal test system"""
    
    logger.info("Initializing test system...")
    
    # Initialize RAG system with minimal config (for token efficiency)
    rag_config = {
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'max_results': 5
    }
    
    try:
        rag_system = RAGSystem(rag_config)
        
        # Try to add documents if available
        documents_dir = os.path.join(os.path.dirname(__file__), 'data', 'documents')
        if os.path.exists(documents_dir):
            logger.info(f"Adding documents from {documents_dir}")
            try:
                # Get list of document files
                doc_files = []
                for file in os.listdir(documents_dir):
                    if file.endswith(('.txt', '.pdf', '.docx')):
                        doc_files.append(os.path.join(documents_dir, file))
                
                if doc_files:
                    rag_system.add_documents(doc_files[:3])  # Limit to 3 docs for efficiency
                    logger.info(f"Added {len(doc_files[:3])} documents successfully")
                else:
                    logger.warning("No suitable document files found")
            except Exception as e:
                logger.warning(f"Failed to add documents: {e}")
        else:
            logger.warning("Documents directory not found - responses will be limited")
    
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        rag_system = None
    
    # Skip ColPali and Salesforce for lightweight testing
    logger.info("Skipping ColPali and Salesforce for lightweight testing")
    
    return rag_system, None, None, None

def run_comparison_test(query: str, agentic_orch: AgenticOrchestrator, baseline_orch: BaselineOrchestrator, debug_mode: bool = False):
    """Run side-by-side comparison"""
    
    print(f"\n=== TESTING QUERY: {query} ===")
    
    # Test agentic approach
    print("\n--- AGENTIC APPROACH ---")
    agentic_result = agentic_orch.query(query)
    display_single_result(agentic_result, debug_mode)
    
    # Test baseline approach  
    print("\n--- BASELINE APPROACH ---")
    baseline_result = baseline_orch.query(query)
    display_single_result(baseline_result, debug_mode)
    
    # Enhanced comparison
    print("\n--- COMPARISON ---")
    if agentic_result['total_steps'] > baseline_result['total_steps']:
        print("+ Agentic: More thorough reasoning")
    if len(agentic_result['sources_used']) > len(baseline_result['sources_used']):
        print("+ Agentic: Better source utilization")
    if baseline_result['execution_time'] < agentic_result['execution_time']:
        print("+ Baseline: Faster execution")
    if agentic_result['confidence_score'] > baseline_result['confidence_score']:
        print("+ Agentic: Higher confidence")
    elif baseline_result['confidence_score'] > agentic_result['confidence_score']:
        print("+ Baseline: Higher confidence")
    
    # Show reasoning advantage
    agentic_sources = len(agentic_result['sources_used'])
    baseline_sources = len(baseline_result['sources_used'])
    if agentic_sources > 0 and baseline_sources > 0:
        print(f"Source efficiency: Agentic {agentic_sources} vs Baseline {baseline_sources}")
    
    return agentic_result, baseline_result

def interactive_mode(agentic_orch: AgenticOrchestrator, baseline_orch: BaselineOrchestrator):
    """Interactive testing mode with user input"""
    
    print("\n=== INTERACTIVE AGENTIC RAG TESTING ===")
    print("Enter your queries to test both agentic and baseline approaches")
    print("Commands:")
    print("  - Type any question to test both approaches")
    print("  - 'agentic <query>' - Test only agentic approach")
    print("  - 'baseline <query>' - Test only baseline approach")
    print("  - 'debug on/off' - Toggle debug mode for detailed reasoning")
    print("  - 'help' - Show this help")
    print("  - 'quit' or 'exit' - Exit interactive mode")
    print("-" * 60)
    
    debug_mode = False
    
    while True:
        try:
            user_input = input("\nEnter your query: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode...")
                break
                
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  - Type any question to test both approaches")
                print("  - 'agentic <query>' - Test only agentic approach")
                print("  - 'baseline <query>' - Test only baseline approach")
                print("  - 'debug on/off' - Toggle debug mode")
                print("  - 'quit' or 'exit' - Exit")
                continue
                
            elif user_input.lower().startswith('debug '):
                mode = user_input[6:].lower()
                if mode == 'on':
                    debug_mode = True
                    print("Debug mode ON - Will show detailed reasoning")
                elif mode == 'off':
                    debug_mode = False
                    print("Debug mode OFF - Standard output")
                else:
                    print("Use 'debug on' or 'debug off'")
                continue
                
            elif user_input.lower().startswith('agentic '):
                query = user_input[8:].strip()
                if query:
                    print(f"\n--- AGENTIC ONLY: {query} ---")
                    result = agentic_orch.query(query)
                    display_single_result(result, debug_mode)
                else:
                    print("Please provide a query after 'agentic'")
                continue
                
            elif user_input.lower().startswith('baseline '):
                query = user_input[9:].strip()
                if query:
                    print(f"\n--- BASELINE ONLY: {query} ---")
                    result = baseline_orch.query(query)
                    display_single_result(result, debug_mode)
                else:
                    print("Please provide a query after 'baseline'")
                continue
                
            else:
                # Regular comparison query
                run_comparison_test(user_input, agentic_orch, baseline_orch, debug_mode)
                
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

def display_single_result(result: Dict[str, Any], debug_mode: bool = False):
    """Display results from a single approach"""
    
    print(f"Time: {result['execution_time']:.2f}s")
    print(f"Steps: {result['total_steps']}")
    print(f"Sources: {result['sources_used']}")
    print(f"Confidence: {result['confidence_score']:.2f}")
    print(f"Approach: {result['approach']}")
    
    if debug_mode and 'reasoning_chain' in result:
        print("\nDETAILED REASONING CHAIN:")
        for i, step in enumerate(result['reasoning_chain'], 1):
            step_type = step.get('step_type', 'UNKNOWN')
            print(f"  Step {i} - {step_type}:")
            
            if step_type == 'THINK':
                print(f"    Reasoning: {step.get('reasoning', 'N/A')}")
                print(f"    Plan: {step.get('plan', 'N/A')}")
            elif step_type == 'RETRIEVE':
                print(f"    Source: {step.get('source', 'N/A')}")
                print(f"    Success: {step.get('success', False)}")
                print(f"    Reasoning: {step.get('reasoning', 'N/A')}")
                if debug_mode and step.get('success'):
                    content = step.get('content', '')
                    print(f"    Content: {content[:100]}..." if len(content) > 100 else f"    Content: {content}")
            elif step_type == 'RETHINK':
                print(f"    Assessment: {step.get('assessment', 'N/A')}")
                print(f"    Confidence: {step.get('confidence', 0):.2f}")
                print(f"    Should Stop: {step.get('should_stop', False)}")
                if step.get('stop_reason'):
                    print(f"    Stop Reason: {step.get('stop_reason')}")
            elif step_type == 'GENERATE':
                print(f"    Reasoning: {step.get('reasoning', 'N/A')}")
                print(f"    Confidence: {step.get('confidence', 0):.2f}")
    
    print(f"\nAnswer: {result['final_answer'][:300]}...")

def batch_test_mode(agentic_orch: AgenticOrchestrator, baseline_orch: BaselineOrchestrator):
    """Run predefined test queries for quick validation"""
    
    print("\n=== BATCH TEST MODE ===")
    print("Running predefined test queries...")
    
    test_queries = [
        "What is a transformer architecture?",
        "How does attention mechanism work?", 
        "What are the benefits of RAG systems?",
        "Explain the concept of self-attention",
        "What are the components of a transformer model?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}/{len(test_queries)}: {query} ---")
        run_comparison_test(query, agentic_orch, baseline_orch, debug_mode=False)
        print("-" * 60)
    
    print("\nBatch testing complete!")

def main():
    """Main test interface with multiple modes"""
    
    print("=== LIGHTWEIGHT AGENTIC RAG TEST ===")
    print("Token-efficient testing without emojis")
    
    # Initialize systems
    rag_system, colpali, salesforce, reranker = initialize_test_system()
    
    # Create orchestrators
    agentic_orch = AgenticOrchestrator(
        rag_system=rag_system,
        colpali_retriever=colpali,
        salesforce_connector=salesforce,
        reranker=reranker,
        max_reasoning_steps=6,
        confidence_threshold=0.8
    )
    
    baseline_orch = BaselineOrchestrator(
        rag_system=rag_system,
        colpali_retriever=colpali,
        salesforce_connector=salesforce,
        reranker=reranker
    )
    
    print("\nSelect testing mode:")
    print("1. Interactive mode (recommended) - Input your own queries")
    print("2. Batch test mode - Run predefined test queries")
    print("3. Single query mode - Test one query and exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == '1':
                interactive_mode(agentic_orch, baseline_orch)
                break
            elif choice == '2':
                batch_test_mode(agentic_orch, baseline_orch)
                break
            elif choice == '3':
                query = input("Enter your query: ").strip()
                if query:
                    run_comparison_test(query, agentic_orch, baseline_orch, debug_mode=True)
                break
            else:
                print("Please enter 1, 2, or 3")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()