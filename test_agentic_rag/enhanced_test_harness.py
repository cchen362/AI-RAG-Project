"""
Enhanced Test Harness - True vs Pseudo-Agentic Testing

Comprehensive testing framework for comparing true LLM-driven agentic reasoning
against pseudo-agentic fixed pipelines. Provides A/B testing, performance analysis,
and detailed reasoning comparison capabilities.

Key Features:
- Side-by-side true vs pseudo-agentic testing
- Performance metrics and cost analysis
- Reasoning quality assessment
- Query diversity testing for validation
- Comprehensive reporting and analysis
"""

import sys
import os
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import enhanced components
from enhanced_agentic_orchestrator import (
    EnhancedAgenticOrchestrator, ReasoningMode, ComparisonResult, UnifiedResponse
)

# Import existing production components
from src.rag_system import RAGSystem
from src.colpali_retriever import ColPaliRetriever
from src.salesforce_connector import SalesforceConnector
from src.cross_encoder_reranker import CrossEncoderReRanker

class EnhancedTestHarness:
    """
    Enhanced test harness for comprehensive agentic reasoning evaluation.
    Supports true vs pseudo-agentic A/B testing with detailed analysis.
    """
    
    def __init__(self, test_config: Optional[Dict] = None):
        """
        Initialize enhanced test harness.
        
        Args:
            test_config: Configuration for testing parameters
        """
        self.config = test_config or self._get_default_config()
        
        # Initialize components (will be set up lazily)
        self.rag_system = None
        self.colpali_retriever = None
        self.salesforce_connector = None
        self.reranker = None
        
        # Initialize enhanced orchestrator
        self.enhanced_orchestrator = None
        
        # Test results storage
        self.test_results = []
        self.comparison_results = []
        self.performance_metrics = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("Enhanced Test Harness initialized")
        print(f"Configuration: {json.dumps(self.config, indent=2)}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default test configuration."""
        return {
            "max_conversation_length": 10,
            "confidence_threshold": 0.75,
            "max_reasoning_steps": 10,
            "cost_threshold": 0.10,
            "enable_cost_monitoring": True,
            "enable_detailed_logging": True,
            "default_reasoning_mode": "true_agentic"
        }
    
    def setup_components(self, init_colpali: bool = False):
        """
        Set up all RAG components for testing.
        
        Args:
            init_colpali: Whether to initialize ColPali (slow, skip for quick testing)
        """
        print("\n[INIT] Setting up enhanced RAG components...")
        
        # Initialize Text RAG system with production-compatible configuration
        try:
            text_config = {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'embedding_model': 'openai',
                'generation_model': 'gpt-3.5-turbo',
                'max_retrieved_chunks': 5,
                'temperature': 0.1,
                'retrieval_mode': 'text',  # Ensure text mode is used
                'enable_synthesis': True   # Enable LLM synthesis
            }
            self.rag_system = RAGSystem(text_config)
            print(f"[DEBUG] RAG system created with config: {text_config}")
            print(f"[DEBUG] RAG system initialized: {getattr(self.rag_system, 'is_initialized', 'Unknown')}")
            print(f"[DEBUG] RAG system has vector_db: {hasattr(self.rag_system, 'vector_db')}")
            print(f"[DEBUG] RAG system has embedding_manager: {hasattr(self.rag_system, 'embedding_manager')}")
            
            # Process documents with enhanced debugging and better path resolution
            def find_documents_directory():
                """Find the correct documents directory with multiple fallback options."""
                potential_paths = [
                    # Primary path (from test_agentic_rag directory)
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'documents'),
                    # Alternative paths
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'documents'),
                    os.path.join(os.getcwd(), 'data', 'documents'),
                    # Direct path based on debug logs
                    r'C:\Users\cchen362\OneDrive\Desktop\AI-RAG-Project\data\documents',
                    # Additional fallbacks
                    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'documents'),
                    os.path.join(os.path.dirname(__file__), '..', 'data', 'documents')
                ]
                
                for path in potential_paths:
                    abs_path = os.path.abspath(path)
                    print(f"[DEBUG] Checking path: {abs_path}")
                    if os.path.exists(abs_path):
                        print(f"[DEBUG] [SUCCESS] Found valid path: {abs_path}")
                        return abs_path
                    else:
                        print(f"[DEBUG] [FAIL] Path not found: {abs_path}")
                
                return None
            
            data_dir = find_documents_directory()
            
            if data_dir:
                print(f"[INIT] Processing documents from: {data_dir}")
                try:
                    all_files = os.listdir(data_dir)
                    print(f"[DEBUG] All files in directory: {all_files}")
                    
                    doc_files = [os.path.join(data_dir, f) for f in all_files 
                               if f.endswith(('.txt', '.pdf', '.docx'))]
                    print(f"[DEBUG] Filtered document files: {doc_files}")
                    
                    if doc_files:
                        print(f"[INIT] Attempting to process {len(doc_files)} documents...")
                        try:
                            result = self.rag_system.add_documents(doc_files)
                            # Check for successful documents using correct key
                            docs_processed = len(result.get('successful', [])) if result.get('successful') else result.get('documents_processed', 0)
                            
                            # Production-compatible vector count detection
                            try:
                                print(f"[DEBUG] Checking vector count with production architecture...")
                                
                                # Check the production RAG system vector_db first
                                if hasattr(self.rag_system, 'vector_db') and hasattr(self.rag_system.vector_db, 'index'):
                                    vectors_count = self.rag_system.vector_db.index.ntotal
                                    print(f"[DEBUG] Used vector_db.index.ntotal: {vectors_count}")
                                elif hasattr(self.rag_system, 'vector_db'):
                                    print(f"[DEBUG] vector_db exists but no index attribute")
                                    print(f"[DEBUG] vector_db methods: {[m for m in dir(self.rag_system.vector_db) if not m.startswith('_')]}")
                                    vectors_count = "no_index_found"
                                # Fallback to embedding manager approaches
                                elif hasattr(self.rag_system.embedding_manager, 'get_vector_count'):
                                    vectors_count = self.rag_system.embedding_manager.get_vector_count()
                                    print(f"[DEBUG] Used embedding_manager.get_vector_count(): {vectors_count}")
                                elif hasattr(self.rag_system.embedding_manager, 'vector_store'):
                                    vector_store = self.rag_system.embedding_manager.vector_store
                                    print(f"[DEBUG] Vector store type: {type(vector_store)}")
                                    
                                    if hasattr(vector_store, 'ntotal'):
                                        vectors_count = vector_store.ntotal
                                        print(f"[DEBUG] Used ntotal: {vectors_count}")
                                    elif hasattr(vector_store, 'index') and hasattr(vector_store.index, 'ntotal'):
                                        vectors_count = vector_store.index.ntotal
                                        print(f"[DEBUG] Used index.ntotal: {vectors_count}")
                                    else:
                                        vectors_count = "no_count_method"
                                        print(f"[DEBUG] Vector store methods: {[m for m in dir(vector_store) if not m.startswith('_')]}")
                                else:
                                    vectors_count = "no_vector_store"
                                    print(f"[DEBUG] RAG system methods: {[m for m in dir(self.rag_system) if not m.startswith('_')]}")
                                    
                            except Exception as ve:
                                vectors_count = f"error: {str(ve)}"
                                print(f"[DEBUG] Vector count error: {ve}")
                            
                            print(f"[SUCCESS] Text RAG: Processed {docs_processed} documents, {vectors_count} vectors created")
                            
                            # Enhanced verification
                            if docs_processed == 0:
                                print(f"[CRITICAL] No documents were processed successfully!")
                                print(f"[DEBUG] Result details: {result}")
                            elif vectors_count == 0 or vectors_count == "unknown":
                                print(f"[WARNING] Vector count is {vectors_count} - document indexing may have failed")
                            else:
                                print(f"[SUCCESS] Document processing completed successfully!")
                                
                        except Exception as doc_error:
                            print(f"[ERROR] Document processing error: {doc_error}")
                            print(f"[DEBUG] Error type: {type(doc_error)}")
                            print(f"[DEBUG] RAG system state: {self.rag_system}")
                            import traceback
                            print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
                    else:
                        print(f"[WARNING] No valid document files found in {data_dir}")
                        print(f"[DEBUG] Supported extensions: .txt, .pdf, .docx")
                        print(f"[DEBUG] Available files: {all_files}")
                        
                except Exception as list_error:
                    print(f"[ERROR] Error listing directory contents: {list_error}")
                    print(f"[DEBUG] Directory: {data_dir}")
            else:
                print(f"[CRITICAL] Could not find documents directory in any expected location!")
                print(f"[DEBUG] Please ensure documents exist in one of these locations:")
                print(f"  - AI-RAG-Project/data/documents/")
                print(f"  - Current working directory: {os.getcwd()}")
                
                # Set documents_loaded flag for validation
                setattr(self, '_documents_loaded', False)
            print("[SUCCESS] Text RAG system initialized")
        except Exception as e:
            print(f"[ERROR] Text RAG initialization failed: {e}")
            self.rag_system = None
        
        # Initialize ColPali (conditional) with production-compatible config
        if init_colpali:
            try:
                colpali_config = {
                    'model_name': 'vidore/colqwen2-v1.0',
                    'device': 'auto',
                    'max_pages_per_doc': 50,
                    'cache_embeddings': True,
                    'cache_dir': 'cache/embeddings'
                }
                self.colpali_retriever = ColPaliRetriever(colpali_config)
                print("[SUCCESS] ColPali visual retriever initialized with production config")
            except Exception as e:
                print(f"[WARNING] ColPali initialization failed: {e}")
                print(f"[DEBUG] Trying without config parameter as fallback...")
                try:
                    self.colpali_retriever = ColPaliRetriever()
                    print("[SUCCESS] ColPali initialized with fallback method")
                except Exception as e2:
                    print(f"[ERROR] ColPali fallback also failed: {e2}")
                    self.colpali_retriever = None
        else:
            print("[SKIP] ColPali initialization skipped for faster testing")
            self.colpali_retriever = None
        
        # Initialize Salesforce connector
        try:
            self.salesforce_connector = SalesforceConnector()
            if self.salesforce_connector.test_connection():
                print("[SUCCESS] Salesforce connector initialized and tested")
            else:
                print("[WARNING] Salesforce connection test failed")
                self.salesforce_connector = None
        except Exception as e:
            print(f"[WARNING] Salesforce initialization failed: {e}")
            self.salesforce_connector = None
        
        # Initialize re-ranker
        try:
            self.reranker = CrossEncoderReRanker(
                model_name='BAAI/bge-reranker-base',
                relevance_threshold=0.3
            )
            if self.reranker.initialize():
                print("[SUCCESS] BGE re-ranker initialized")
            else:
                print("[WARNING] Re-ranker initialization failed")
                self.reranker = None
        except Exception as e:
            print(f"[WARNING] Re-ranker initialization failed: {e}")
            self.reranker = None
        
        # Initialize enhanced orchestrator
        try:
            self.enhanced_orchestrator = EnhancedAgenticOrchestrator(
                rag_system=self.rag_system,
                colpali_retriever=self.colpali_retriever,
                salesforce_connector=self.salesforce_connector,
                reranker=self.reranker,
                default_mode=ReasoningMode.TRUE_AGENTIC,
                enable_cost_monitoring=self.config.get("enable_cost_monitoring", True),
                enable_logging=self.config.get("enable_detailed_logging", True)
            )
            print("[SUCCESS] Enhanced Agentic Orchestrator initialized")
        except Exception as e:
            print(f"[ERROR] Enhanced orchestrator initialization failed: {e}")
            raise
        
        print(f"[SETUP] Setup complete! Available sources: {self._get_available_sources()}")
        
        # Run enhanced validation to verify document loading
        self._run_enhanced_preflight_validation()
    
    def _run_enhanced_preflight_validation(self):
        """Enhanced validation with detailed document loading verification."""
        print(f"\n[VALIDATION] RUNNING ENHANCED PRE-FLIGHT VALIDATION")
        print(f"{'='*60}")
        
        # Test Text RAG with simple queries
        if self.rag_system:
            try:
                test_queries = [
                    "transformer",
                    "attention mechanism", 
                    "neural network"
                ]
                
                for query in test_queries:
                    result = self.rag_system.query(query)
                    chunks = result.get('chunks', []) if isinstance(result, dict) else []
                    success = result.get('success', False) if isinstance(result, dict) else False
                    
                    print(f"[TEST] Query '{query}': {len(chunks)} chunks, success: {success}")
                    if not success and isinstance(result, dict):
                        print(f"[ERROR] Query failed: {result.get('error', 'Unknown error')}")
                    
                    if len(chunks) > 0:
                        print(f"[SUCCESS] Document loading appears to be working!")
                        break
                else:
                    print(f"[WARNING] No queries returned data - documents may not be properly loaded")
                    
            except Exception as e:
                print(f"[ERROR] Text RAG validation failed: {e}")
        
        # Test orchestration with simple query
        if hasattr(self, 'enhanced_orchestrator'):
            try:
                print(f"\n[TEST] Testing orchestration with simple query...")
                test_query = "What is attention mechanism?"
                findings = self.enhanced_orchestrator.llm_agent.orchestrate_retrieval(test_query)
                
                print(f"[RESULT] Orchestration test:")
                print(f"   Sources queried: {[s.value for s in findings.sources_queried]}")
                print(f"   Text chunks: {len(findings.text_chunks)}")
                print(f"   Insufficient data: {findings.insufficient_data}")
                print(f"   Confidence: {findings.confidence_assessment}")
                
                if not findings.insufficient_data:
                    print(f"[SUCCESS] Orchestration is working correctly!")
                else:
                    print(f"[WARNING] Orchestration reports insufficient data")
                    
            except Exception as e:
                print(f"[ERROR] Orchestration validation failed: {e}")
        
        print(f"{'='*60}")
    
    def _run_preflight_validation(self):
        """Run comprehensive pre-flight validation to ensure test environment is ready."""
        print(f"\n[VALIDATION] RUNNING PRE-FLIGHT VALIDATION")
        print(f"{'='*50}")
        
        validation_results = {
            "text_rag_ready": False,
            "salesforce_ready": False,
            "orchestrator_ready": False,
            "documents_loaded": False
        }
        
        # Check Text RAG system
        if self.rag_system:
            try:
                # Try a simple query to check if documents are loaded
                test_result = self.rag_system.query("test query", max_chunks=1)
                if test_result and test_result.get('chunks'):
                    validation_results["text_rag_ready"] = True
                    validation_results["documents_loaded"] = True
                    print("✅ Text RAG: Ready with documents loaded")
                else:
                    print("⚠️ Text RAG: System ready but no documents loaded")
            except Exception as e:
                print(f"[ERROR] Text RAG: Not ready - {e}")
        else:
            print("[ERROR] Text RAG: Not initialized")
        
        # Check Salesforce connector
        if self.salesforce_connector:
            try:
                # Test Salesforce connection
                if hasattr(self.salesforce_connector, 'search_knowledge_realtime'):
                    validation_results["salesforce_ready"] = True
                    print("[SUCCESS] Salesforce: Ready with correct interface")
                else:
                    print("[ERROR] Salesforce: Missing search_knowledge_realtime method")
            except Exception as e:
                print(f"[ERROR] Salesforce: Not ready - {e}")
        else:
            print("[WARNING] Salesforce: Not initialized")
        
        # Check enhanced orchestrator
        if self.enhanced_orchestrator:
            validation_results["orchestrator_ready"] = True
            print("[SUCCESS] Enhanced Orchestrator: Ready")
        else:
            print("[ERROR] Enhanced Orchestrator: Not initialized")
        
        # Summary
        ready_count = sum(validation_results.values())
        total_checks = len(validation_results)
        
        print(f"\n📊 PRE-FLIGHT VALIDATION SUMMARY:")
        print(f"   Ready components: {ready_count}/{total_checks}")
        print(f"   Text RAG documents: {'✅' if validation_results['documents_loaded'] else '❌'}")
        print(f"   Salesforce interface: {'✅' if validation_results['salesforce_ready'] else '❌'}")
        print(f"   Test environment: {'✅ READY' if ready_count >= 2 else '⚠️ PARTIAL' if ready_count >= 1 else '❌ NOT READY'}")
        
        if not validation_results["documents_loaded"]:
            print(f"\n🚨 WARNING: No documents loaded in Text RAG!")
            print(f"   This will cause the agentic system to always fallback to other sources.")
            print(f"   Consider adding documents to improve test accuracy.")
        
        print(f"{'='*50}")
        
        return validation_results
    
    def run_single_comparison_test(self, query: str) -> ComparisonResult:
        """
        Run A/B comparison test for a single query.
        
        Args:
            query: Test query to compare approaches
            
        Returns:
            ComparisonResult with detailed analysis
        """
        print(f"\n🧪 Running A/B comparison test")
        print(f"Query: {query}")
        print("-" * 60)
        
        if not self.enhanced_orchestrator:
            raise RuntimeError("Enhanced orchestrator not initialized. Call setup_components() first.")
        
        start_time = time.time()
        
        try:
            # Run A/B comparison
            comparison_result = self.enhanced_orchestrator.query(
                query, 
                mode=ReasoningMode.A_B_COMPARISON
            )
            
            total_time = time.time() - start_time
            
            # Store results
            self.comparison_results.append({
                "query": query,
                "result": comparison_result,
                "total_test_time": total_time,
                "timestamp": datetime.now().isoformat()
            })
            
            # Display summary
            self._display_comparison_summary(comparison_result, total_time)
            
            return comparison_result
            
        except Exception as e:
            print(f"❌ Comparison test failed: {e}")
            raise
    
    def run_single_mode_test(self, query: str, mode: ReasoningMode) -> UnifiedResponse:
        """
        Run test using a specific reasoning mode.
        
        Args:
            query: Test query
            mode: Reasoning mode to use
            
        Returns:
            UnifiedResponse with results
        """
        print(f"\n🎯 Running {mode.value} test")
        print(f"Query: {query}")
        print("-" * 60)
        
        if not self.enhanced_orchestrator:
            raise RuntimeError("Enhanced orchestrator not initialized. Call setup_components() first.")
        
        start_time = time.time()
        
        try:
            response = self.enhanced_orchestrator.query(query, mode=mode)
            total_time = time.time() - start_time
            
            # Display results
            self._display_single_mode_results(response, total_time)
            
            return response
            
        except Exception as e:
            print(f"❌ {mode.value} test failed: {e}")
            raise
    
    def run_query_diversity_test(self) -> Dict[str, Any]:
        """
        Run comprehensive test across diverse query types to validate
        dynamic reasoning behavior vs fixed pipeline approaches.
        
        Returns:
            Comprehensive test results with analysis
        """
        print("\n🌟 Running Query Diversity Test Suite")
        print("Testing both approaches across diverse query types...")
        print("=" * 70)
        
        # Diverse test queries
        test_queries = [
            {
                "query": "What is a transformer architecture in machine learning?",
                "category": "technical",
                "expected_sources": ["text_rag"],
                "description": "Basic technical query - should prefer Text RAG"
            },
            {
                "query": "What are the latest trends in artificial intelligence for business applications?",
                "category": "business",
                "expected_sources": ["salesforce", "text_rag"],
                "description": "Business query - should prefer Salesforce + Text RAG"
            },
            {
                "query": "How do transformers use attention mechanisms and what are the computational advantages over RNNs?",
                "category": "complex",
                "expected_sources": ["text_rag"],
                "description": "Complex technical query requiring deep analysis"
            },
            {
                "query": "Explain the mathematical formula for attention mechanism in transformers",
                "category": "mathematical",
                "expected_sources": ["text_rag"],
                "description": "Mathematical query requiring precise technical content"
            },
            {
                "query": "Compare transformer models with traditional RNN architectures",
                "category": "comparative",
                "expected_sources": ["text_rag"],
                "description": "Comparative analysis requiring synthesis"
            }
        ]
        
        diversity_results = {
            "test_queries": test_queries,
            "comparison_results": [],
            "analysis": {},
            "validation_summary": {}
        }
        
        # Run tests
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n📋 Test {i}/{len(test_queries)}: {test_case['category'].upper()}")
            print(f"Query: {test_case['query']}")
            print(f"Expected: {test_case['description']}")
            
            try:
                comparison_result = self.run_single_comparison_test(test_case['query'])
                
                # Add test case metadata
                test_result = {
                    "test_case": test_case,
                    "comparison_result": comparison_result,
                    "validation": self._validate_reasoning_behavior(comparison_result, test_case)
                }
                
                diversity_results["comparison_results"].append(test_result)
                
            except Exception as e:
                print(f"❌ Test {i} failed: {e}")
                diversity_results["comparison_results"].append({
                    "test_case": test_case,
                    "error": str(e)
                })
        
        # Analyze results
        diversity_results["analysis"] = self._analyze_diversity_results(diversity_results["comparison_results"])
        diversity_results["validation_summary"] = self._generate_validation_summary(diversity_results["analysis"])
        
        # Display final analysis
        self._display_diversity_analysis(diversity_results)
        
        return diversity_results
    
    def _validate_reasoning_behavior(self, comparison_result: ComparisonResult, test_case: Dict) -> Dict[str, Any]:
        """Validate that reasoning behavior matches expectations."""
        validation = {
            "query_category": test_case["category"],
            "expected_sources": test_case["expected_sources"],
            "true_agentic_validation": {},
            "pseudo_agentic_validation": {},
            "overall_validation": {}
        }
        
        # Validate true agentic behavior
        true_sources = [s.value for s in comparison_result.true_agentic_response.sources_queried]
        validation["true_agentic_validation"] = {
            "sources_used": true_sources,
            "intelligent_selection": len(true_sources) != len(self._get_available_sources()),  # Not all sources
            "dynamic_behavior": hasattr(comparison_result.true_agentic_response, 'reasoning_chain') and len(comparison_result.true_agentic_response.reasoning_chain) > 3,
            "source_relevance": any(expected in true_sources for expected in test_case["expected_sources"])
        }
        
        # Validate pseudo agentic behavior  
        pseudo_sources = [s.value for s in comparison_result.pseudo_agentic_response.sources_used]
        validation["pseudo_agentic_validation"] = {
            "sources_used": pseudo_sources,
            "fixed_behavior": True,  # Always uses fixed sequence
            "all_sources_queried": len(pseudo_sources) == len(self._get_available_sources())
        }
        
        # Overall validation
        validation["overall_validation"] = {
            "different_approaches": true_sources != pseudo_sources,
            "true_agentic_advantage": validation["true_agentic_validation"]["intelligent_selection"],
            "reasoning_quality_difference": "TRUE_AGENTIC recommended" in comparison_result.recommendation
        }
        
        return validation
    
    def _analyze_diversity_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze results across diverse query types."""
        successful_tests = [r for r in results if "comparison_result" in r]
        
        if not successful_tests:
            return {"error": "No successful tests to analyze"}
        
        analysis = {
            "total_tests": len(results),
            "successful_tests": len(successful_tests),
            "true_agentic_performance": {},
            "pseudo_agentic_performance": {},
            "reasoning_behavior_analysis": {},
            "recommendation_trends": {}
        }
        
        # Analyze true agentic performance
        true_wins = sum(1 for r in successful_tests if "TRUE_AGENTIC recommended" in r["comparison_result"].recommendation)
        intelligent_selections = sum(1 for r in successful_tests if r["validation"]["true_agentic_validation"]["intelligent_selection"])
        dynamic_behaviors = sum(1 for r in successful_tests if r["validation"]["true_agentic_validation"]["dynamic_behavior"])
        
        analysis["true_agentic_performance"] = {
            "recommendation_wins": true_wins,
            "intelligent_selections": intelligent_selections,
            "dynamic_behaviors": dynamic_behaviors,
            "success_rate": true_wins / len(successful_tests) if successful_tests else 0
        }
        
        # Analyze reasoning behavior differences
        different_approaches = sum(1 for r in successful_tests if r["validation"]["overall_validation"]["different_approaches"])
        
        analysis["reasoning_behavior_analysis"] = {
            "different_source_selections": different_approaches,
            "percentage_different": different_approaches / len(successful_tests) if successful_tests else 0,
            "true_agentic_advantages": sum(1 for r in successful_tests if r["validation"]["overall_validation"]["true_agentic_advantage"])
        }
        
        # Recommendation trends by category
        category_recommendations = {}
        for result in successful_tests:
            category = result["test_case"]["category"]
            recommendation = "true" if "TRUE_AGENTIC recommended" in result["comparison_result"].recommendation else "pseudo"
            
            if category not in category_recommendations:
                category_recommendations[category] = {"true": 0, "pseudo": 0}
            category_recommendations[category][recommendation] += 1
        
        analysis["recommendation_trends"] = category_recommendations
        
        return analysis
    
    def _generate_validation_summary(self, analysis: Dict) -> Dict[str, Any]:
        """Generate validation summary from analysis."""
        if "error" in analysis:
            return {"status": "failed", "reason": analysis["error"]}
        
        # Determine if true agentic reasoning is working properly
        true_success_rate = analysis["true_agentic_performance"]["success_rate"]
        different_behaviors = analysis["reasoning_behavior_analysis"]["percentage_different"]
        intelligent_selections = analysis["true_agentic_performance"]["intelligent_selections"]
        
        validation_status = "excellent" if (
            true_success_rate >= 0.6 and 
            different_behaviors >= 0.8 and 
            intelligent_selections >= analysis["true_agentic_performance"]["recommendation_wins"]
        ) else "good" if (
            true_success_rate >= 0.4 and 
            different_behaviors >= 0.6
        ) else "needs_improvement"
        
        return {
            "status": validation_status,
            "true_agentic_success_rate": true_success_rate,
            "behavioral_differences": different_behaviors,
            "intelligent_selection_rate": intelligent_selections / analysis["total_tests"] if analysis["total_tests"] > 0 else 0,
            "key_findings": self._generate_key_findings(analysis),
            "recommendations": self._generate_improvement_recommendations(analysis, validation_status)
        }
    
    def _generate_key_findings(self, analysis: Dict) -> List[str]:
        """Generate key findings from analysis."""
        findings = []
        
        if analysis["reasoning_behavior_analysis"]["percentage_different"] >= 0.8:
            findings.append("✅ True agentic shows distinct reasoning behavior vs fixed pipeline")
        else:
            findings.append("⚠️ Limited behavioral differences detected between approaches")
        
        if analysis["true_agentic_performance"]["success_rate"] >= 0.6:
            findings.append(f"✅ True agentic recommended in {analysis['true_agentic_performance']['success_rate']:.1%} of tests")
        else:
            findings.append(f"⚠️ True agentic only recommended in {analysis['true_agentic_performance']['success_rate']:.1%} of tests")
        
        if analysis["true_agentic_performance"]["intelligent_selections"] > 0:
            findings.append(f"✅ Intelligent source selection demonstrated in {analysis['true_agentic_performance']['intelligent_selections']} cases")
        else:
            findings.append("❌ No evidence of intelligent source selection")
        
        return findings
    
    def _generate_improvement_recommendations(self, analysis: Dict, status: str) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []
        
        if status == "needs_improvement":
            recommendations.append("🔧 Review LLM prompts for clearer decision-making instructions")
            recommendations.append("🔧 Adjust confidence thresholds for earlier stopping")
            recommendations.append("🔧 Enhance source capability descriptions for better selection")
        elif status == "good":
            recommendations.append("📈 Fine-tune stopping criteria for better efficiency")
            recommendations.append("📈 Expand test scenarios to validate edge cases")
        else:  # excellent
            recommendations.append("🚀 Ready for production integration")
            recommendations.append("🚀 Consider expanding to additional source types")
        
        return recommendations
    
    def _display_comparison_summary(self, result: ComparisonResult, total_time: float):
        """Display comparison test summary."""
        print(f"\n📊 COMPARISON RESULTS (Total time: {total_time:.2f}s)")
        print("=" * 60)
        
        # Performance comparison
        perf = result.performance_comparison
        print(f"⏱️  Execution Time:")
        print(f"   • True Agentic: {perf['execution_time_comparison']['true_agentic']:.2f}s")
        print(f"   • Pseudo Agentic: {perf['execution_time_comparison']['pseudo_agentic']:.2f}s")
        print(f"   • Winner: {perf['execution_time_comparison']['winner']} agentic")
        
        print(f"\n🎯 Source Usage:")
        print(f"   • True Agentic: {perf['source_utilization']['true_agentic_sources']} sources")
        print(f"   • Pseudo Agentic: {perf['source_utilization']['pseudo_agentic_sources']} sources")
        print(f"   • Intelligent Selection: {perf['source_utilization']['intelligent_selection']}")
        
        print(f"\n🧠 Reasoning Complexity:")
        print(f"   • True Agentic: {perf['reasoning_complexity']['true_agentic_steps']} steps")
        print(f"   • Pseudo Agentic: {perf['reasoning_complexity']['pseudo_agentic_steps']} steps")
        print(f"   • Dynamic Decisions: {perf['reasoning_complexity']['dynamic_decisions']}")
        
        print(f"\n💰 Cost Analysis:")
        cost = result.cost_analysis
        print(f"   • True Agentic: ${cost['true_agentic_cost']:.4f}")
        print(f"   • Pseudo Agentic: ${cost['pseudo_agentic_cost']:.4f}")
        print(f"   • Efficiency: {cost['cost_efficiency']}")
        
        print(f"\n🏆 RECOMMENDATION: {result.recommendation}")
    
    def _display_single_mode_results(self, response: UnifiedResponse, total_time: float):
        """Display single mode test results."""
        print(f"\n📊 {response.reasoning_approach} RESULTS (Total time: {total_time:.2f}s)")
        print("=" * 60)
        print(f"⏱️  Execution Time: {response.execution_time:.2f}s")
        print(f"🎯 Sources Used: {response.sources_used}")
        print(f"🧠 Reasoning Steps: {response.reasoning_transparency.get('total_reasoning_steps', 0)}")
        print(f"💡 Confidence: {response.confidence_score:.2f}")
        print(f"💰 Cost: ${response.cost_breakdown.get('total', 0):.4f}")
        print(f"\n📝 Answer: {response.final_answer[:200]}...")
    
    def _display_diversity_analysis(self, results: Dict):
        """Display comprehensive diversity test analysis."""
        print(f"\n🌟 QUERY DIVERSITY TEST ANALYSIS")
        print("=" * 70)
        
        analysis = results["analysis"]
        validation = results["validation_summary"]
        
        if "error" in analysis:
            print(f"❌ Analysis failed: {analysis['error']}")
            return
        
        print(f"📊 Test Summary:")
        print(f"   • Total Tests: {analysis['total_tests']}")
        print(f"   • Successful: {analysis['successful_tests']}")
        print(f"   • Success Rate: {analysis['successful_tests']/analysis['total_tests']:.1%}")
        
        print(f"\n🧠 True Agentic Performance:")
        true_perf = analysis["true_agentic_performance"]
        print(f"   • Recommendation Wins: {true_perf['recommendation_wins']}/{analysis['successful_tests']}")
        print(f"   • Intelligent Selections: {true_perf['intelligent_selections']}")
        print(f"   • Dynamic Behaviors: {true_perf['dynamic_behaviors']}")
        print(f"   • Success Rate: {true_perf['success_rate']:.1%}")
        
        print(f"\n📈 Behavioral Analysis:")
        behavior = analysis["reasoning_behavior_analysis"]
        print(f"   • Different Source Selections: {behavior['different_source_selections']}/{analysis['successful_tests']}")
        print(f"   • Behavioral Difference Rate: {behavior['percentage_different']:.1%}")
        print(f"   • True Agentic Advantages: {behavior['true_agentic_advantages']}")
        
        print(f"\n🎯 Validation Status: {validation['status'].upper()}")
        
        print(f"\n[FINDINGS] Key Findings:")
        for finding in validation["key_findings"]:
            print(f"   {finding}")
        
        print(f"\n💡 Recommendations:")
        for rec in validation["recommendations"]:
            print(f"   {rec}")
    
    def _get_available_sources(self) -> List[str]:
        """Get list of available sources."""
        sources = []
        if self.rag_system:
            sources.append("text_rag")
        if self.colpali_retriever:
            sources.append("colpali_visual")
        if self.salesforce_connector:
            sources.append("salesforce")
        return sources
    
    def save_test_results(self, filename: Optional[str] = None):
        """Save all test results to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_test_results_{timestamp}.json"
        
        results = {
            "config": self.config,
            "test_results": self.test_results,
            "comparison_results": [
                {
                    "query": r["query"],
                    "result": asdict(r["result"]) if hasattr(r["result"], '__dict__') else str(r["result"]),
                    "total_test_time": r["total_test_time"],
                    "timestamp": r["timestamp"]
                }
                for r in self.comparison_results
            ],
            "performance_metrics": self.performance_metrics,
            "available_sources": self._get_available_sources(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📁 Test results saved to: {filename}")
        return filename

# Example usage and testing utilities
if __name__ == "__main__":
    print("🧪 Enhanced Test Harness for True vs Pseudo-Agentic Comparison")
    print("=" * 60)
    
    # Initialize test harness
    harness = EnhancedTestHarness()
    
    print("Available test methods:")
    print("  • harness.setup_components() - Initialize all systems")
    print("  • harness.run_single_comparison_test(query) - A/B test single query")
    print("  • harness.run_query_diversity_test() - Comprehensive validation")
    print("  • harness.save_test_results() - Save results to file")
    print("\nReady for enhanced agentic testing!")