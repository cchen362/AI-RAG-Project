"""
Graph-R1 Agentic RAG Demonstration App

Showcases true agentic behavior with:
1. Multi-source intelligence (Text + ColPali + Salesforce)
2. Interpretable reasoning chains with audit trails
3. Graph traversal visualization
4. Side-by-side baseline vs Graph-R1 comparison
5. Professional output without source annotations

Key Features:
- Dynamic graph construction from multiple sources
- LLM-driven path planning and traversal
- Budgeted retrieval with early stopping
- Complete audit trail of decisions
- Interactive reasoning chain visualization
"""

import os
import sys
import streamlit as st
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))

# Import Graph-R1 components
try:
    from src.hypergraph_constructor import create_hypergraph_constructor, ValidationSuite
    from src.graph_traversal_engine import (
        create_graph_traversal_engine, TraversalBudget, QueryType
    )
    from src.interpretable_reasoning_chain import ReasoningChain, PathVisualizer
    from src.rag_system import RAGSystem, create_rag_system
    from src.colpali_retriever import ColPaliRetriever
    from src.salesforce_connector import SalesforceConnector
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.info("Please ensure all Graph-R1 components are available in src/")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Graph-R1 Agentic RAG Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .comparison-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    
    .agentic-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .baseline-highlight {
        background: #f8f9fa;
        border: 2px solid #6c757d;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .reasoning-step {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .audit-trail {
        background: #f3e5f5;
        border: 1px solid #9c27b0;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.9em;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    
    .source-indicator {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8em;
        font-weight: bold;
        margin: 0.25rem;
    }
    
    .source-text { background: #e3f2fd; color: #1976d2; }
    .source-visual { background: #e8f5e8; color: #388e3c; }
    .source-salesforce { background: #fff3e0; color: #f57c00; }
</style>
""", unsafe_allow_html=True)

class GraphR1Demo:
    """Main demo application class."""
    
    def __init__(self):
        self.config = {
            'chunk_size': 800,
            'chunk_overlap': 150,
            'semantic_similarity_threshold': 0.7,
            'cross_modal_similarity_threshold': 0.6,
            'visual_model': 'vidore/colqwen2-v1.0'
        }
        
        # Initialize session state
        if 'hypergraph_built' not in st.session_state:
            st.session_state.hypergraph_built = False
        if 'baseline_system' not in st.session_state:
            st.session_state.baseline_system = None
        if 'graph_r1_system' not in st.session_state:
            st.session_state.graph_r1_system = None
        if 'demo_results' not in st.session_state:
            st.session_state.demo_results = {}
    
    def render_header(self):
        """Render main header with agentic emphasis."""
        st.markdown("""
        <div class="main-header">
            <h1>🧠 Graph-R1 Agentic RAG System</h1>
            <h3>Interpretable Multi-Source Intelligence with Graph Traversal</h3>
            <p>Experience true agentic behavior with complete audit trails and reasoning transparency</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with system configuration."""
        st.sidebar.title("🎛️ System Configuration")
        
        # Document sources section
        st.sidebar.subheader("📚 Document Sources")
        
        # Text documents
        st.sidebar.write("**Text Documents:**")
        text_docs = st.sidebar.file_uploader(
            "Upload text documents", 
            type=['txt', 'pdf', 'docx', 'md'],
            accept_multiple_files=True,
            key="text_docs"
        )
        
        # Visual documents
        st.sidebar.write("**Visual Documents (PDFs):**")
        visual_docs = st.sidebar.file_uploader(
            "Upload PDF documents for visual analysis",
            type=['pdf'],
            accept_multiple_files=True,
            key="visual_docs"
        )
        
        # Salesforce queries
        st.sidebar.write("**Salesforce Knowledge Base:**")
        sf_queries = st.sidebar.text_area(
            "Query terms (one per line)",
            value="booking\ntravel\ncustomer service\npolicy\nprocedure\nsupport\ncancellation\nmodification\ncheck-in\nrefund",
            key="sf_queries"
        )
        
        # Traversal settings
        st.sidebar.subheader("🔧 Traversal Settings")
        
        max_hops = st.sidebar.slider(
            "Max Graph Hops", 
            min_value=1, max_value=5, value=3,
            help="Maximum depth of graph traversal"
        )
        
        max_nodes = st.sidebar.slider(
            "Max Nodes to Visit",
            min_value=5, max_value=50, value=20,
            help="Budget limit for node exploration"
        )
        
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.1, max_value=0.9, value=0.6, step=0.1,
            help="Minimum confidence for path continuation"
        )
        
        # Build hypergraph button
        if st.sidebar.button("🏗️ Build Hypergraph", type="primary"):
            self.build_hypergraph(text_docs, visual_docs, sf_queries.split('\n'))
        
        # System status
        st.sidebar.subheader("📊 System Status")
        
        # Check API credentials
        openai_key = os.getenv('OPENAI_API_KEY')
        sf_username = os.getenv('SALESFORCE_USERNAME')
        sf_password = os.getenv('SALESFORCE_PASSWORD')
        sf_token = os.getenv('SALESFORCE_SECURITY_TOKEN')
        
        st.sidebar.write("**🔑 Credentials Status:**")
        if openai_key:
            st.sidebar.success("✅ OpenAI API Key loaded")
        else:
            st.sidebar.error("❌ OpenAI API Key missing")
        
        if sf_username and sf_password and sf_token:
            st.sidebar.success("✅ Salesforce credentials loaded")
        else:
            st.sidebar.warning("⚠️ Salesforce credentials incomplete")
        
        st.sidebar.write("**🏗️ System Status:**")
        if st.session_state.hypergraph_built:
            st.sidebar.success("✅ Hypergraph constructed")
            st.sidebar.success("✅ Graph-R1 system ready")
        else:
            st.sidebar.warning("⚠️ Hypergraph not built")
        
        return {
            'max_hops': max_hops,
            'max_nodes': max_nodes,
            'confidence_threshold': confidence_threshold,
            'text_docs': text_docs,
            'visual_docs': visual_docs,
            'sf_queries': sf_queries.split('\n') if sf_queries else []
        }
    
    def build_hypergraph(self, text_docs, visual_docs, sf_queries):
        """Build the unified hypergraph from multiple sources."""
        with st.spinner("🏗️ Building unified hypergraph..."):
            try:
                # Save uploaded files temporarily
                source_paths = {
                    'text_documents': [],
                    'visual_documents': [],
                    'salesforce_queries': sf_queries
                }
                
                # Save text documents
                if text_docs:
                    os.makedirs("temp/text_docs", exist_ok=True)
                    for doc in text_docs:
                        file_path = f"temp/text_docs/{doc.name}"
                        with open(file_path, "wb") as f:
                            f.write(doc.getbuffer())
                        source_paths['text_documents'].append(file_path)
                
                # Save visual documents
                if visual_docs:
                    os.makedirs("temp/visual_docs", exist_ok=True)
                    for doc in visual_docs:
                        file_path = f"temp/visual_docs/{doc.name}"
                        with open(file_path, "wb") as f:
                            f.write(doc.getbuffer())
                        source_paths['visual_documents'].append(file_path)
                
                # Create hypergraph constructor
                st.session_state.hypergraph_builder = create_hypergraph_constructor(self.config)
                
                # Build hypergraph
                build_results = st.session_state.hypergraph_builder.build_hypergraph(source_paths)
                
                # Create Graph-R1 system components
                (st.session_state.path_planner, 
                 st.session_state.graph_traverser,
                 st.session_state.confidence_manager,
                 st.session_state.reasoning_logger) = create_graph_traversal_engine(
                    st.session_state.hypergraph_builder, self.config
                )
                
                # Create baseline system for comparison
                baseline_config = {
                    'chunk_size': 800,
                    'chunk_overlap': 150,
                    'embedding_model': 'openai',
                    'max_retrieved_chunks': 5
                }
                st.session_state.baseline_system = create_rag_system(baseline_config)
                
                # Add documents to baseline (text only)
                if source_paths['text_documents']:
                    st.session_state.baseline_system.add_documents(source_paths['text_documents'])
                
                st.session_state.hypergraph_built = True
                st.session_state.build_results = build_results
                
                st.success("✅ Hypergraph built successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Failed to build hypergraph: {e}")
                logger.error(f"Hypergraph build error: {e}")
    
    def render_query_interface(self, settings):
        """Render the main query interface."""
        st.subheader("💭 Ask Your Question")
        
        # Example queries
        example_queries = [
            "What are the key components of transformer architecture?",
            "How does attention mechanism work in neural networks?",
            "Compare different RAG approaches for document retrieval",
            "What are the latest AI developments in our knowledge base?",
            "Explain the process of fine-tuning language models"
        ]
        
        # Query input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Enter your question:",
                placeholder="Ask anything about your documents and knowledge base...",
                key="main_query"
            )
        
        with col2:
            if st.selectbox("Example queries:", [""] + example_queries, key="example_selector"):
                query = st.session_state.example_selector
                st.session_state.main_query = query
        
        # Query execution buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            run_comparison = st.button("🚀 Run Comparison", type="primary", disabled=not st.session_state.hypergraph_built)
        
        with col2:
            run_agentic_only = st.button("🧠 Graph-R1 Only", disabled=not st.session_state.hypergraph_built)
        
        with col3:
            run_baseline_only = st.button("📄 Baseline Only", disabled=not st.session_state.hypergraph_built)
        
        if query and (run_comparison or run_agentic_only or run_baseline_only):
            # Create traversal budget
            budget = TraversalBudget(
                max_hops=settings['max_hops'],
                max_nodes_visited=settings['max_nodes'],
                max_tokens_used=2000,
                max_time_seconds=60.0,
                min_confidence_threshold=settings['confidence_threshold']
            )
            
            if run_comparison:
                self.run_comparison_demo(query, budget)
            elif run_agentic_only:
                self.run_agentic_demo(query, budget)
            elif run_baseline_only:
                self.run_baseline_demo(query)
    
    def run_comparison_demo(self, query: str, budget: TraversalBudget):
        """Run side-by-side comparison between baseline and Graph-R1."""
        st.subheader("⚔️ Baseline vs Graph-R1 Comparison")
        
        col1, col2 = st.columns(2)
        
        # Baseline response
        with col1:
            st.markdown('<div class="baseline-highlight">', unsafe_allow_html=True)
            st.write("### 📄 Baseline RAG Response")
            
            with st.spinner("Running baseline RAG..."):
                baseline_start = time.time()
                baseline_result = st.session_state.baseline_system.query(query)
                baseline_time = time.time() - baseline_start
            
            if baseline_result.get('success'):
                st.write(baseline_result['answer'])
                
                # Baseline metrics
                st.write("**📊 Baseline Metrics:**")
                st.write(f"- Response time: {baseline_time:.2f}s")
                st.write(f"- Sources used: {baseline_result.get('chunks_used', 0)}")
                st.write(f"- Confidence: {baseline_result.get('confidence', 0):.3f}")
                st.write("- Sources: Text documents only")
                st.write("- Reasoning: Traditional similarity search")
            else:
                st.error(f"❌ Baseline failed: {baseline_result.get('error')}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Graph-R1 response
        with col2:
            st.markdown('<div class="agentic-highlight">', unsafe_allow_html=True)
            st.write("### 🧠 Graph-R1 Agentic Response")
            
            with st.spinner("Running Graph-R1 agentic traversal..."):
                agentic_result = self.execute_agentic_query(query, budget)
            
            if agentic_result.get('success'):
                st.write(agentic_result['answer'])
                
                # Agentic metrics
                st.write("**📊 Graph-R1 Metrics:**")
                st.write(f"- Response time: {agentic_result['response_time']:.2f}s")
                st.write(f"- Nodes explored: {agentic_result['nodes_explored']}")
                st.write(f"- Graph hops: {agentic_result['hops_used']}")
                st.write(f"- Sources: {', '.join(agentic_result['sources_used'])}")
                st.write(f"- Paths evaluated: {agentic_result['paths_evaluated']}")
                st.write("- Reasoning: LLM-guided graph traversal")
            else:
                st.error(f"❌ Graph-R1 failed: {agentic_result.get('error')}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed comparison
        if baseline_result.get('success') and agentic_result.get('success'):
            self.render_detailed_comparison(baseline_result, agentic_result, baseline_time)
    
    def run_agentic_demo(self, query: str, budget: TraversalBudget):
        """Run Graph-R1 agentic demo with full reasoning visualization."""
        st.subheader("🧠 Graph-R1 Agentic Analysis")
        
        with st.spinner("🔍 Analyzing query and planning traversal..."):
            agentic_result = self.execute_agentic_query(query, budget, detailed=True)
        
        if agentic_result.get('success'):
            # Main response
            st.markdown('<div class="agentic-highlight">', unsafe_allow_html=True)
            st.write("### 💡 Graph-R1 Response")
            st.write(agentic_result['answer'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Reasoning visualization
            self.render_reasoning_visualization(agentic_result)
            
            # Audit trail
            self.render_audit_trail(agentic_result.get('audit_trail', []))
            
        else:
            st.error(f"❌ Graph-R1 analysis failed: {agentic_result.get('error')}")
    
    def run_baseline_demo(self, query: str):
        """Run baseline demo only."""
        st.subheader("📄 Baseline RAG Analysis")
        
        with st.spinner("Running baseline RAG query..."):
            baseline_result = st.session_state.baseline_system.query(query)
        
        if baseline_result.get('success'):
            st.write("### Response")
            st.write(baseline_result['answer'])
            
            # Show sources
            st.write("### Sources")
            for i, source in enumerate(baseline_result.get('sources', []), 1):
                st.write(f"**{i}. {source['filename']}** (relevance: {source['relevance_score']:.3f})")
                st.write(f"   {source['chunk_text']}")
        else:
            st.error(f"❌ Baseline query failed: {baseline_result.get('error')}")
    
    def execute_agentic_query(self, query: str, budget: TraversalBudget, detailed: bool = False) -> Dict[str, Any]:
        """Execute Graph-R1 agentic query with full reasoning."""
        try:
            start_time = time.time()
            
            # Step 1: Query analysis
            analysis = st.session_state.path_planner.analyze_query(query)
            st.session_state.reasoning_logger.log_query_analysis(query, analysis)
            
            # Step 2: Plan entry points
            entry_points = st.session_state.path_planner.plan_entry_points(
                query, st.session_state.hypergraph_builder, analysis
            )
            st.session_state.reasoning_logger.log_entry_point_selection(
                entry_points, f"Selected {len(entry_points)} optimal entry points based on query similarity"
            )
            
            # Step 3: Execute graph traversal
            traversal_mode = analysis['strategy']['mode']
            completed_paths = st.session_state.graph_traverser.traverse_graph(
                entry_points, budget, traversal_mode, query
            )
            
            # Step 4: Evaluate paths and select best
            best_path = None
            if completed_paths:
                path_evaluations = []
                for path in completed_paths:
                    evaluation = st.session_state.confidence_manager.evaluate_path_quality(path)
                    path_evaluations.append((path, evaluation))
                
                # Sort by overall quality
                path_evaluations.sort(key=lambda x: x[1]['overall_quality'], reverse=True)
                best_path = path_evaluations[0][0]
            
            # Step 5: Generate final answer
            if best_path:
                answer = self.synthesize_final_answer(query, best_path, analysis)
                sources_used = list(best_path.source_types_visited)
            else:
                answer = "I couldn't find sufficient information to answer your question through graph traversal."
                sources_used = []
            
            response_time = time.time() - start_time
            
            # Log stopping decision
            st.session_state.reasoning_logger.log_stopping_decision(
                completed_paths, 
                f"Analysis complete with {len(completed_paths)} paths evaluated",
                budget
            )
            
            result = {
                'success': True,
                'answer': answer,
                'response_time': response_time,
                'nodes_explored': budget.nodes_visited,
                'hops_used': budget.hops_used,
                'sources_used': sources_used,
                'paths_evaluated': len(completed_paths),
                'best_path': best_path,
                'query_analysis': analysis,
                'audit_trail': st.session_state.reasoning_logger.get_audit_trail(),
                'reasoning_summary': st.session_state.reasoning_logger.generate_summary_report()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Agentic query execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def synthesize_final_answer(self, query: str, best_path, analysis) -> str:
        """Synthesize final answer from best path using LLM with relevance checking."""
        try:
            # Extract relevant content from path nodes
            relevant_content = []
            query_lower = query.lower()
            query_keywords = set(query_lower.split())
            
            for node in best_path.nodes:
                # Calculate content relevance to query
                content_lower = node.node.content.lower()
                content_words = set(content_lower.split())
                keyword_overlap = len(query_keywords.intersection(content_words))
                relevance_score = keyword_overlap / max(len(query_keywords), 1)
                
                # Only include content with decent relevance or high confidence
                if relevance_score > 0.1 or node.confidence > 0.7:
                    relevant_content.append({
                        'source': node.node.source_type,
                        'content': node.node.content[:500],
                        'confidence': node.confidence,
                        'relevance': relevance_score,
                        'filename': node.node.source_metadata.get('filename', 'Unknown')
                    })
            
            # Sort by relevance and confidence
            relevant_content.sort(key=lambda x: (x['relevance'] + x['confidence']) / 2, reverse=True)
            
            # Check if we have sufficient relevant content
            if not relevant_content or max(c['relevance'] for c in relevant_content) < 0.05:
                return f"I don't have enough relevant information in my knowledge base to answer the question: '{query}'. The available documents don't seem to contain content related to this topic."
            
            # Use OpenAI to synthesize proper answer
            if hasattr(st.session_state, 'baseline_system') and hasattr(st.session_state.baseline_system, 'llm_client'):
                return self._llm_synthesize_answer(query, relevant_content, analysis)
            else:
                # Fallback to improved template if LLM not available
                return self._template_synthesize_answer(query, relevant_content, best_path)
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def _llm_synthesize_answer(self, query: str, relevant_content: List[Dict], analysis: Dict) -> str:
        """Use LLM to synthesize coherent answer from relevant content."""
        try:
            # Prepare context from relevant content
            context_parts = []
            for i, content in enumerate(relevant_content[:5], 1):  # Limit to top 5 most relevant
                source_info = f"Source {i} ({content['source']} - {content['filename']})"
                context_parts.append(f"{source_info}:\n{content['content']}\n")
            
            context_text = "\n".join(context_parts)
            
            system_prompt = """You are an expert assistant that provides accurate, helpful answers based strictly on the provided context. 

IMPORTANT RULES:
1. Only use information from the provided context
2. If the context doesn't contain relevant information, say "I don't have enough information to answer this question"
3. Be concise and direct
4. Don't make assumptions or add information not in the context
5. Cite which sources you're using (e.g., "According to Source 1...")"""

            user_prompt = f"""Question: {query}

Context from knowledge base:
{context_text}

Please provide a helpful answer based only on the context above."""

            response = st.session_state.baseline_system.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return self._template_synthesize_answer(query, relevant_content, None)
    
    def _template_synthesize_answer(self, query: str, relevant_content: List[Dict], best_path) -> str:
        """Improved template-based answer synthesis as fallback."""
        if not relevant_content:
            return f"I don't have relevant information to answer the question: '{query}'"
        
        # Build answer focusing on most relevant content
        answer_parts = []
        
        # Use the most relevant content
        top_content = relevant_content[0]
        if top_content['relevance'] > 0.3:
            answer_parts.append(f"Based on the available information in {top_content['filename']}:")
            answer_parts.append(f"\n{top_content['content'][:400]}...")
        else:
            answer_parts.append("Based on the available documents, I found some related information, but it may not directly answer your specific question:")
            answer_parts.append(f"\n{top_content['content'][:300]}...")
        
        # Add additional relevant sources if available
        if len(relevant_content) > 1:
            additional_sources = [c['source'] for c in relevant_content[1:3]]
            unique_sources = list(set(additional_sources))
            if unique_sources:
                answer_parts.append(f"\n\nAdditional context was found in: {', '.join(unique_sources)} sources.")
        
        return '\n'.join(answer_parts)
    
    def render_detailed_comparison(self, baseline_result, agentic_result, baseline_time):
        """Render detailed comparison metrics."""
        st.subheader("📊 Detailed Comparison")
        
        # Create comparison metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>Response Time</h4>
                <p><strong>Baseline:</strong> {:.2f}s</p>
                <p><strong>Graph-R1:</strong> {:.2f}s</p>
            </div>
            """.format(baseline_time, agentic_result['response_time']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>Sources Used</h4>
                <p><strong>Baseline:</strong> Text only</p>
                <p><strong>Graph-R1:</strong> {}</p>
            </div>
            """.format(', '.join(agentic_result['sources_used'])), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>Intelligence Level</h4>
                <p><strong>Baseline:</strong> Similarity search</p>
                <p><strong>Graph-R1:</strong> Agentic reasoning</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h4>Interpretability</h4>
                <p><strong>Baseline:</strong> Limited</p>
                <p><strong>Graph-R1:</strong> Complete audit trail</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Advantages breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🔍 Graph-R1 Advantages")
            st.write("✅ **Multi-source intelligence** - Combines text, visual, and business data")
            st.write("✅ **Agentic reasoning** - LLM-driven path planning and exploration")
            st.write("✅ **Complete transparency** - Full audit trail of decisions")
            st.write("✅ **Adaptive exploration** - Dynamic stopping and path pruning")
            st.write("✅ **Cross-modal understanding** - Unified embedding space")
        
        with col2:
            st.write("### ⚖️ Trade-offs")
            st.write("⚠️ **Complexity** - More sophisticated architecture")
            st.write("⚠️ **Response time** - Additional processing for reasoning")
            st.write("⚠️ **Token usage** - LLM calls for path planning")
            st.write("✅ **Quality** - Higher quality, more comprehensive answers")
            st.write("✅ **Insights** - Discovers connections across sources")
    
    def render_reasoning_visualization(self, agentic_result):
        """Render interactive reasoning chain visualization."""
        st.subheader("🔗 Reasoning Chain Visualization")
        
        if 'best_path' in agentic_result and agentic_result['best_path']:
            best_path = agentic_result['best_path']
            
            # Create reasoning steps
            steps_data = []
            for i, node in enumerate(best_path.nodes):
                steps_data.append({
                    'Step': i + 1,
                    'Source': node.node.source_type,
                    'Confidence': node.confidence,
                    'Reasoning': node.reasoning,
                    'Content Preview': node.node.content[:100] + "..."
                })
            
            # Display as interactive table
            df = pd.DataFrame(steps_data)
            st.dataframe(df, use_container_width=True)
            
            # Create path visualization
            fig = go.Figure()
            
            # Add nodes
            for i, node in enumerate(best_path.nodes):
                color_map = {
                    'text': '#1976d2',
                    'visual': '#388e3c', 
                    'salesforce': '#f57c00'
                }
                
                fig.add_trace(go.Scatter(
                    x=[i],
                    y=[node.confidence],
                    mode='markers+text',
                    marker=dict(
                        size=20,
                        color=color_map.get(node.node.source_type, '#666666'),
                        line=dict(width=2, color='white')
                    ),
                    text=[f"Step {i+1}<br>{node.node.source_type}"],
                    textposition="top center",
                    name=node.node.source_type,
                    showlegend=(i == 0)  # Only show legend for first occurrence
                ))
            
            # Add path connections
            if len(best_path.nodes) > 1:
                x_coords = list(range(len(best_path.nodes)))
                y_coords = [node.confidence for node in best_path.nodes]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0.3)', width=2),
                    name='Reasoning Path',
                    showlegend=True
                ))
            
            fig.update_layout(
                title="Graph Traversal Path with Confidence Scores",
                xaxis_title="Reasoning Step",
                yaxis_title="Confidence Score",
                yaxis=dict(range=[0, 1]),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("No reasoning path available for visualization")
    
    def render_audit_trail(self, audit_trail):
        """Render complete audit trail."""
        st.subheader("📋 Complete Audit Trail")
        
        if audit_trail:
            st.markdown('<div class="audit-trail">', unsafe_allow_html=True)
            
            for entry in audit_trail:
                timestamp = entry['timestamp']
                event_type = entry['event_type']
                
                if event_type == 'query_analysis':
                    st.write(f"[{timestamp:.2f}s] 🔍 **Query Analysis**: {entry['reasoning']}")
                
                elif event_type == 'entry_point_selection':
                    st.write(f"[{timestamp:.2f}s] 🎯 **Entry Points**: Selected {entry['count']} starting points")
                
                elif event_type == 'path_decision':
                    st.write(f"[{timestamp:.2f}s] 🚶 **Path Decision**: {entry['decision']} - {entry['reasoning']}")
                
                elif event_type == 'stopping_decision':
                    budget = entry['budget_used']
                    st.write(f"[{timestamp:.2f}s] 🛑 **Completion**: {entry['stopping_reason']}")
                    st.write(f"    📊 Final stats: {budget['nodes_visited']} nodes, {budget['hops_used']} hops")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No audit trail available")
    
    def render_system_stats(self):
        """Render system statistics and hypergraph info."""
        if st.session_state.hypergraph_built:
            st.subheader("📊 System Statistics")
            
            # Get hypergraph stats
            stats = st.session_state.hypergraph_builder.get_hypergraph_stats()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Nodes", stats['total_nodes'])
                st.metric("Total Edges", stats['total_edges'])
            
            with col2:
                st.write("**Nodes by Source:**")
                for source, count in stats['nodes_by_source'].items():
                    st.write(f"- {source}: {count}")
            
            with col3:
                st.write("**Edges by Type:**")
                for edge_type, count in stats['edges_by_type'].items():
                    st.write(f"- {edge_type}: {count}")
            
            # Unified embedding space info
            st.write("**Unified Embedding Space:**")
            st.write(f"- Target dimension: {stats['unified_dimension']}D")
            st.write(f"- Total projections: {stats['projection_stats']['total_projections']}")
            
            # Source indicators
            st.write("**Active Sources:**")
            sources_html = ""
            for source in stats['nodes_by_source'].keys():
                if source == 'text':
                    sources_html += '<span class="source-indicator source-text">📄 Text</span>'
                elif source == 'visual':
                    sources_html += '<span class="source-indicator source-visual">🖼️ Visual</span>'
                elif source == 'salesforce':
                    sources_html += '<span class="source-indicator source-salesforce">🏢 Salesforce</span>'
            
            st.markdown(sources_html, unsafe_allow_html=True)

def main():
    """Main application entry point."""
    demo = GraphR1Demo()
    
    # Render header
    demo.render_header()
    
    # Environment variables are checked in the sidebar with better user feedback
    
    # Render sidebar and get settings
    settings = demo.render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🚀 Query Interface", "📊 System Stats", "📖 About"])
    
    with tab1:
        demo.render_query_interface(settings)
    
    with tab2:
        demo.render_system_stats()
    
    with tab3:
        st.subheader("About Graph-R1 Agentic RAG")
        
        st.write("""
        ### 🧠 What is Graph-R1?
        
        Graph-R1 is a next-generation agentic RAG system that goes beyond traditional similarity search to provide:
        
        **🔍 True Agentic Behavior:**
        - LLM-driven query analysis and path planning
        - Dynamic graph traversal with intelligent stopping
        - Multi-source reasoning across text, visual, and business data
        
        **🏗️ Unified Architecture:**
        - Hypergraph construction with cross-modal embeddings
        - 512D unified embedding space for all modalities
        - Hierarchical relationships and semantic connections
        
        **🔗 Interpretable Reasoning:**
        - Complete audit trail of all decisions
        - Path visualization and confidence tracking
        - Professional output without source annotations
        
        ### 🆚 Compared to Traditional RAG:
        
        | Feature | Traditional RAG | Graph-R1 Agentic |
        |---------|-----------------|-------------------|
        | Sources | Single type | Multi-source |
        | Search | Similarity only | Graph traversal |
        | Reasoning | Limited | Complete audit trail |
        | Intelligence | Reactive | Proactive planning |
        | Transparency | Minimal | Full interpretability |
        
        ### 🎯 Key Innovations:
        
        1. **Cross-Modal Projection**: Unifies text (512D), visual (128D→512D), and business data
        2. **LLM Path Planning**: Intelligent query analysis and traversal strategy
        3. **Budgeted Retrieval**: Dynamic resource allocation and early stopping
        4. **Reasoning Chains**: Complete transparency of decision process
        5. **Graph Visualization**: Interactive exploration of reasoning paths
        """)

if __name__ == "__main__":
    main()