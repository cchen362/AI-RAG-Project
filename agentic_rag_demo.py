"""
Agentic RAG Demo Application

This demonstrates the new federated RAG architecture with true agentic behavior:
- ReAct-style reasoning (Think → Act → Observe → Repeat)
- LLM-driven query analysis and retrieval strategy
- Proper ColPali MaxSim scoring (no patch averaging)
- Multi-source intelligent routing (Text + Visual + Salesforce)
- Complete reasoning transparency and audit trails

Key Innovation: The agent actually reasons about each query individually,
deciding which sources to search and adapting based on results.
"""

import streamlit as st
import asyncio
import logging
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import agentic components
try:
    from src.agentic_rag_orchestrator import AgenticRAGOrchestrator, create_agentic_rag_orchestrator
    from src.colpali_maxsim_retriever import ColPaliMaxSimRetriever
except ImportError as e:
    st.error(f"Failed to import agentic components: {e}")
    st.stop()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Demo - True AI Reasoning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
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
    
    .reasoning-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .agent-thinking {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .agent-action {
        background: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .agent-observation {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .source-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .metric-box {
        background: #ffffff;
        border: 1px solid #e9ecef;
        padding: 1rem;
        text-align: center;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'agentic_orchestrator' not in st.session_state:
        st.session_state.agentic_orchestrator = None
    
    if 'reasoning_history' not in st.session_state:
        st.session_state.reasoning_history = []
    
    if 'documents_uploaded' not in st.session_state:
        st.session_state.documents_uploaded = False
    
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False

def load_agentic_config() -> Dict[str, Any]:
    """Load configuration for agentic RAG system."""
    return {
        'chunk_size': 800,
        'chunk_overlap': 150,
        'embedding_model': 'local',
        'max_retrieved_chunks': 5,
        'max_reasoning_steps': 5,
        'min_confidence_threshold': 0.7,
        'model_name': 'vidore/colqwen2-v1.0',
        'device': 'auto',
        'max_pages_per_doc': 50,
        'cache_embeddings': True,
        'cache_dir': 'cache/embeddings'
    }

@st.cache_resource
def initialize_agentic_system():
    """Initialize the agentic RAG orchestrator."""
    try:
        st.info("🧠 Initializing Agentic RAG System...")
        
        config = load_agentic_config()
        orchestrator = create_agentic_rag_orchestrator(config)
        
        st.success("✅ Agentic RAG System initialized successfully!")
        return orchestrator
    except Exception as e:
        st.error(f"❌ Failed to initialize agentic system: {e}")
        return None

def display_main_header():
    """Display the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Agentic RAG Demo</h1>
        <h3>True AI Reasoning with ReAct Framework</h3>
        <p>Watch the AI agent think, plan, and adapt its retrieval strategy in real-time</p>
    </div>
    """, unsafe_allow_html=True)

def display_system_status(orchestrator: AgenticRAGOrchestrator):
    """Display system status and capabilities."""
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        if orchestrator:
            system_info = orchestrator.get_system_info()
            components = system_info.get('components', {})
            
            # Component status
            st.markdown("#### Components")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Text RAG", "✅" if components.get('text_rag') else "❌")
                st.metric("Reasoning LLM", "✅" if components.get('reasoning_llm') else "❌")
            
            with col2:
                st.metric("ColPali MaxSim", "✅" if components.get('colpali_retriever') else "❌")
                st.metric("Salesforce", "✅" if components.get('salesforce_connector') else "❌")
            
            # Agentic settings
            st.markdown("#### Agentic Settings")
            st.write(f"**Max Reasoning Steps**: {system_info.get('max_reasoning_steps', 'N/A')}")
            st.write(f"**Confidence Threshold**: {system_info.get('min_confidence_threshold', 'N/A')}")
            st.write(f"**Last Reasoning Steps**: {system_info.get('last_reasoning_steps', 0)}")
        else:
            st.error("❌ System not initialized")

def upload_documents(orchestrator: AgenticRAGOrchestrator):
    """Handle document upload and processing."""
    with st.sidebar:
        st.markdown("### 📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload documents for analysis",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF files for visual analysis, text files for text RAG"
        )
        
        if uploaded_files and st.button("🔄 Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files temporarily
                    temp_paths = []
                    for file in uploaded_files:
                        temp_path = f"temp_{file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())
                        temp_paths.append(temp_path)
                    
                    # Process documents with different retrievers
                    text_files = [p for p in temp_paths if p.lower().endswith(('.txt', '.docx'))]
                    pdf_files = [p for p in temp_paths if p.lower().endswith('.pdf')]
                    
                    results = {'text': None, 'visual': None}
                    
                    # Process text documents
                    if text_files and hasattr(orchestrator, 'text_rag'):
                        text_results = orchestrator.text_rag.add_documents(text_files)
                        results['text'] = text_results
                    
                    # Process PDF documents with MaxSim
                    if pdf_files and hasattr(orchestrator, 'colpali_retriever'):
                        visual_results = orchestrator.colpali_retriever.add_documents(pdf_files)
                        results['visual'] = visual_results
                    
                    # Display results
                    if results['text']:
                        st.success(f"✅ Processed {len(results['text'].get('successful', []))} text documents")
                    
                    if results['visual']:
                        st.success(f"✅ Processed {len(results['visual'].get('successful', []))} visual documents")
                        st.info(f"📊 Total patches preserved: {results['visual'].get('patches_preserved', 0)}")
                    
                    st.session_state.documents_uploaded = True
                    
                    # Cleanup
                    for path in temp_paths:
                        if os.path.exists(path):
                            os.remove(path)
                            
                except Exception as e:
                    st.error(f"❌ Document processing failed: {e}")

def display_reasoning_trace(reasoning_history: List[Dict[str, Any]]):
    """Display the agent's reasoning trace with enhanced visualization."""
    if not reasoning_history:
        st.info("🤔 No reasoning trace available yet. Submit a query to see the agent think!")
        return
    
    st.markdown("### 🧠 Agent Reasoning Trace")
    st.markdown("*Follow the agent's thought process step by step*")
    
    for i, step in enumerate(reasoning_history, 1):
        step_type = step.get('step_type', 'unknown')
        
        if step_type == 'initial_reasoning':
            with st.expander(f"💭 Step {i}: Initial Analysis", expanded=True):
                st.markdown(f"""
                <div class="agent-thinking">
                    <h4>🎯 Query Analysis</h4>
                    <p>{step.get('reasoning', 'No reasoning available')}</p>
                    
                    <h4>📋 Planned Approach</h4>
                    <p>{step.get('plan', 'No plan available')}</p>
                    
                    <h4>🎲 Initial Confidence</h4>
                    <p>{step.get('initial_confidence', 0.0):.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        elif step_type == 'action_decision':
            with st.expander(f"🎯 Step {i}: Action Decision", expanded=True):
                st.markdown(f"""
                <div class="agent-action">
                    <h4>💭 Agent Thinking</h4>
                    <p>{step.get('thinking', 'No thinking available')}</p>
                    
                    <h4>⚡ Action Chosen</h4>
                    <p><strong>{step.get('action', 'Unknown action')}</strong></p>
                    
                    <h4>🎯 Target Sources</h4>
                    <p>{step.get('target_sources', 'No target specified')}</p>
                    
                    <h4>💡 Reasoning</h4>
                    <p>{step.get('reasoning', 'No reasoning available')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        elif step_type == 'observation':
            with st.expander(f"👁️ Step {i}: Result Observation", expanded=True):
                confidence = step.get('confidence_score', 0.0)
                quality = step.get('results_quality', 'unknown')
                
                # Color code confidence
                if confidence >= 0.8:
                    conf_color = "#28a745"  # Green
                elif confidence >= 0.6:
                    conf_color = "#ffc107"  # Yellow
                else:
                    conf_color = "#dc3545"  # Red
                
                st.markdown(f"""
                <div class="agent-observation">
                    <h4>🔍 Agent Evaluation</h4>
                    <p>{step.get('thinking', 'No evaluation available')}</p>
                    
                    <h4>📊 Quality Assessment</h4>
                    <p><strong>{quality.title()}</strong></p>
                    
                    <h4>✅ Information Found</h4>
                    <p>{step.get('information_found', 'No information specified')}</p>
                    
                    <h4>❓ Information Gaps</h4>
                    <p>{step.get('information_gaps', 'No gaps specified')}</p>
                    
                    <h4>🎯 Confidence Score</h4>
                    <p style="color: {conf_color}; font-size: 1.2em; font-weight: bold;">{confidence:.2f}</p>
                    
                    <h4>💡 Recommendation</h4>
                    <p>{step.get('recommendation', 'No recommendation')}</p>
                </div>
                """, unsafe_allow_html=True)

def display_query_interface(orchestrator: AgenticRAGOrchestrator):
    """Display the main query interface."""
    st.markdown("### 🔍 Query the Agentic System")
    
    # Example queries
    with st.expander("💡 Example Queries", expanded=False):
        st.markdown("""
        **Visual Queries** (will trigger ColPali MaxSim):
        - "What's the retrieval time in a ColPali pipeline?"
        - "Based on the chart, what are the performance metrics?"
        - "Show me the diagram that explains the architecture"
        
        **Business Queries** (will trigger Salesforce):
        - "What's the cancellation policy for bookings?"
        - "How do I handle customer complaints?"
        - "What are the refund procedures?"
        
        **Complex Queries** (will trigger multi-source search):
        - "Compare the performance metrics with our service policies"
        - "What technical specifications support our booking procedures?"
        """)
    
    # Query input
    query = st.text_input(
        "Enter your query:",
        placeholder="Ask me anything - I'll reason through the best search strategy...",
        help="The agent will analyze your query and decide which sources to search"
    )
    
    # Query execution
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        submit_query = st.button("🧠 Ask Agent", type="primary")
    
    with col2:
        show_reasoning = st.checkbox("Show Reasoning", value=True)
    
    with col3:
        clear_history = st.button("🗑️ Clear History")
    
    if clear_history:
        st.session_state.reasoning_history = []
        st.experimental_rerun()
    
    if submit_query and query:
        if not orchestrator:
            st.error("❌ Agentic system not initialized")
            return
        
        if not st.session_state.documents_uploaded:
            st.warning("⚠️ No documents uploaded. Upload documents first for better results.")
        
        # Execute agentic query
        with st.spinner("🧠 Agent is thinking and searching..."):
            try:
                # Run async agentic processing
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(
                    orchestrator.process_query_agentically(query)
                )
                
                if result.get('success', False):
                    # Store reasoning history
                    st.session_state.reasoning_history = result.get('reasoning_trace', [])
                    
                    # Display main answer
                    st.markdown("### 🎯 Agent's Answer")
                    st.markdown(f"""
                    <div class="reasoning-box">
                        <h4>Response</h4>
                        {result.get('answer', 'No answer generated')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h4>🔄</h4>
                            <p>Reasoning Steps</p>
                            <h3>{result.get('total_reasoning_steps', 0)}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h4>🎯</h4>
                            <p>Final Confidence</p>
                            <h3>{result.get('final_confidence', 0.0):.2f}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        sources_used = len(result.get('sources_used', []))
                        st.markdown(f"""
                        <div class="metric-box">
                            <h4>📚</h4>
                            <p>Sources Used</p>
                            <h3>{sources_used}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        decisions = len([d for d in result.get('agent_decisions', []) if d])
                        st.markdown(f"""
                        <div class="metric-box">
                            <h4>🧠</h4>
                            <p>Decisions Made</p>
                            <h3>{decisions}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display reasoning trace if requested
                    if show_reasoning:
                        display_reasoning_trace(st.session_state.reasoning_history)
                        
                        # Add reasoning report download
                        if st.session_state.reasoning_history:
                            reasoning_report = orchestrator.generate_reasoning_report()
                            st.download_button(
                                "📥 Download Reasoning Report",
                                reasoning_report,
                                file_name=f"reasoning_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown"
                            )
                    
                else:
                    st.error(f"❌ Agentic processing failed: {result.get('error', 'Unknown error')}")
                    if result.get('reasoning_trace'):
                        st.session_state.reasoning_history = result['reasoning_trace']
                        if show_reasoning:
                            display_reasoning_trace(st.session_state.reasoning_history)
                
            except Exception as e:
                st.error(f"❌ Query execution failed: {e}")
                logger.error(f"Query execution error: {e}")

def display_architecture_info():
    """Display information about the agentic architecture."""
    with st.sidebar:
        st.markdown("### 🏗️ Architecture")
        
        with st.expander("🧠 Agentic Features", expanded=False):
            st.markdown("""
            **True Agentic Behavior:**
            - ReAct Framework (Think → Act → Observe)
            - LLM-driven query analysis
            - Dynamic retrieval strategy
            - Adaptive source selection
            - Complete reasoning transparency
            
            **Key Improvements:**
            - Fixed ColPali MaxSim scoring
            - No patch averaging
            - Query-specific routing
            - Multi-step reasoning
            - Audit trail generation
            """)
        
        with st.expander("📊 MaxSim Scoring", expanded=False):
            st.markdown("""
            **ColPali MaxSim Algorithm:**
            1. Preserve all 1030 patches (128D each)
            2. Encode query preserving token structure
            3. For each query token:
               - Compute similarity with ALL patches
               - Take MAX similarity per token
            4. SUM max similarities across tokens
            
            **Why This Matters:**
            - Patch averaging destroys spatial info
            - MaxSim preserves visual structure
            - Proper late interaction mechanism
            - Better visual understanding
            """)

def main():
    """Main application entry point."""
    initialize_session_state()
    display_main_header()
    
    # Initialize agentic system
    if not st.session_state.system_initialized:
        orchestrator = initialize_agentic_system()
        if orchestrator:
            st.session_state.agentic_orchestrator = orchestrator
            st.session_state.system_initialized = True
        else:
            st.error("❌ Failed to initialize agentic system. Please check your configuration.")
            st.stop()
    else:
        orchestrator = st.session_state.agentic_orchestrator
    
    # Display system status and architecture info
    display_system_status(orchestrator)
    display_architecture_info()
    
    # Document upload
    upload_documents(orchestrator)
    
    # Main query interface
    display_query_interface(orchestrator)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🧠 Agentic RAG Demo - True AI Reasoning with ReAct Framework</p>
        <p>Built with Streamlit | Enhanced with ColPali MaxSim | Powered by LLM Agents</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()