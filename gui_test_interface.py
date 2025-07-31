"""
GUI Testing Interface for Agentic RAG System

Interactive Streamlit-based interface for testing, demonstrating, and analyzing
the agentic RAG orchestrator with real-time reasoning visualization and 
side-by-side comparison with baseline system.
"""

import streamlit as st
import sys
import os
import time
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

# Add test directory to path
test_dir = os.path.join(os.path.dirname(__file__), 'test_agentic_rag')
sys.path.append(test_dir)

# Import agentic components
try:
    from test_harness import TestHarness
    from agentic_orchestrator import AgentStep, SourceType
    from agent_memory import AgentMemory
except ImportError as e:
    st.error(f"❌ Failed to import agentic components: {e}")
    st.error("Please ensure you're running from the project root directory")
    st.stop()

# Configure Streamlit
st.set_page_config(
    page_title="🧪 Agentic RAG Testing Interface",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for testing interface
def load_testing_interface_css():
    st.markdown("""
    <style>
    /* Testing Interface Specific Styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .reasoning-step {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .step-think { border-left-color: #4CAF50; }
    .step-retrieve { border-left-color: #2196F3; }
    .step-rethink { border-left-color: #FF9800; }
    .step-generate { border-left-color: #9C27B0; }
    
    .confidence-high { color: #4CAF50; font-weight: bold; }
    .confidence-medium { color: #FF9800; font-weight: bold; }
    .confidence-low { color: #f44336; font-weight: bold; }
    
    .comparison-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .comparison-side {
        flex: 1;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(5px);
        border-radius: 10px;
        padding: 1rem;
    }
    
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .source-indicator {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    
    .source-text-rag { background: #4CAF50; color: white; }
    .source-colpali { background: #2196F3; color: white; }
    .source-salesforce { background: #FF9800; color: white; }
    </style>
    """, unsafe_allow_html=True)

class AgenticTestInterface:
    """Main testing interface class with real-time visualization"""
    
    def __init__(self):
        self.test_harness = None
        self.test_results_cache = {}
        self.predefined_scenarios = [
            {
                "name": "Simple Technical Query",
                "query": "What is a transformer architecture in machine learning?",
                "category": "technical",
                "description": "Basic technical query to test fundamental capabilities"
            },
            {
                "name": "Attention Mechanism Deep Dive", 
                "query": "What is attention mechanism in transformers?",
                "category": "technical",
                "description": "Complex technical query requiring mathematical understanding"
            },
            {
                "name": "Multi-hop Complex Query",
                "query": "How do transformers use attention mechanisms for language modeling and what are the computational advantages?",
                "category": "complex",
                "description": "Multi-step reasoning requiring cross-source synthesis"
            },
            {
                "name": "Visual Content Query",
                "query": "Analyze any diagrams or visual content in the uploaded documents",
                "category": "visual", 
                "description": "ColPali visual analysis test"
            },
            {
                "name": "Business Context Query",
                "query": "What are the latest trends in artificial intelligence for business applications?",
                "category": "business",
                "description": "Salesforce knowledge base integration test"
            }
        ]
        
    def initialize_test_harness(self):
        """Initialize the test harness with progress indication"""
        if self.test_harness is None:
            with st.spinner("🚀 Initializing Agentic RAG Test Environment..."):
                try:
                    # Initialize with testing configuration
                    test_config = {
                        "max_conversation_length": 10,
                        "confidence_threshold": 0.7,
                        "max_reasoning_steps": 10,
                        "enable_memory": True,
                        "debug_mode": True
                    }
                    
                    self.test_harness = TestHarness(test_config)
                    
                    # Setup components with progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Setting up RAG components...")
                    progress_bar.progress(25)
                    
                    self.test_harness.setup_components(init_all=False)  # Skip ColPali for faster setup
                    progress_bar.progress(50)
                    
                    status_text.text("Agentic orchestrator ready...")
                    progress_bar.progress(100)
                    
                    status_text.text("✅ Test environment ready!")
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                    return True
                    
                except Exception as e:
                    st.error(f"❌ Failed to initialize test environment: {str(e)}")
                    return False
        return True
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🧪 Agentic RAG Testing Interface</h1>
            <p>Interactive testing and analysis of Graph-R1 inspired multi-turn reasoning</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with test controls"""
        st.sidebar.title("🛠️ Test Controls")
        
        # Test mode selection
        test_mode = st.sidebar.selectbox(
            "Test Mode",
            ["Interactive Testing", "Predefined Scenarios", "Performance Analysis", "Memory Inspection"],
            help="Choose testing approach"
        )
        
        # System status
        st.sidebar.subheader("🔧 System Status")
        if self.test_harness:
            st.sidebar.success("✅ Test Harness Ready")
            
            # Show available components
            components = {
                "Text RAG": "✅" if self.test_harness.rag_system else "❌",
                "ColPali Visual": "✅" if self.test_harness.colpali_retriever else "❌", 
                "Salesforce": "✅" if self.test_harness.salesforce_connector else "❌",
                "Agentic Orchestrator": "✅" if self.test_harness.agentic_orchestrator else "❌"
            }
            
            for component, status in components.items():
                st.sidebar.text(f"{status} {component}")
        else:
            st.sidebar.warning("⚠️ Test Harness Not Initialized")
        
        # Test configuration
        st.sidebar.subheader("⚙️ Configuration")
        max_steps = st.sidebar.slider("Max Reasoning Steps", 1, 15, 10)
        confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.7, 0.05)
        enable_comparison = st.sidebar.checkbox("Enable Baseline Comparison", True)
        
        return {
            "test_mode": test_mode,
            "max_steps": max_steps,
            "confidence_threshold": confidence_threshold,
            "enable_comparison": enable_comparison
        }
    
    def render_interactive_testing(self, config):
        """Render interactive testing interface"""
        st.subheader("🎯 Interactive Query Testing")
        
        # Query input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_area(
                "Enter your query:",
                placeholder="Ask anything about the documents...",
                height=100
            )
        
        with col2:
            st.write("**Quick Tests:**")
            for scenario in self.predefined_scenarios[:3]:  # Show top 3
                if st.button(scenario["name"], key=f"quick_{scenario['name']}"):
                    query = scenario["query"]
                    st.rerun()
        
        # Test execution
        if st.button("🚀 Run Agentic Test", type="primary") and query:
            self.run_test_query(query, config)
    
    def run_test_query(self, query: str, config: Dict):
        """Run test query with real-time visualization"""
        st.subheader("🧠 Real-Time Reasoning Process")
        
        # Create containers for real-time updates
        reasoning_container = st.container()
        results_container = st.container()
        
        with reasoning_container:
            st.write("**Reasoning Chain:**")
            reasoning_placeholder = st.empty()
            
        # Run agentic test
        start_time = time.time()
        
        try:
            with st.spinner("🤖 Agentic reasoning in progress..."):
                result = self.test_harness.run_single_test(
                    query, 
                    test_both=config["enable_comparison"]
                )
            
            execution_time = time.time() - start_time
            
            # Display reasoning chain
            self.display_reasoning_chain(result.get("agentic_response"), reasoning_placeholder)
            
            # Display results
            with results_container:
                self.display_test_results(result, execution_time, config["enable_comparison"])
                
        except Exception as e:
            st.error(f"❌ Test failed: {str(e)}")
            st.error(f"Error details: {type(e).__name__}")
            import traceback
            st.text(traceback.format_exc())
    
    def display_reasoning_chain(self, agentic_response, placeholder):
        """Display the reasoning chain with visual indicators"""
        if not agentic_response or not hasattr(agentic_response, 'reasoning_chain'):
            placeholder.warning("No reasoning chain available")
            return
        
        reasoning_html = ""
        step_colors = {
            AgentStep.THINK: "#4CAF50",
            AgentStep.RETRIEVE: "#2196F3", 
            AgentStep.RETHINK: "#FF9800",
            AgentStep.GENERATE: "#9C27B0"
        }
        
        for i, action in enumerate(agentic_response.reasoning_chain, 1):
            step_color = step_colors.get(action.step, "#666")
            confidence_class = self.get_confidence_class(action.confidence)
            
            source_badge = ""
            if action.source:
                source_name = action.source.value.replace("_", " ").title()
                source_badge = f'<span class="source-indicator source-{action.source.value.lower().replace("_", "-")}">{source_name}</span>'
            
            reasoning_html += f"""
            <div class="reasoning-step step-{action.step.value}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="color: {step_color}; margin: 0;">Step {i}: {action.step.value.upper()}</h4>
                    <div>
                        {source_badge}
                        <span class="{confidence_class}">Confidence: {action.confidence:.2f}</span>
                    </div>
                </div>
                <p><strong>Query:</strong> {action.query}</p>
                <p><strong>Reasoning:</strong> {action.reasoning}</p>
                {f'<p><strong>Result:</strong> {str(action.result)[:200]}...</p>' if action.result else ''}
            </div>
            """
        
        placeholder.markdown(reasoning_html, unsafe_allow_html=True)
    
    def display_test_results(self, result: Dict, execution_time: float, enable_comparison: bool):
        """Display comprehensive test results"""
        st.subheader("📊 Test Results")
        
        # Metrics grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Execution Time",
                f"{execution_time:.2f}s",
                help="Total test execution time"
            )
        
        with col2:
            agentic_steps = len(result.get("agentic_response", {}).get("reasoning_chain", []))
            st.metric(
                "Reasoning Steps", 
                agentic_steps,
                help="Number of reasoning steps taken"
            )
        
        with col3:
            agentic_confidence = getattr(result.get("agentic_response"), 'confidence', 0)
            st.metric(
                "Final Confidence",
                f"{agentic_confidence:.2f}",
                help="Agent's confidence in the final answer"
            )
        
        with col4:
            sources_used = len(set(
                action.source.value for action in result.get("agentic_response", {}).get("reasoning_chain", [])
                if action.source
            ))
            st.metric(
                "Sources Used",
                sources_used,
                help="Number of different sources queried"
            )
        
        # Response comparison
        if enable_comparison and "baseline_response" in result:
            st.subheader("🔍 Response Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🤖 Agentic Response")
                agentic_answer = getattr(result.get("agentic_response"), 'answer', 'No response')
                st.write(agentic_answer)
                
                # Agentic metrics
                st.markdown("**Metrics:**")
                st.write(f"• Steps: {agentic_steps}")
                st.write(f"• Confidence: {agentic_confidence:.2f}")
                st.write(f"• Sources: {sources_used}")
            
            with col2:
                st.markdown("### 📝 Baseline Response")
                baseline_answer = getattr(result.get("baseline_response"), 'answer', 'No response')
                st.write(baseline_answer)
                
                # Baseline metrics
                st.markdown("**Metrics:**")
                baseline_confidence = getattr(result.get("baseline_response"), 'confidence', 0)
                st.write(f"• Steps: 1 (single-turn)")
                st.write(f"• Confidence: {baseline_confidence:.2f}")
                st.write(f"• Sources: 1 (parallel selection)")
        else:
            st.subheader("📝 Agentic Response")
            agentic_answer = getattr(result.get("agentic_response"), 'answer', 'No response')
            st.write(agentic_answer)
    
    def render_predefined_scenarios(self, config):
        """Render predefined scenario testing"""
        st.subheader("📋 Predefined Test Scenarios")
        
        # Scenario selection
        scenario_names = [s["name"] for s in self.predefined_scenarios]
        selected_scenario = st.selectbox("Choose Test Scenario:", scenario_names)
        
        # Scenario details
        scenario = next(s for s in self.predefined_scenarios if s["name"] == selected_scenario)
        
        st.markdown(f"**Category:** {scenario['category'].title()}")
        st.markdown(f"**Description:** {scenario['description']}")
        st.markdown(f"**Query:** *{scenario['query']}*")
        
        # Run scenario
        if st.button("🧪 Run Scenario Test", type="primary"):
            self.run_test_query(scenario["query"], config)
    
    def render_performance_analysis(self):
        """Render performance analysis dashboard"""
        st.subheader("📈 Performance Analysis")
        
        # Placeholder for performance analytics
        st.info("🚧 Performance analysis dashboard coming soon...")
        
        # Mock performance data for demonstration
        performance_data = {
            "Query Type": ["Simple", "Complex", "Visual", "Business"],
            "Avg Steps (Agentic)": [6, 10, 4, 8],
            "Avg Steps (Baseline)": [1, 1, 1, 1],
            "Avg Confidence": [0.65, 0.49, 0.72, 0.58],
            "Avg Time (s)": [8.5, 12.3, 15.2, 9.1]
        }
        
        df = pd.DataFrame(performance_data)
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_steps = px.bar(
                df, 
                x="Query Type", 
                y=["Avg Steps (Agentic)", "Avg Steps (Baseline)"],
                title="Reasoning Steps Comparison",
                barmode="group"
            )
            st.plotly_chart(fig_steps, use_container_width=True)
        
        with col2:
            fig_time = px.line(
                df,
                x="Query Type",
                y="Avg Time (s)",
                title="Execution Time by Query Type",
                markers=True
            )
            st.plotly_chart(fig_time, use_container_width=True)
    
    def render_memory_inspection(self):
        """Render agent memory inspection interface"""
        st.subheader("🧠 Agent Memory Inspection")
        
        if self.test_harness and self.test_harness.agent_memory:
            memory = self.test_harness.agent_memory
            
            # Conversation history
            st.markdown("### 💬 Conversation History")
            conversation_turns = memory.get_conversation_history()
            
            if conversation_turns:
                for i, turn in enumerate(conversation_turns[-5:], 1):  # Show last 5
                    with st.expander(f"Turn {i}: {turn.query[:50]}..."):
                        st.write(f"**Query:** {turn.query}")
                        st.write(f"**Response:** {turn.response[:200]}...")
                        st.write(f"**Timestamp:** {turn.timestamp}")
                        st.write(f"**Sources:** {', '.join(turn.sources_used)}")
            else:
                st.info("No conversation history available")
            
            # Knowledge fragments
            st.markdown("### 🧩 Knowledge Fragments")
            fragments = memory.get_knowledge_fragments()
            
            if fragments:
                for fragment in fragments[-3:]:  # Show last 3
                    with st.expander(f"Fragment: {fragment.content[:50]}..."):
                        st.write(f"**Content:** {fragment.content}")
                        st.write(f"**Source:** {fragment.source}")
                        st.write(f"**Confidence:** {fragment.confidence:.2f}")
                        st.write(f"**Usage Count:** {fragment.usage_count}")
            else:
                st.info("No knowledge fragments available")
        else:
            st.warning("Agent memory not available")
    
    def get_confidence_class(self, confidence: float) -> str:
        """Get CSS class for confidence level"""
        if confidence >= 0.7:
            return "confidence-high"
        elif confidence >= 0.4:
            return "confidence-medium"
        else:
            return "confidence-low"

def main():
    """Main application entry point"""
    load_testing_interface_css()
    
    # Initialize interface
    interface = AgenticTestInterface()
    interface.render_header()
    
    # Initialize test harness
    if not interface.initialize_test_harness():
        st.stop()
    
    # Render sidebar and get config
    config = interface.render_sidebar()
    
    # Render main content based on mode
    if config["test_mode"] == "Interactive Testing":
        interface.render_interactive_testing(config)
    elif config["test_mode"] == "Predefined Scenarios":
        interface.render_predefined_scenarios(config)
    elif config["test_mode"] == "Performance Analysis":
        interface.render_performance_analysis()
    elif config["test_mode"] == "Memory Inspection":
        interface.render_memory_inspection()

if __name__ == "__main__":
    main()