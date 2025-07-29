"""
Glassmorphism UI Prototype with Collapsible Sidebar

This prototype tests:
1. Collapsible sidebar with session state management
2. Glassmorphism design system with dark theme
3. Responsive layout that maximizes screen real estate
4. All visual components with elegant glass styling
"""

import streamlit as st
import time
from datetime import datetime

# Page config
st.set_page_config(
    page_title="🎨 Glassmorphism UI Prototype",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {
            'query': 'What is the revenue breakdown by region?',
            'answer': 'Based on the financial data, North America accounts for 45% of revenue ($2.3M), Europe 30% ($1.5M), and Asia-Pacific 25% ($1.3M).',
            'source_type': 'text',
            'timestamp': datetime.now(),
            'token_breakdown': {
                'query_tokens': 8,
                'text_rag_tokens': 156,
                'vlm_analysis_tokens': 0,
                'salesforce_api_tokens': 0,
                'reranker_tokens': 10,
                'response_tokens': 45,
                'total_tokens': 219
            }
        },
        {
            'query': 'Show me the organizational chart',
            'answer': 'I found the organizational chart in the HR document. It shows CEO at the top, followed by 4 VPs (Engineering, Sales, Marketing, Operations), with team leads and individual contributors below.',
            'source_type': 'visual',
            'timestamp': datetime.now(),
            'token_breakdown': {
                'query_tokens': 6,
                'text_rag_tokens': 0,
                'vlm_analysis_tokens': 284,
                'salesforce_api_tokens': 0,
                'reranker_tokens': 10,
                'response_tokens': 52,
                'total_tokens': 352
            }
        }
    ]

# Global glassmorphism CSS
def load_glassmorphism_css():
    st.markdown("""
    <style>
    /* Cache buster v2.0 */
    /* Import elegant fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    /* Global dark theme background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Main content area styling */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: none !important;
    }
    
    /* Glass container base class */
    .glass-container {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        padding: 0.75rem !important;
        margin: 0.4rem 0 !important;
        transition: all 0.3s ease !important;
    }
    
    .glass-container:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Glass header styling */
    .glass-header {
        background: rgba(255, 255, 255, 0.12) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        padding: 1rem 1.5rem !important;
        margin-bottom: 1rem !important;
        text-align: center !important;
    }
    
    .glass-header h1 {
        color: #ffffff !important;
        font-size: 2.2rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
    }
    
    .glass-header p {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 1rem !important;
        margin: 0.5rem 0 0 0 !important;
        font-weight: 300 !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Sidebar content styling */
    .css-1d391kg .stMarkdown,
    .css-1d391kg .stButton,
    .css-1d391kg h1,
    .css-1d391kg h2,
    .css-1d391kg h3,
    .css-1d391kg h4 {
        color: #ffffff !important;
    }
    
    /* Sidebar button styling */
    .css-1d391kg .stButton > button {
        background: rgba(100, 181, 246, 0.2) !important;
        border: 1px solid rgba(100, 181, 246, 0.4) !important;
        color: #64b5f6 !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Sidebar text elements */
    .css-1d391kg p,
    .css-1d391kg span,
    .css-1d391kg div {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Sidebar specific overrides */
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(100, 181, 246, 0.2) !important;
        border: 1px solid rgba(100, 181, 246, 0.4) !important;
        color: #64b5f6 !important;
        backdrop-filter: blur(10px) !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(100, 181, 246, 0.3) !important;
        border-color: rgba(100, 181, 246, 0.6) !important;
        color: #ffffff !important;
    }
    
    /* Compact status cards */
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 0.75rem;
        margin: 1rem 0;
    }
    
    .status-card {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        padding: 0.75rem !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
    }
    
    .status-card:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        transform: scale(1.02) !important;
    }
    
    .status-label {
        font-size: 0.7rem !important;
        font-weight: 500 !important;
        color: rgba(255, 255, 255, 0.7) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 0.25rem !important;
    }
    
    .status-value {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #64b5f6 !important;
        margin: 0 !important;
    }
    
    /* Query interface styling */
    .query-container {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(15px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Chat bubble styling */
    .chat-bubble {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(12px) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        padding: 0.75rem 1rem !important;
        margin: 0.5rem 0 !important;
        transition: all 0.3s ease !important;
    }
    
    .chat-bubble:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        border-color: rgba(100, 181, 246, 0.4) !important;
    }
    
    .chat-query {
        color: #81c784 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .chat-answer {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 0.9rem !important;
        line-height: 1.5 !important;
        margin-bottom: 0.75rem !important;
    }
    
    .chat-meta {
        color: rgba(255, 255, 255, 0.6) !important;
        font-size: 0.75rem !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
    }
    
    /* Toggle button styling */
    .sidebar-toggle {
        background: rgba(100, 181, 246, 0.2) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(100, 181, 246, 0.3) !important;
        border-radius: 8px !important;
        color: #64b5f6 !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        margin-bottom: 1rem !important;
    }
    
    .sidebar-toggle:hover {
        background: rgba(100, 181, 246, 0.3) !important;
        border-color: rgba(100, 181, 246, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Text styling improvements */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    p, div, span, label {
        color: rgba(255, 255, 255, 0.85) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Glassmorphic form elements styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        background: rgba(255, 255, 255, 0.15) !important;
        border-color: #64b5f6 !important;
        box-shadow: 0 0 0 2px rgba(100, 181, 246, 0.3), 0 6px 20px rgba(0, 0, 0, 0.15) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
        outline: none !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    
    /* Ensure text input labels are visible */
    .stTextInput > label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: rgba(100, 181, 246, 0.2) !important;
        border: 1px solid rgba(100, 181, 246, 0.3) !important;
        border-radius: 12px !important;
        color: #64b5f6 !important;
        font-weight: 500 !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: rgba(100, 181, 246, 0.3) !important;
        border-color: rgba(100, 181, 246, 0.5) !important;
        transform: translateY(-2px) !important;
        color: #ffffff !important;
    }
    
    /* Hide Streamlit branding but keep native sidebar toggle functional */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hide any potential white boxes or default streamlit elements */
    .element-container {
        margin: 0 !important;
    }
    
    /* Ensure no white backgrounds appear */
    .stApp > div {
        background: transparent !important;
    }
    
    /* Remove extra spacing and white containers */
    .block-container > div {
        gap: 0.5rem !important;
    }
    
    /* Style the native sidebar toggle when collapsed */
    [data-testid="collapsedControl"] {
        background: rgba(100, 181, 246, 0.3) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(100, 181, 246, 0.5) !important;
        backdrop-filter: blur(15px) !important;
        margin: 1rem !important;
        padding: 0.75rem !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="collapsedControl"]:hover {
        background: rgba(100, 181, 246, 0.4) !important;
        border-color: rgba(100, 181, 246, 0.7) !important;
        transform: translateX(2px) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
    }
    
    [data-testid="collapsedControl"] svg {
        color: #ffffff !important;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3)) !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

def create_glass_container(content, title=None):
    """Create a glassmorphic container"""
    if title:
        return f"""
        <div class="glass-container">
            <h3 style="color: #64b5f6; margin-bottom: 1rem; font-size: 1.1rem;">{title}</h3>
            {content}
        </div>
        """
    return f'<div class="glass-container">{content}</div>'

def create_status_card(label, value, color="#64b5f6"):
    """Create a compact status card"""
    return f"""
    <div class="status-card">
        <div class="status-label">{label}</div>
        <div class="status-value" style="color: {color};">{value}</div>
    </div>
    """

def create_chat_bubble(query, answer, source_type, timestamp, tokens):
    """Create a glassmorphic chat bubble with token counter"""
    source_colors = {
        'text': '#81c784',
        'visual': '#64b5f6', 
        'salesforce': '#ffb74d'
    }
    
    source_icons = {
        'text': '📝',
        'visual': '🖼️',
        'salesforce': '🏢'
    }
    
    color = source_colors.get(source_type, '#64b5f6')
    icon = source_icons.get(source_type, '📝')
    
    # Create compact token counter for this specific result
    token_counter = f"""
    <div style="background: rgba(255, 255, 255, 0.08); backdrop-filter: blur(10px); border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.15); padding: 8px 12px; margin: 8px 0 4px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11px;">
            <div style="text-align: center; flex: 1;">
                <div style="color: rgba(255, 255, 255, 0.6); font-size: 9px; text-transform: uppercase;">Query</div>
                <div style="color: #64b5f6; font-weight: 600;">{tokens['query_tokens']}</div>
            </div>
            <div style="text-align: center; flex: 1;">
                <div style="color: rgba(255, 255, 255, 0.6); font-size: 9px; text-transform: uppercase;">Text RAG</div>
                <div style="color: #64b5f6; font-weight: 600;">{tokens['text_rag_tokens']}</div>
            </div>
            <div style="text-align: center; flex: 1;">
                <div style="color: rgba(255, 255, 255, 0.6); font-size: 9px; text-transform: uppercase;">VLM</div>
                <div style="color: #64b5f6; font-weight: 600;">{tokens['vlm_analysis_tokens']}</div>
            </div>
            <div style="text-align: center; flex: 1;">
                <div style="color: rgba(255, 255, 255, 0.6); font-size: 9px; text-transform: uppercase;">SF</div>
                <div style="color: #64b5f6; font-weight: 600;">{tokens['salesforce_api_tokens']}</div>
            </div>
            <div style="text-align: center; flex: 1;">
                <div style="color: rgba(255, 255, 255, 0.6); font-size: 9px; text-transform: uppercase;">Re-rank</div>
                <div style="color: #64b5f6; font-weight: 600;">{tokens['reranker_tokens']}</div>
            </div>
            <div style="text-align: center; flex: 1;">
                <div style="color: rgba(255, 255, 255, 0.6); font-size: 9px; text-transform: uppercase;">Response</div>
                <div style="color: #64b5f6; font-weight: 600;">{tokens['response_tokens']}</div>
            </div>
            <div style="text-align: center; flex: 1; border-left: 1px solid rgba(255, 255, 255, 0.2); padding-left: 8px;">
                <div style="color: rgba(255, 255, 255, 0.6); font-size: 9px; text-transform: uppercase;">Total</div>
                <div style="color: #81c784; font-weight: 600;">{tokens['total_tokens']}</div>
            </div>
        </div>
    </div>
    """
    
    return f"""
    <div class="chat-bubble">
        <div class="chat-query">🔍 {query}</div>
        <div class="chat-answer">{answer}</div>
        {token_counter}
        <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.75rem; display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem;">
            <span>{icon} {source_type.upper()} | {timestamp.strftime('%H:%M:%S')}</span>
        </div>
    </div>
    """

def main():
    # Load glassmorphism CSS
    load_glassmorphism_css()
    
    # Header
    st.markdown("""
    <div class="glass-header">
        <h1>🤖 Smart Document Assistant</h1>
        <p>Glassmorphism UI Prototype with Collapsible Sidebar</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar (using native Streamlit toggle)
    with st.sidebar:
        st.markdown('<h3 style="color: #64b5f6; font-size: 1.1rem; margin-bottom: 1rem;">🎛️ System Configuration</h3>', unsafe_allow_html=True)
        
        # Feature Status
        st.markdown('<h4 style="color: #81c784; font-size: 1rem; margin-bottom: 0.75rem;">📋 Feature Status</h4>', unsafe_allow_html=True)
        
        # Create compact status grid
        status_html = f"""
        <div class="status-grid">
            {create_status_card("Text RAG", "✅ Ready", "#81c784")}
            {create_status_card("ColPali", "✅ GPU", "#64b5f6")}
            {create_status_card("Salesforce", "⚠️ Config", "#ffb74d")}
            {create_status_card("Re-ranker", "✅ BGE", "#81c784")}
            {create_status_card("GPU Mode", "⚡ Fast", "#64b5f6")}
            {create_status_card("Queries", str(len(st.session_state.chat_history)), "#64b5f6")}
        </div>
        """
        st.markdown(status_html, unsafe_allow_html=True)
        
        # System Status Section
        st.markdown('<h4 style="color: #ffb74d; font-size: 1rem; margin-bottom: 0.75rem;">📊 System Status</h4>', unsafe_allow_html=True)
        
        # Compact system info
        system_info = f"""
        <div style="background: rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1rem; margin: 0.5rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: rgba(255, 255, 255, 0.7); font-size: 0.8rem;">Mode:</span>
                <span style="color: #64b5f6; font-weight: 500;">GPU</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: rgba(255, 255, 255, 0.7); font-size: 0.8rem;">Sources:</span>
                <span style="color: #64b5f6; font-weight: 500;">3 Active</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: rgba(255, 255, 255, 0.7); font-size: 0.8rem;">Status:</span>
                <span style="color: #81c784; font-weight: 500;">Ready</span>
            </div>
        </div>
        """
        st.markdown(system_info, unsafe_allow_html=True)
        
        # Clear button
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2.5, 1.2])
    
    with col1:
        # Query Interface - Combined in single glass panel
        query_panel_content = """
        <div style="padding: 0.5rem;">
            <h3 style="color: #64b5f6; margin-bottom: 1rem; font-size: 1.1rem;">💬 Query Interface</h3>
        </div>
        """
        st.markdown(create_glass_container(query_panel_content), unsafe_allow_html=True)
        
        # Query input with glassmorphic styling inside the panel
        with st.container():
            query = st.text_input(
                "Ask your question:",
                placeholder="e.g., What are the Q3 sales figures?",
                key="query_input"
            )
            
            col_submit, col_clear = st.columns([3, 1])
            with col_submit:
                if st.button("🔍 Search", type="primary", use_container_width=True):
                    if query:
                        # Simulate processing
                        with st.spinner("🔄 Searching across all sources..."):
                            time.sleep(1)
                        
                        # Add to history
                        new_chat = {
                            'query': query,
                            'answer': 'This is a simulated response for the glassmorphism UI prototype. The answer would contain relevant information from the selected source.',
                            'source_type': 'text',
                            'timestamp': datetime.now(),
                            'token_breakdown': {
                                'query_tokens': len(query.split()),
                                'text_rag_tokens': 89,
                                'vlm_analysis_tokens': 0,
                                'salesforce_api_tokens': 0,
                                'reranker_tokens': 10,
                                'response_tokens': 34,
                                'total_tokens': len(query.split()) + 89 + 10 + 34
                            }
                        }
                        st.session_state.chat_history.insert(0, new_chat)
                        st.rerun()
            
            with col_clear:
                if st.button("🗑️ Clear", use_container_width=True):
                    # Clear the input
                    st.rerun()
        
        # Note: Token counter will appear in each query result below
        
        # Recent Results - Compact container
        if st.session_state.chat_history:
            results_header = """
            <div style="padding: 0.5rem 0 0.25rem 0;">
                <h3 style="color: #64b5f6; margin-bottom: 0.5rem; font-size: 1.1rem;">📝 Recent Results</h3>
            </div>
            """
            st.markdown(create_glass_container(results_header), unsafe_allow_html=True)
            
            for chat in st.session_state.chat_history[:3]:  # Show last 3 chats
                bubble_html = create_chat_bubble(
                    chat['query'],
                    chat['answer'],
                    chat['source_type'],
                    chat['timestamp'],
                    chat['token_breakdown']
                )
                st.markdown(bubble_html, unsafe_allow_html=True)
    
    with col2:
        # Document Management Section
        st.markdown(create_glass_container("""
            <h3 style="color: #64b5f6; margin-bottom: 1rem;">📁 Document Management</h3>
            <div style="background: rgba(255, 255, 255, 0.05); border: 2px dashed rgba(100, 181, 246, 0.3); border-radius: 12px; padding: 2rem 1rem; text-align: center; margin: 1rem 0;">
                <div style="color: #64b5f6; font-size: 2rem; margin-bottom: 0.5rem;">📤</div>
                <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem;">Drop files here or click to upload</div>
                <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.75rem; margin-top: 0.5rem;">PDF, DOCX, TXT, CSV supported</div>
            </div>
        """), unsafe_allow_html=True)
        
        # Processing Status
        st.markdown(create_glass_container("""
            <h4 style="color: #64b5f6; margin-bottom: 1rem;">⚡ Processing Status</h4>
            <div style="margin: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span style="color: rgba(255, 255, 255, 0.7); font-size: 0.8rem;">Documents:</span>
                    <span style="color: #81c784; font-weight: 500;">5 processed</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span style="color: rgba(255, 255, 255, 0.7); font-size: 0.8rem;">Text Chunks:</span>
                    <span style="color: #64b5f6; font-weight: 500;">1,247</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span style="color: rgba(255, 255, 255, 0.7); font-size: 0.8rem;">Visual Pages:</span>
                    <span style="color: #64b5f6; font-weight: 500;">23</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: rgba(255, 255, 255, 0.7); font-size: 0.8rem;">Last Updated:</span>
                    <span style="color: rgba(255, 255, 255, 0.8); font-weight: 500;">2 min ago</span>
                </div>
            </div>
        """), unsafe_allow_html=True)
    
    # Footer info
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(255, 255, 255, 0.05); border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.1);">
        <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.8rem;">🎨 Glassmorphism UI Prototype | Use native sidebar toggle to collapse/expand</div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()