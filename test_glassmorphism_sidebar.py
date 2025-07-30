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
from streamlit_extras.stylable_container import stylable_container

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
                'text_rag_tokens': 1856,
                'vlm_analysis_tokens': 0,
                'salesforce_api_tokens': 0,
                'reranker_tokens': 10,
                'response_tokens': 445,
                'total_tokens': 2319
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
    /* TRUE GLASSMORPHISM v9.0 - COMPLETE REDESIGN */
    /* Import elegant fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    /* VIBRANT GLASSMORPHIC BACKGROUND - Essential for glass effect */
    .stApp {
        background: 
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%),
            linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%) !important;
        font-family: 'Inter', sans-serif !important;
        min-height: 100vh !important;
    }
    
    /* Enhanced glassmorphic base styles */
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* QUERY INPUT FIELD ENHANCEMENT - High contrast for 2560x1440 displays */
    .stTextInput > div > div > input {
        background: rgba(0, 0, 0, 0.45) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
        box-shadow: 
            0 4px 16px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transition: all 0.3s ease !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.9) !important;
    }
    
    .stTextInput > div > div > input:focus {
        background: rgba(0, 0, 0, 0.55) !important;
        border-color: #64b5f6 !important;
        box-shadow: 
            0 0 0 2px rgba(100, 181, 246, 0.4), 
            0 6px 20px rgba(0, 0, 0, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
        outline: none !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.8) !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.7) !important;
    }
    
    /* Ensure text input labels are visible */
    .stTextInput > label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* TOKEN COUNTER - Using working flexbox design from test_glassmorphism_fixed.py */
    /* No complex CSS needed - using inline styles for reliability */
    
    /* Main content area styling */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: none !important;
    }
    
    /* Enhanced Glass container base class - 2025 Design */
    .glass-container {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border-radius: 18px !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        padding: 0.75rem !important;
        margin: 0.4rem 0 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
    }
    
    .glass-container::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        border-radius: 18px !important;
        padding: 1px !important;
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05)) !important;
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0) !important;
        mask-composite: exclude !important;
        -webkit-mask-composite: xor !important;
        pointer-events: none !important;
    }
    
    .glass-container:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 
            0 16px 48px rgba(0, 0, 0, 0.35),
            inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
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
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(16px) saturate(160%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(160%) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        padding: 0.75rem !important;
        text-align: center !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .status-card::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent) !important;
        pointer-events: none !important;
    }
    
    .status-card:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        transform: translateY(-1px) scale(1.02) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2) !important;
    }
    
    .status-label {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        color: rgba(255, 255, 255, 0.7) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 0.25rem !important;
    }
    
    .status-value {
        font-size: 1.1rem !important;
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
    
    /* Duplicate input styling removed - using optimized version above */
    
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
    
    /* Comprehensive white box removal and background override */
    .element-container {
        margin: 0 !important;
        background: transparent !important;
    }
    
    /* Ensure no white backgrounds appear */
    .stApp > div {
        background: transparent !important;
    }
    
    /* Remove extra spacing and white containers */
    .block-container > div {
        gap: 0.5rem !important;
        background: transparent !important;
    }
    
    /* Target all sidebar content containers */
    [data-testid="stSidebar"] .element-container,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stButton,
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stTextInput,
    [data-testid="stSidebar"] .stTextArea,
    [data-testid="stSidebar"] .stNumberInput,
    [data-testid="stSidebar"] .stDateInput,
    [data-testid="stSidebar"] .stTimeInput,
    [data-testid="stSidebar"] .stFileUploader,
    [data-testid="stSidebar"] .stColorPicker,
    [data-testid="stSidebar"] .stSlider,
    [data-testid="stSidebar"] .stRadio,
    [data-testid="stSidebar"] .stCheckbox,
    [data-testid="stSidebar"] .stMultiSelect,
    [data-testid="stSidebar"] .stSelectSlider,
    [data-testid="stSidebar"] .stDataFrame,
    [data-testid="stSidebar"] .stTable,
    [data-testid="stSidebar"] .stMetric {
        background: transparent !important;
        backdrop-filter: none !important;
    }
    
    /* Target sidebar user content specifically */
    [data-testid="stSidebarUserContent"] {
        background: transparent !important;
    }
    
    [data-testid="stSidebarUserContent"] > div,
    [data-testid="stSidebarUserContent"] .element-container,
    [data-testid="stSidebarUserContent"] .stMarkdown > div,
    [data-testid="stSidebarUserContent"] .block-container {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Selective white box removal - Target specific containers only */
    [data-testid="stSidebar"] .element-container:not(.status-card):not(.glass-container):not(.chat-bubble),
    [data-testid="stSidebar"] .stMarkdown > div:not(.status-card):not(.glass-container):not(.chat-bubble) {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Target specific Streamlit container classes that create white boxes */
    [data-testid="stSidebar"] .css-1d391kg,
    [data-testid="stSidebar"] .css-12w0qpk,
    [data-testid="stSidebar"] .css-17lntkn {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Exception: Keep our custom glass containers styled */
    [data-testid="stSidebar"] .glass-container,
    [data-testid="stSidebar"] .status-card,
    [data-testid="stSidebar"] .chat-bubble {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 14px !important;
    }
    
    /* Target specific problematic containers */
    [data-testid="stSidebar"] .stMetric > div {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Ensure proper text contrast for accessibility */
    [data-testid="stSidebar"] .status-label,
    [data-testid="stSidebar"] .status-value {
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5) !important;
    }
    
    /* HIDE sidebar toggle completely for fixed layout */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    button[data-testid="collapsedControl"],
    button[data-testid="stSidebarCollapseButton"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Hide the button container entirely */
    .css-1y4p8pa,
    .css-1vencpc {
        display: none !important;
    }
    
    /* TARGETED WHITE BOX FIXES - Very specific selectors */
    /* Force all sidebar divs to be transparent */
    [data-testid="stSidebar"] div:not(.status-card):not(.glass-container):not(.chat-bubble) {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Target the specific markdown containers */
    [data-testid="stSidebar"] .stMarkdown {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown > div {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Target element containers specifically */
    [data-testid="stSidebar"] .element-container {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Target any remaining white backgrounds with high specificity */
    [data-testid="stSidebar"] div[style*="background-color: rgb(255, 255, 255)"],
    [data-testid="stSidebar"] div[style*="background-color: white"],
    [data-testid="stSidebar"] div[style*="background: rgb(255, 255, 255)"],
    [data-testid="stSidebar"] div[style*="background: white"] {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* SPECIFIC FIXES for status grid and chat bubble containers */
    /* Target the parent containers of status-grid and chat bubbles */
    [data-testid="stSidebar"] .stMarkdown:has(.status-grid),
    [data-testid="stSidebar"] .stMarkdown:has(.status-grid) > div,
    [data-testid="stSidebar"] .element-container:has(.status-grid),
    [data-testid="stSidebar"] .element-container:has(.status-grid) > div {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Target containers around chat bubbles specifically */
    [data-testid="stSidebar"] .stMarkdown:has(.chat-bubble),
    [data-testid="stSidebar"] .stMarkdown:has(.chat-bubble) > div,
    [data-testid="stSidebar"] .element-container:has(.chat-bubble),
    [data-testid="stSidebar"] .element-container:has(.chat-bubble) > div {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Fallback: Target all nested divs in sidebar more aggressively */
    [data-testid="stSidebar"] > div > div,
    [data-testid="stSidebar"] > div > div > div,
    [data-testid="stSidebar"] > div > div > div > div {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* TOKEN COUNTER container styling - Grid system defined above */
    .token-counter {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        margin: 1rem 0 !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
        overflow: hidden !important;
    }
    
    /* NUCLEAR APPROACH - FORCE EVERY POSSIBLE ELEMENT TO BE TRANSPARENT */
    /* Maximum specificity override for EVERYTHING in sidebar */
    html body div[data-testid="stSidebar"] * {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    html body div[data-testid="stSidebar"] div {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    html body div[data-testid="stSidebar"] .stMarkdown {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    html body div[data-testid="stSidebar"] .stMarkdown div {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    html body div[data-testid="stSidebar"] .element-container {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    /* FORCE specific white box areas */
    html body div[data-testid="stSidebar"] [style*="background"] {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Exception: Keep only our glass elements */
    html body div[data-testid="stSidebar"] .status-card {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(16px) saturate(160%) !important;
    }
    
    html body div[data-testid="stSidebar"] .token-counter {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    html body div[data-testid="stSidebar"] .chat-bubble {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(12px) !important;
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
    
    # Create readable token counter using CSS classes instead of inline styles
    token_counter = f"""
    <div class="token-counter">
        <div class="token-counter-grid">
            <div class="token-counter-item">
                <div class="token-counter-label">Query</div>
                <div class="token-counter-value">{tokens['query_tokens']}</div>
            </div>
            <div class="token-counter-item">
                <div class="token-counter-label">Text RAG</div>
                <div class="token-counter-value">{tokens['text_rag_tokens']}</div>
            </div>
            <div class="token-counter-item">
                <div class="token-counter-label">VLM</div>
                <div class="token-counter-value">{tokens['vlm_analysis_tokens']}</div>
            </div>
            <div class="token-counter-item">
                <div class="token-counter-label">SF</div>
                <div class="token-counter-value">{tokens['salesforce_api_tokens']}</div>
            </div>
            <div class="token-counter-item">
                <div class="token-counter-label">Re-rank</div>
                <div class="token-counter-value">{tokens['reranker_tokens']}</div>
            </div>
            <div class="token-counter-item">
                <div class="token-counter-label">Response</div>
                <div class="token-counter-value">{tokens['response_tokens']}</div>
            </div>
            <div class="token-counter-item total">
                <div class="token-counter-label">Total</div>
                <div class="token-counter-value total">{tokens['total_tokens']}</div>
            </div>
        </div>
    </div>
    """
    
    return f"""
    <div class="chat-bubble">
        <div class="chat-query">🔍 {query}</div>
        <div class="chat-answer">{answer}</div>
        {token_counter}
        <div class="chat-meta">
            <span>{icon} {source_type.upper()} | {timestamp.strftime('%H:%M:%S')}</span>
        </div>
    </div>
    """

def main():
    # Load glassmorphism CSS
    load_glassmorphism_css()
    
    # Header with proper spacing
    st.markdown("""
    <div class="glass-header" style="margin-bottom: 2rem;">
        <h1>🤖 Smart Document Assistant</h1>
        <p>Glassmorphism UI Prototype with Collapsible Sidebar</p>
    </div>
    """, unsafe_allow_html=True)
    
    # TRUE GLASSMORPHIC SIDEBAR - Using stylable_container
    with st.sidebar:
        # System Configuration Header - Aligned with main header
        with stylable_container(
            key="sidebar_header",
            css_styles="""
            {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(20px) saturate(180%);
                -webkit-backdrop-filter: blur(20px) saturate(180%);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                padding: 1.5rem 1rem;
                margin-top: 0rem;
                margin-bottom: 1.5rem;
                box-shadow: 
                    0 8px 32px rgba(0, 0, 0, 0.15),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
            }
            """
        ):
            st.markdown('<h3 style="color: #ffffff; font-size: 1.2rem; margin: 0; text-align: center; text-shadow: 0 2px 10px rgba(0,0,0,0.3);">🎛️ System Configuration</h3>', unsafe_allow_html=True)
        
        # Feature Status Glass Panel
        with stylable_container(
            key="feature_status_panel",
            css_styles="""
            {
                background: rgba(255, 255, 255, 0.03);
                backdrop-filter: blur(25px) saturate(200%);
                -webkit-backdrop-filter: blur(25px) saturate(200%);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 18px;
                padding: 1.5rem 1rem;
                margin-bottom: 1.5rem;
                box-shadow: 
                    0 12px 40px rgba(0, 0, 0, 0.2),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
                position: relative;
            }
            """
        ):
            st.markdown('<h4 style="color: #81c784; font-size: 1rem; margin-bottom: 1rem; text-shadow: 0 1px 3px rgba(0,0,0,0.5);">📋 Feature Status</h4>', unsafe_allow_html=True)
            
            # Clean Vertical Status Cards Layout
            status_cards = [
                ("Text RAG", "✅ Ready", "#81c784", "rgba(129, 199, 132, 0.1)", "rgba(129, 199, 132, 0.2)"),
                ("ColPali", "✅ GPU", "#64b5f6", "rgba(100, 181, 246, 0.1)", "rgba(100, 181, 246, 0.2)"),
                ("Salesforce", "⚠️ Config", "#ffb74d", "rgba(255, 183, 77, 0.1)", "rgba(255, 183, 77, 0.2)"),
                ("Re-rank", "✅ BGE", "#81c784", "rgba(129, 199, 132, 0.1)", "rgba(129, 199, 132, 0.2)"),
                ("GPU Mode", "⚡ Fast", "#64b5f6", "rgba(100, 181, 246, 0.1)", "rgba(100, 181, 246, 0.2)"),
                ("Queries", str(len(st.session_state.chat_history)), "#64b5f6", "rgba(100, 181, 246, 0.1)", "rgba(100, 181, 246, 0.2)")
            ]
            
            for label, value, color, bg_color, border_color in status_cards:
                # Complete HTML control for perfect centering
                status_card_html = f"""
                <div style="
                    background: {bg_color};
                    backdrop-filter: blur(15px);
                    border: 1px solid {border_color};
                    border-radius: 12px;
                    margin-bottom: 0.5rem;
                    text-align: center;
                    height: 60px;
                    position: relative;
                    transition: all 0.3s ease;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                ">
                    <div style="
                        color: rgba(255,255,255,0.7); 
                        font-size: 0.75rem; 
                        text-transform: uppercase; 
                        line-height: 1;
                        margin-bottom: 4px;
                    ">{label}</div>
                    <div style="
                        color: {color}; 
                        font-weight: 600; 
                        font-size: 1rem;
                        line-height: 1;
                    ">{value}</div>
                </div>
                """
                st.markdown(status_card_html, unsafe_allow_html=True)
        
        # System Status Glass Panel
        with stylable_container(
            key="system_status_panel",
            css_styles="""
            {
                background: rgba(255, 255, 255, 0.04);
                backdrop-filter: blur(22px) saturate(170%);
                -webkit-backdrop-filter: blur(22px) saturate(170%);
                border: 1px solid rgba(255, 183, 77, 0.15);
                border-radius: 16px;
                padding: 1.5rem 1rem;
                margin-bottom: 1.5rem;
                box-shadow: 
                    0 10px 35px rgba(255, 183, 77, 0.1),
                    inset 0 1px 0 rgba(255, 183, 77, 0.1);
            }
            """
        ):
            st.markdown('<h4 style="color: #ffb74d; font-size: 1rem; margin-bottom: 1rem; text-shadow: 0 1px 3px rgba(0,0,0,0.5);">📊 System Status</h4>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding: 0.5rem 0;">
                <span style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">Mode:</span>
                <span style="color: #64b5f6; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">GPU</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding: 0.5rem 0;">
                <span style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">Sources:</span>
                <span style="color: #64b5f6; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">3 Active</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 0.5rem 0;">
                <span style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">Status:</span>
                <span style="color: #81c784; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">Ready</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Clear History Button
        with stylable_container(
            key="clear_button",
            css_styles="""
            {
                background: rgba(244, 67, 54, 0.1);
                backdrop-filter: blur(15px);
                border: 1px solid rgba(244, 67, 54, 0.2);
                border-radius: 12px;
                padding: 0.75rem;
                transition: all 0.3s ease;
            }
            """
        ):
            if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
    
    # Main content area - Aligned with sidebar panels
    col1, col2 = st.columns([2.5, 1.2])
    
    with col1:
        # Query Interface Glass Panel - Aligned with sidebar
        with stylable_container(
            key="query_interface_panel",
            css_styles="""
            {
                background: rgba(255, 255, 255, 0.06);
                backdrop-filter: blur(20px) saturate(180%);
                -webkit-backdrop-filter: blur(20px) saturate(180%);
                border: 1px solid rgba(100, 181, 246, 0.2);
                border-radius: 18px;
                padding: 1.5rem;
                margin-top: 0rem;
                margin-bottom: 1.5rem;
                box-shadow: 
                    0 12px 40px rgba(100, 181, 246, 0.1),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
            }
            """
        ):
            st.markdown('<h3 style="color: #64b5f6; margin-bottom: 1.5rem; font-size: 1.2rem; text-align: center; text-shadow: 0 2px 8px rgba(0,0,0,0.3);">💬 Query Interface</h3>', unsafe_allow_html=True)
            
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
                    st.rerun()
        
        # Note: Token counter will appear in each query result below
        
        # Recent Results Glass Panel
        if st.session_state.chat_history:
            with stylable_container(
                key="results_header_panel",
                css_styles="""
                {
                    background: rgba(255, 255, 255, 0.04);
                    backdrop-filter: blur(18px) saturate(160%);
                    -webkit-backdrop-filter: blur(18px) saturate(160%);
                    border: 1px solid rgba(100, 181, 246, 0.15);
                    border-radius: 16px;
                    padding: 1rem 1.5rem;
                    margin-bottom: 1.5rem;
                    box-shadow: 
                        0 8px 30px rgba(100, 181, 246, 0.08),
                        inset 0 1px 0 rgba(255, 255, 255, 0.08);
                }
                """
            ):
                st.markdown('<h3 style="color: #64b5f6; margin: 0; font-size: 1.2rem; text-align: center; text-shadow: 0 2px 8px rgba(0,0,0,0.3);">📝 Recent Results</h3>', unsafe_allow_html=True)
            
            for i, chat in enumerate(st.session_state.chat_history[:3]):  # Show last 3 chats
                # Create individual chat bubble using stylable_container
                with stylable_container(
                    key=f"chat_bubble_{i}",
                    css_styles="""
                    {
                        background: rgba(255, 255, 255, 0.05);
                        backdrop-filter: blur(15px) saturate(150%);
                        -webkit-backdrop-filter: blur(15px) saturate(150%);
                        border: 1px solid rgba(255, 255, 255, 0.12);
                        border-radius: 16px;
                        padding: 1.5rem;
                        margin-bottom: 1rem;
                        box-shadow: 
                            0 6px 25px rgba(0, 0, 0, 0.1),
                            inset 0 1px 0 rgba(255, 255, 255, 0.08);
                        transition: all 0.3s ease;
                    }
                    """
                ):
                    # Query
                    st.markdown(f'<div style="color: #81c784; font-weight: 500; font-size: 0.95rem; margin-bottom: 0.75rem; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">🔍 {chat["query"]}</div>', unsafe_allow_html=True)
                    
                    # Answer
                    st.markdown(f'<div style="color: rgba(255, 255, 255, 0.9); font-size: 0.9rem; line-height: 1.6; margin-bottom: 1rem;">{chat["answer"]}</div>', unsafe_allow_html=True)
                    
                    # Revolutionary CSS Grid Token Counter - Handles 4-digit values and responsive design
                    with stylable_container(
                        key=f"token_counter_{i}",
                        css_styles="""
                        {
                            background: rgba(255, 255, 255, 0.08);
                            backdrop-filter: blur(12px);
                            -webkit-backdrop-filter: blur(12px);
                            border: 1px solid rgba(255, 255, 255, 0.15);
                            border-radius: 12px;
                            margin: 1rem 0;
                            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                            overflow: hidden;
                        }
                        """
                    ):
                        tokens = chat['token_breakdown']
                        
                        # Perfect Token Counter - Line-height based centering
                        perfect_token_counter = f"""
                        <div style="padding: 8px 12px; margin: 8px 0 4px 0;">
                            <div style="display: flex; justify-content: space-between; align-items: center; height: 40px;">
                                <div style="text-align: center; flex: 1;">
                                    <div style="color: rgba(255, 255, 255, 0.6); font-size: 11px; text-transform: uppercase; line-height: 1; margin: 0;">Query</div>
                                    <div style="color: #64b5f6; font-weight: 600; font-size: 13px; line-height: 1.8; margin: 0;">{tokens['query_tokens']}</div>
                                </div>
                                <div style="text-align: center; flex: 1;">
                                    <div style="color: rgba(255, 255, 255, 0.6); font-size: 11px; text-transform: uppercase; line-height: 1; margin: 0;">Text RAG</div>
                                    <div style="color: #64b5f6; font-weight: 600; font-size: 13px; line-height: 1.8; margin: 0;">{tokens['text_rag_tokens']}</div>
                                </div>
                                <div style="text-align: center; flex: 1;">
                                    <div style="color: rgba(255, 255, 255, 0.6); font-size: 11px; text-transform: uppercase; line-height: 1; margin: 0;">VLM</div>
                                    <div style="color: #64b5f6; font-weight: 600; font-size: 13px; line-height: 1.8; margin: 0;">{tokens['vlm_analysis_tokens']}</div>
                                </div>
                                <div style="text-align: center; flex: 1;">
                                    <div style="color: rgba(255, 255, 255, 0.6); font-size: 11px; text-transform: uppercase; line-height: 1; margin: 0;">SF</div>
                                    <div style="color: #64b5f6; font-weight: 600; font-size: 13px; line-height: 1.8; margin: 0;">{tokens['salesforce_api_tokens']}</div>
                                </div>
                                <div style="text-align: center; flex: 1;">
                                    <div style="color: rgba(255, 255, 255, 0.6); font-size: 11px; text-transform: uppercase; line-height: 1; margin: 0;">Re-rank</div>
                                    <div style="color: #64b5f6; font-weight: 600; font-size: 13px; line-height: 1.8; margin: 0;">{tokens['reranker_tokens']}</div>
                                </div>
                                <div style="text-align: center; flex: 1;">
                                    <div style="color: rgba(255, 255, 255, 0.6); font-size: 11px; text-transform: uppercase; line-height: 1; margin: 0;">Response</div>
                                    <div style="color: #64b5f6; font-weight: 600; font-size: 13px; line-height: 1.8; margin: 0;">{tokens['response_tokens']}</div>
                                </div>
                                <div style="text-align: center; flex: 1; border-left: 1px solid rgba(255, 255, 255, 0.2); padding-left: 8px;">
                                    <div style="color: rgba(255, 255, 255, 0.6); font-size: 11px; text-transform: uppercase; line-height: 1; margin: 0;">Total</div>
                                    <div style="color: #81c784; font-weight: 600; font-size: 13px; line-height: 1.8; margin: 0;">{tokens['total_tokens']}</div>
                                </div>
                            </div>
                        </div>
                        """
                        
                        st.markdown(perfect_token_counter, unsafe_allow_html=True)
                    
                    # Source and timestamp
                    source_icons = {'text': '📝', 'visual': '🖼️', 'salesforce': '🏢'}
                    icon = source_icons.get(chat['source_type'], '📝')
                    st.markdown(f'<div style="color: rgba(255, 255, 255, 0.6); font-size: 0.75rem; text-align: center; margin-top: 0.75rem;">{icon} {chat["source_type"].upper()} | {chat["timestamp"].strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
    
    with col2:
        # Document Management Glass Panel - Aligned with Query Interface
        with stylable_container(
            key="document_management_panel",
            css_styles="""
            {
                background: rgba(255, 255, 255, 0.06);
                backdrop-filter: blur(20px) saturate(180%);
                -webkit-backdrop-filter: blur(20px) saturate(180%);
                border: 1px solid rgba(100, 181, 246, 0.2);
                border-radius: 18px;
                padding: 1.5rem;
                margin-top: 0rem;
                margin-bottom: 1.5rem;
                box-shadow: 
                    0 12px 40px rgba(100, 181, 246, 0.1),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
            }
            """
        ):
            st.markdown('<h3 style="color: #64b5f6; margin-bottom: 1.5rem; font-size: 1.2rem; text-align: center; text-shadow: 0 2px 8px rgba(0,0,0,0.3);">📁 Document Management</h3>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.05); border: 2px dashed rgba(100, 181, 246, 0.3); border-radius: 12px; padding: 2rem 1rem; text-align: center; margin: 1rem 0;">
                <div style="color: #64b5f6; font-size: 2rem; margin-bottom: 0.5rem;">📤</div>
                <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem;">Drop files here or click to upload</div>
                <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.75rem; margin-top: 0.5rem;">PDF, DOCX, TXT, CSV supported</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Processing Status Glass Panel - Aligned with Query Interface
        with stylable_container(
            key="processing_status_panel",
            css_styles="""
            {
                background: rgba(255, 255, 255, 0.04);
                backdrop-filter: blur(18px) saturate(160%);
                -webkit-backdrop-filter: blur(18px) saturate(160%);
                border: 1px solid rgba(100, 181, 246, 0.15);
                border-radius: 18px;
                padding: 1.5rem;
                margin-top: 0rem;
                margin-bottom: 1.5rem;
                box-shadow: 
                    0 8px 30px rgba(100, 181, 246, 0.08),
                    inset 0 1px 0 rgba(255, 255, 255, 0.08);
            }
            """
        ):
            st.markdown('<h3 style="color: #64b5f6; margin-bottom: 1.5rem; font-size: 1.2rem; text-align: center; text-shadow: 0 2px 8px rgba(0,0,0,0.3);">⚡ Processing Status</h3>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="margin: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding: 0.5rem 0;">
                    <span style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">Documents:</span>
                    <span style="color: #81c784; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">5 processed</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding: 0.5rem 0;">
                    <span style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">Text Chunks:</span>
                    <span style="color: #64b5f6; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">1,247</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding: 0.5rem 0;">
                    <span style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">Visual Pages:</span>
                    <span style="color: #64b5f6; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">23</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0;">
                    <span style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;">Last Updated:</span>
                    <span style="color: rgba(255, 255, 255, 0.9); font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">2 min ago</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer info
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(255, 255, 255, 0.05); border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.1);">
        <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.8rem;">🎨 True Glassmorphism v9.0 | stylable_container + Vibrant Backgrounds + Real Backdrop Filters</div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()