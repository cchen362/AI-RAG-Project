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
    
    /* TOKEN COUNTER specific styling to replace inline styles */
    .token-counter {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        padding: 10px 14px !important;
        margin: 10px 0 6px 0 !important;
    }
    
    .token-counter-grid {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        font-size: 12px !important;
    }
    
    .token-counter-item {
        text-align: center !important;
        flex: 1 !important;
    }
    
    .token-counter-item.total {
        border-left: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding-left: 8px !important;
    }
    
    .token-counter-label {
        color: rgba(255, 255, 255, 0.6) !important;
        font-size: 10px !important;
        text-transform: uppercase !important;
        margin-bottom: 2px !important;
    }
    
    .token-counter-value {
        color: #64b5f6 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }
    
    .token-counter-value.total {
        color: #81c784 !important;
        font-size: 14px !important;
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
    
    # Header
    st.markdown("""
    <div class="glass-header">
        <h1>🤖 Smart Document Assistant</h1>
        <p>Glassmorphism UI Prototype with Collapsible Sidebar</p>
    </div>
    """, unsafe_allow_html=True)
    
    # TRUE GLASSMORPHIC SIDEBAR - Using stylable_container
    with st.sidebar:
        # System Configuration Header
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
            
            # Create individual status cards using stylable containers
            col1, col2 = st.columns(2)
            
            with col1:
                with stylable_container(
                    key="text_rag_status",
                    css_styles="""
                    {
                        background: rgba(129, 199, 132, 0.1);
                        backdrop-filter: blur(15px);
                        border: 1px solid rgba(129, 199, 132, 0.2);
                        border-radius: 12px;
                        padding: 0.75rem;
                        margin-bottom: 0.5rem;
                        text-align: center;
                        transition: all 0.3s ease;
                    }
                    {
                        background: rgba(129, 199, 132, 0.15);
                        transform: translateY(-2px);
                        box-shadow: 0 8px 25px rgba(129, 199, 132, 0.2);
                    }
                    """
                ):
                    st.markdown('<div style="color: rgba(255,255,255,0.7); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 0.25rem;">Text RAG</div>', unsafe_allow_html=True)
                    st.markdown('<div style="color: #81c784; font-weight: 600; font-size: 1.1rem;">✅ Ready</div>', unsafe_allow_html=True)
                
                with stylable_container(
                    key="salesforce_status", 
                    css_styles="""
                    {
                        background: rgba(255, 183, 77, 0.1);
                        backdrop-filter: blur(15px);
                        border: 1px solid rgba(255, 183, 77, 0.2);
                        border-radius: 12px;
                        padding: 0.75rem;
                        margin-bottom: 0.5rem;
                        text-align: center;
                        transition: all 0.3s ease;
                    }
                    """
                ):
                    st.markdown('<div style="color: rgba(255,255,255,0.7); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 0.25rem;">Salesforce</div>', unsafe_allow_html=True)
                    st.markdown('<div style="color: #ffb74d; font-weight: 600; font-size: 1.1rem;">⚠️ Config</div>', unsafe_allow_html=True)
                
                with stylable_container(
                    key="gpu_mode_status",
                    css_styles="""
                    {
                        background: rgba(100, 181, 246, 0.1);
                        backdrop-filter: blur(15px);
                        border: 1px solid rgba(100, 181, 246, 0.2);
                        border-radius: 12px;
                        padding: 0.75rem;
                        text-align: center;
                        transition: all 0.3s ease;
                    }
                    """
                ):
                    st.markdown('<div style="color: rgba(255,255,255,0.7); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 0.25rem;">GPU Mode</div>', unsafe_allow_html=True)
                    st.markdown('<div style="color: #64b5f6; font-weight: 600; font-size: 1.1rem;">⚡ Fast</div>', unsafe_allow_html=True)
            
            with col2:
                with stylable_container(
                    key="colpali_status",
                    css_styles="""
                    {
                        background: rgba(100, 181, 246, 0.1);
                        backdrop-filter: blur(15px);
                        border: 1px solid rgba(100, 181, 246, 0.2);
                        border-radius: 12px;
                        padding: 0.75rem;
                        margin-bottom: 0.5rem;
                        text-align: center;
                        transition: all 0.3s ease;
                    }
                    """
                ):
                    st.markdown('<div style="color: rgba(255,255,255,0.7); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 0.25rem;">ColPali</div>', unsafe_allow_html=True)
                    st.markdown('<div style="color: #64b5f6; font-weight: 600; font-size: 1.1rem;">✅ GPU</div>', unsafe_allow_html=True)
                
                with stylable_container(
                    key="reranker_status",
                    css_styles="""
                    {
                        background: rgba(129, 199, 132, 0.1);
                        backdrop-filter: blur(15px);
                        border: 1px solid rgba(129, 199, 132, 0.2);
                        border-radius: 12px;
                        padding: 0.75rem;
                        margin-bottom: 0.5rem;
                        text-align: center;
                        transition: all 0.3s ease;
                    }
                    """
                ):
                    st.markdown('<div style="color: rgba(255,255,255,0.7); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 0.25rem;">Re-ranker</div>', unsafe_allow_html=True)
                    st.markdown('<div style="color: #81c784; font-weight: 600; font-size: 1.1rem;">✅ BGE</div>', unsafe_allow_html=True)
                
                with stylable_container(
                    key="queries_status",
                    css_styles="""
                    {
                        background: rgba(100, 181, 246, 0.1);
                        backdrop-filter: blur(15px);
                        border: 1px solid rgba(100, 181, 246, 0.2);
                        border-radius: 12px;
                        padding: 0.75rem;
                        text-align: center;
                        transition: all 0.3s ease;
                    }
                    """
                ):
                    st.markdown('<div style="color: rgba(255,255,255,0.7); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 0.25rem;">Queries</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="color: #64b5f6; font-weight: 600; font-size: 1.1rem;">{len(st.session_state.chat_history)}</div>', unsafe_allow_html=True)
        
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
    
    # Main content area
    col1, col2 = st.columns([2.5, 1.2])
    
    with col1:
        # Query Interface Glass Panel
        with stylable_container(
            key="query_interface_panel",
            css_styles="""
            {
                background: rgba(255, 255, 255, 0.06);
                backdrop-filter: blur(20px) saturate(180%);
                -webkit-backdrop-filter: blur(20px) saturate(180%);
                border: 1px solid rgba(100, 181, 246, 0.2);
                border-radius: 18px;
                padding: 2rem 1.5rem;
                margin-bottom: 2rem;
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
                    
                    # Token Counter using stylable_container
                    with stylable_container(
                        key=f"token_counter_{i}",
                        css_styles="""
                        {
                            background: rgba(255, 255, 255, 0.08);
                            backdrop-filter: blur(12px);
                            -webkit-backdrop-filter: blur(12px);
                            border: 1px solid rgba(255, 255, 255, 0.15);
                            border-radius: 12px;
                            padding: 1rem;
                            margin: 1rem 0;
                            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                        }
                        """
                    ):
                        # Create token display grid
                        token_cols = st.columns(7)
                        tokens = chat['token_breakdown']
                        
                        with token_cols[0]:
                            st.markdown('<div style="text-align: center;"><div style="color: rgba(255,255,255,0.6); font-size: 0.7rem; text-transform: uppercase; margin-bottom: 0.25rem;">Query</div><div style="color: #64b5f6; font-weight: 600; font-size: 0.9rem;">{}</div></div>'.format(tokens['query_tokens']), unsafe_allow_html=True)
                        
                        with token_cols[1]:
                            st.markdown('<div style="text-align: center;"><div style="color: rgba(255,255,255,0.6); font-size: 0.7rem; text-transform: uppercase; margin-bottom: 0.25rem;">Text RAG</div><div style="color: #64b5f6; font-weight: 600; font-size: 0.9rem;">{}</div></div>'.format(tokens['text_rag_tokens']), unsafe_allow_html=True)
                        
                        with token_cols[2]:
                            st.markdown('<div style="text-align: center;"><div style="color: rgba(255,255,255,0.6); font-size: 0.7rem; text-transform: uppercase; margin-bottom: 0.25rem;">VLM</div><div style="color: #64b5f6; font-weight: 600; font-size: 0.9rem;">{}</div></div>'.format(tokens['vlm_analysis_tokens']), unsafe_allow_html=True)
                        
                        with token_cols[3]:
                            st.markdown('<div style="text-align: center;"><div style="color: rgba(255,255,255,0.6); font-size: 0.7rem; text-transform: uppercase; margin-bottom: 0.25rem;">SF</div><div style="color: #64b5f6; font-weight: 600; font-size: 0.9rem;">{}</div></div>'.format(tokens['salesforce_api_tokens']), unsafe_allow_html=True)
                        
                        with token_cols[4]:
                            st.markdown('<div style="text-align: center;"><div style="color: rgba(255,255,255,0.6); font-size: 0.7rem; text-transform: uppercase; margin-bottom: 0.25rem;">Re-rank</div><div style="color: #64b5f6; font-weight: 600; font-size: 0.9rem;">{}</div></div>'.format(tokens['reranker_tokens']), unsafe_allow_html=True)
                        
                        with token_cols[5]:
                            st.markdown('<div style="text-align: center;"><div style="color: rgba(255,255,255,0.6); font-size: 0.7rem; text-transform: uppercase; margin-bottom: 0.25rem;">Response</div><div style="color: #64b5f6; font-weight: 600; font-size: 0.9rem;">{}</div></div>'.format(tokens['response_tokens']), unsafe_allow_html=True)
                        
                        with token_cols[6]:
                            st.markdown('<div style="text-align: center; border-left: 1px solid rgba(255,255,255,0.2); padding-left: 0.5rem;"><div style="color: rgba(255,255,255,0.6); font-size: 0.7rem; text-transform: uppercase; margin-bottom: 0.25rem;">Total</div><div style="color: #81c784; font-weight: 600; font-size: 1rem;">{}</div></div>'.format(tokens['total_tokens']), unsafe_allow_html=True)
                    
                    # Source and timestamp
                    source_icons = {'text': '📝', 'visual': '🖼️', 'salesforce': '🏢'}
                    icon = source_icons.get(chat['source_type'], '📝')
                    st.markdown(f'<div style="color: rgba(255, 255, 255, 0.6); font-size: 0.75rem; text-align: center; margin-top: 0.75rem;">{icon} {chat["source_type"].upper()} | {chat["timestamp"].strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
    
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
        <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.8rem;">🎨 True Glassmorphism v9.0 | stylable_container + Vibrant Backgrounds + Real Backdrop Filters</div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()