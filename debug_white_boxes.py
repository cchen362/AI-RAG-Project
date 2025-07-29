import streamlit as st

st.set_page_config(page_title="Debug White Boxes", layout="wide")

# Simple CSS to see the problem
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
}

/* Try to hide white containers */
.stApp * {
    background: transparent !important;
}

div[data-testid="element-container"] {
    background: red !important;
}

.stMarkdown {
    background: blue !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Debug: White Box Investigation")

# Test 1: Simple HTML
st.markdown("## Test 1: Simple HTML")
st.markdown("""
<div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 12px;">
    <p style="color: white;">This is a test div</p>
</div>
""", unsafe_allow_html=True)

# Test 2: Sidebar content
with st.sidebar:
    st.markdown("### Sidebar Test")
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem;">
        <p style="color: white;">Sidebar content</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("Look at the browser's developer tools (F12) to see what HTML containers Streamlit is creating around our custom HTML.")