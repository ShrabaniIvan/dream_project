"""Agricultural Statistics Dashboard — entry point."""
import streamlit as st
from dream_project.modules.theme import apply_sidebar_style

st.set_page_config(
    page_title="Agricultural Statistics Dashboard",
    page_icon="🌾",
    layout="wide",
)

apply_sidebar_style()
st.title("Agricultural Statistics Dashboard")
st.markdown(
    """
    Welcome to the Agricultural Statistics Dashboard — a domain-agnostic statistical
    analysis platform prototyped for agricultural research.

    Use the sidebar to navigate to an analysis module.
    """
)
