"""Shared theme helpers."""
import streamlit as st


def apply_sidebar_style():
    """Inject CSS to make sidebar text white against the dark-blue background."""
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { background-color: #003087 !important; }
        [data-testid="stSidebar"] * { color: white !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
