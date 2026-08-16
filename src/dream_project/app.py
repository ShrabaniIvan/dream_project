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
    A statistical analysis platform prototyped for agricultural research.
    Upload your own CSV or Excel dataset in any module — no data is stored between sessions.

    **Modules (use the sidebar to navigate)**

    | Module | What it does | Typical question |
    |---|---|---|
    | **Descriptive Statistics** | Summary statistics + distribution charts for a single numeric column | What is the spread and shape of my yield data? |
    | **Correlation Analysis** | Pairwise association tests across any mix of continuous, ordinal, and nominal variables | Is irrigation type associated with crop yield? |

    **Recommended workflow**

    1. Start with **Descriptive Statistics** to understand each variable individually — check for skew, outliers, and data quality issues.
    2. Move to **Correlation Analysis** to test relationships between pairs of variables.
    3. Use the **Generate AI Explanation** button in each module to get a plain-language interpretation of the results.

    **Accepted data formats:** `.csv` and `.xlsx`. The first row must be a header row.
    """
)
