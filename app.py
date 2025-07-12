import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from components import _page1, _page2, _page3, _page4
from scripts.model import MaterialPredictionModel

st.set_page_config(
    page_title="Material Requirement Analysis",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"  # Kembali ke expanded untuk menampilkan sidebar
)

# CSS untuk styling tanpa menyembunyikan sidebar
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">🔧 Material Requirement Analysis System</div>', unsafe_allow_html=True)
    
    # Sidebar dengan dropdown navigation saja
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Data Upload & Preprocessing", "Exploratory Data Analysis", "Model Training & Prediction", "Results & Export"],
        key="page_selector"
    )
    
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None
    if 'model' not in st.session_state:
        st.session_state.model = MaterialPredictionModel()
    
    if page == "Data Upload & Preprocessing":
        _page1.show_page()
    elif page == "Exploratory Data Analysis":
        _page2.show_page()
    elif page == "Model Training & Prediction":
        _page3.show_page()
    elif page == "Results & Export":
        _page4.show_page()
    
    st.markdown("---")
    st.markdown("**Material Requirement Analysis System** - Built with Streamlit")

if __name__ == "__main__":
    main()