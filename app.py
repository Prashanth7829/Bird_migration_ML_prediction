"""Streamlit entry point for the Bird Migration Success Prediction dashboard."""

import streamlit as st

st.set_page_config(
    page_title="Bird Migration Success Prediction",
    page_icon="🦅",
    layout="wide",
)

st.session_state.setdefault("selected_model_key", None)
st.session_state.setdefault("single_prediction", None)

page = st.navigation(
    {
        "Workspace": [
            st.Page(
                "app_pages/home.py",
                title="Home",
                icon=":material/home:",
            ),
            st.Page(
                "app_pages/predict_compare.py",
                title="Predict & Compare",
                icon=":material/model_training:",
            ),
            st.Page(
                "app_pages/model_performance.py",
                title="Model Performance",
                icon=":material/monitoring:",
            ),
            st.Page(
                "app_pages/about.py",
                title="About Project",
                icon=":material/info:",
            ),
        ]
    },
    position="sidebar",
)

page.run()