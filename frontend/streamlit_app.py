"""Streamlit demo frontend for the triage API.

Non-graded extra (see docs/architecture.md) -- calls the FastAPI
/predict endpoint over HTTP, doesn't duplicate any model/urgency logic.
Purely for a friendlier way to exercise the API than curl/Swagger, and a
convenient way to generate real traffic for the Prometheus/Grafana panels.
"""

import os

import requests
import streamlit as st

# st.secrets is how Streamlit Community Cloud configures this (its own
# TOML-based settings panel, not a plain OS env var); os.environ covers
# local runs and docker-compose. Checked in that order since st.secrets
# raises if no secrets.toml exists at all, rather than just being empty.
try:
    API_URL = st.secrets.get("API_URL", os.environ.get("API_URL", "http://localhost:8000"))
except FileNotFoundError:
    API_URL = os.environ.get("API_URL", "http://localhost:8000")

URGENCY_DISPLAY = {
    "urgent": ("🔴", "URGENT"),
    "attention": ("🟡", "ATTENTION"),
    "normal": ("🟢", "NORMAL"),
}

EXAMPLE_REPORT = (
    "The patient presented with acute chest pain, elevated troponin levels, "
    "and ST-segment elevation on ECG consistent with myocardial infarction."
)

st.set_page_config(page_title="MedSys Triage", page_icon="🏥")
st.title("🏥 MedSys — Hospital Laudo Triage")
st.caption(
    "Paste a medical report (laudo médico) below to classify its category "
    "and urgency tier. Calls the live FastAPI /predict endpoint."
)

if "report_text" not in st.session_state:
    st.session_state.report_text = ""

if st.button("Use example report"):
    st.session_state.report_text = EXAMPLE_REPORT

text = st.text_area(
    "Medical report text",
    height=200,
    placeholder="Enter the laudo médico here...",
    key="report_text",
)

if st.button("Classify", type="primary", disabled=not text.strip()):
    with st.spinner("Classifying..."):
        try:
            response = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=10)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as e:
            st.error(f"Request to {API_URL} failed: {e}")
        else:
            icon, label = URGENCY_DISPLAY.get(result["urgency"], ("", result["urgency"]))
            st.markdown(f"### {icon} Urgency: **{label}**")
            st.markdown(f"**Predicted category:** {result['category']}")

st.divider()
st.caption(f"API: `{API_URL}`")
