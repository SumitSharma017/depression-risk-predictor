import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Depression Risk Predictor",page_icon="🧠",layout="wide",initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; }
    .subtitle { color: #6c757d; text-align: center; margin-bottom: 2rem; font-size: 1rem; }

    .risk-low {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 6px solid #28a745;
        padding: 1.2rem 1.5rem; border-radius: 10px;
        margin: 1rem 0;
    }
    .risk-moderate {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border-left: 6px solid #ffc107;
        padding: 1.2rem 1.5rem; border-radius: 10px;
        margin: 1rem 0;
    }
    .risk-high {
        background: linear-gradient(135deg, #f8d7da, #f5c2c7);
        border-left: 6px solid #dc3545;
        padding: 1.2rem 1.5rem; border-radius: 10px;
        margin: 1rem 0;
    }
    .risk-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.3rem; }
    .risk-pct   { font-size: 2.5rem; font-weight: 800; }
    .risk-desc  { font-size: 0.95rem; color: #495057; margin-top: 0.4rem; }

    .metric-box {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem; text-align: center;
        border: 1px solid #dee2e6;
    }
    .section-header {
        font-size: 1.1rem; font-weight: 600;
        color: #343a40; margin-top: 0.5rem;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.3rem; margin-bottom: 0.8rem;
    }
    .disclaimer {
        background: #e9ecef; border-radius: 8px;
        padding: 0.8rem 1rem; font-size: 0.82rem;
        color: #6c757d; margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Training model on your data…")
def load_model():
    model    = joblib.load('depression_model.pkl')
    with open('label_encodings.json') as f:
        encodings = json.load(f)
    with open('feature_columns.json') as f:
        feature_cols = json.load(f)
    return model, encodings, feature_cols

try:
    model, encodings, feature_cols = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


def get_risk_info(prob: float):
    pct = prob * 100
    if pct < 30:
        return "Low Risk", f"{pct:.1f}%", "risk-low", "🟢", \
"No significant indicators of depression detected. Keep maintaining healthy habits."
    elif pct < 70:
        return "Moderate Risk", f"{pct:.1f}%", "risk-moderate", "🟡", \
"Some risk factors are present. Consider speaking with a counsellor or trusted person."
    else:
        return "High Risk", f"{pct:.1f}%", "risk-high", "🔴", \
"Multiple risk factors detected. Please reach out to a mental health professional."

st.markdown("""
<div class="main-header">
    <h1>🧠 Depression Risk Predictor</h1>
</div>
<p class="subtitle">Fill in the details below to receive a personalised depression risk assessment based on a machine-learning model.</p>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ **Model files not found.** Please run the notebook first to generate `depression_model.pkl`, `label_encodings.json`, and `feature_columns.json`.")
    st.stop()

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.markdown("### About")
    st.markdown("""
This tool uses a **Gradient Boosting** classifier trained on survey data to estimate
depression risk on a continuous 0-100% scale, bucketed into three bands:

| Band | Range | Meaning |
|------|-------|---------|
| 🟢 Low | 0–30% | Low concern |
| 🟡 Moderate | 30–70% | Some factors present |
| 🔴 High | 70–100% | Professional help advised |
""")
    st.markdown("---")
    st.markdown("**Note:** This is a screening tool only. It does not replace professional medical advice.")

# ── Input Form ───────────────────────────────────────────────────────────────────
st.markdown("### Enter Your Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<p class="section-header">Demographics</p>', unsafe_allow_html=True)
    gender    = st.selectbox("Gender", options=list(encodings.get('Gender', {'Male':0,'Female':1}).keys()))
    age       = st.slider("Age", min_value=16, max_value=60, value=22, step=1)
    city_opts = list(encodings.get('City', {}).keys()) or ["Mumbai","Delhi","Bangalore","Chennai","Other"]
    city      = st.selectbox("City", options=city_opts)
    prof_opts = list(encodings.get('Profession', {}).keys()) or ["Student","Engineer","Doctor","Teacher","Other"]
    profession = st.selectbox("Profession", options=prof_opts)
    deg_opts  = list(encodings.get('Degree', {}).keys()) or ["B.Tech","M.Tech","MBA","MBBS","Other"]
    degree    = st.selectbox("Degree", options=deg_opts)

with col2:
    st.markdown('<p class="section-header">Academic & Work</p>', unsafe_allow_html=True)
    cgpa              = st.slider("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    academic_pressure = st.slider("Academic Pressure (1–5)", min_value=1, max_value=5, value=3)
    work_pressure     = st.slider("Work Pressure (1–5)",     min_value=1, max_value=5, value=3)
    study_satisfaction = st.slider("Study Satisfaction (1–5)", min_value=1, max_value=5, value=3)
    job_satisfaction  = st.slider("Job Satisfaction (1–5)",  min_value=1, max_value=5, value=3)
    work_study_hours  = st.slider("Work/Study Hours per Day", min_value=0, max_value=16, value=8)

with col3:
    st.markdown('<p class="section-header">Lifestyle & Health</p>', unsafe_allow_html=True)
    sleep_opts  = list(encodings.get('Sleep Duration', {}).keys()) or ["Less than 5 hours","5-6 hours","7-8 hours","More than 8 hours"]
    sleep_dur   = st.selectbox("Sleep Duration", options=sleep_opts)
    diet_opts   = list(encodings.get('Dietary Habits', {}).keys()) or ["Healthy","Moderate","Unhealthy"]
    dietary     = st.selectbox("Dietary Habits", options=diet_opts)
    financial_stress = st.slider("Financial Stress (1–5)", min_value=1, max_value=5, value=3)

    st.markdown('<p class="section-header">Mental Health History</p>', unsafe_allow_html=True)
    suicidal_opts = list(encodings.get('Have you ever had suicidal thoughts ?', {}).keys()) or ["Yes","No"]
    suicidal   = st.selectbox("Ever had suicidal thoughts?", options=suicidal_opts)
    family_opts = list(encodings.get('Family History of Mental Illness', {}).keys()) or ["Yes","No"]
    family_hist = st.selectbox("Family History of Mental Illness", options=family_opts)


st.markdown("---")
predict_col, _ = st.columns([1, 3])

with predict_col:
    predict_btn = st.button("Predict Risk", use_container_width=True, type="primary")

if predict_btn:
    raw = {
        'Gender'          : gender,
        'Age'             : age,
        'City'            : city,
        'Profession'      : profession,
        'Academic Pressure': academic_pressure,
        'Work Pressure'   : work_pressure,
        'CGPA'            : cgpa,
        'Study Satisfaction': study_satisfaction,
        'Job Satisfaction': job_satisfaction,
        'Sleep Duration'  : sleep_dur,
        'Dietary Habits'  : dietary,
        'Degree'          : degree,
        'Have you ever had suicidal thoughts ?': suicidal,
        'Work/Study Hours': work_study_hours,
        'Financial Stress': financial_stress,
        'Family History of Mental Illness': family_hist,
    }

    encoded = {}
    for col in feature_cols:
        val = raw.get(col, 0)
        if col in encodings and isinstance(val, str):
            encoded[col] = encodings[col].get(val, 0)
        else:
            encoded[col] = val

    input_df = pd.DataFrame([encoded])[feature_cols]

    prob = model.predict_proba(input_df)[0][1]
    label, pct_str, css_class, emoji, desc = get_risk_info(prob)

    st.markdown("### Prediction Result")

    r1, r2, r3 = st.columns([2, 1, 1])

    with r1:
        st.markdown(f"""
        <div class="{css_class}">
            <div class="risk-title">{emoji} {label}</div>
            <div class="risk-pct">{pct_str}</div>
            <div class="risk-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        low_p    = max(0, 1 - prob) * 100 if prob > 0.3 else (1 - prob/0.3) * 30
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Depression Probability", f"{prob*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with r3:
        confidence = abs(prob - 0.5) * 200 
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Model Confidence", f"{confidence:.0f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("#### Risk Gauge")
    gauge_pct = prob * 100
    bar_color = "#28a745" if gauge_pct < 30 else ("#ffc107" if gauge_pct < 70 else "#dc3545")
    st.markdown(f"""
    <div style="background:#e9ecef; border-radius:20px; height:28px; width:100%; position:relative; overflow:hidden;">
        <div style="width:{gauge_pct:.1f}%; background:{bar_color}; height:100%; border-radius:20px;
                    transition:width 0.5s ease; display:flex; align-items:center; justify-content:flex-end; padding-right:8px;">
            <span style="color:white; font-weight:700; font-size:0.85rem;">{gauge_pct:.1f}%</span>
        </div>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#6c757d; margin-top:4px;">
        <span>0% (No Risk)</span><span>30%</span><span>70%</span><span>100% (High Risk)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Recommendations")
    if gauge_pct < 30:
        st.success("Your responses suggest low depression risk. Maintain your current healthy lifestyle, social connections, and stress management routines.")
    elif gauge_pct < 70:
        st.warning("Some risk factors detected. Consider: talking to a trusted friend/counsellor, improving sleep habits, reducing academic/work pressure where possible, and practising mindfulness.")
    else:
        st.error("Multiple risk factors detected. **Please consult a mental health professional.** You can also contact iCall (India): **9152987821** or Vandrevala Foundation: **1860-2662-345** (24/7).")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#adb5bd; font-size:0.8rem;'>Depression Risk Predictor · Powered by Gradient Boosting ML · For screening use only</p>", unsafe_allow_html=True)
