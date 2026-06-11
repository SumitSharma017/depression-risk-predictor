import streamlit as st
import pandas as pd
import numpy as np
import json
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Depression Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { text-align:center; padding:1.5rem 0 0.5rem 0; }
    .main-header h1 { font-size:2.2rem; font-weight:700; }
    .subtitle { color:#6c757d; text-align:center; margin-bottom:2rem; font-size:1rem; }
    .risk-low {
        background:linear-gradient(135deg,#d4edda,#c3e6cb);
        border-left:6px solid #28a745; padding:1.2rem 1.5rem;
        border-radius:10px; margin:1rem 0;
    }
    .risk-moderate {
        background:linear-gradient(135deg,#fff3cd,#ffeaa7);
        border-left:6px solid #ffc107; padding:1.2rem 1.5rem;
        border-radius:10px; margin:1rem 0;
    }
    .risk-high {
        background:linear-gradient(135deg,#f8d7da,#f5c2c7);
        border-left:6px solid #dc3545; padding:1.2rem 1.5rem;
        border-radius:10px; margin:1rem 0;
    }
    .risk-title { font-size:1.5rem; font-weight:700; margin-bottom:0.3rem; }
    .risk-pct   { font-size:2.5rem; font-weight:800; }
    .risk-desc  { font-size:0.95rem; color:#495057; margin-top:0.4rem; }
    .section-header {
        font-size:1.1rem; font-weight:600; color:#343a40;
        margin-top:0.5rem; border-bottom:2px solid #e9ecef;
        padding-bottom:0.3rem; margin-bottom:0.8rem;
    }
    .disclaimer {
        background:#e9ecef; border-radius:8px; padding:0.8rem 1rem;
        font-size:0.82rem; color:#6c757d; margin-top:1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Train model from CSV (cached — runs only once per session) ─────────────────
@st.cache_resource(show_spinner="Training model on your data…")
def train_model(csv_path: str):
    df = pd.read_csv(csv_path)

    # Drop id column if present
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    # Fill missing values
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Label encode + save mappings
    encodings = {}
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encodings[col] = {cls: int(idx) for idx, cls in enumerate(le.classes_)}

    feature_cols = [c for c in df_enc.columns if c != 'Depression']
    X = df_enc[feature_cols]
    y = df_enc['Depression']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=4, subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)

    return model, encodings, feature_cols

CSV_CANDIDATES = ['Student_Depression.csv']

csv_found = None
for c in CSV_CANDIDATES:
    if os.path.exists(c):
        csv_found = c
        break


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 Depression Risk Predictor</h1>
</div>
<p class="subtitle">Fill in the details below to receive a personalised depression risk assessment.</p>
""", unsafe_allow_html=True)


# ── CSV Upload fallback ────────────────────────────────────────────────────────
if csv_found is None:
    st.warning("📂 **Dataset not found.** Please upload your `depression_data.csv` below (it is only used to train the model — never stored).")
    uploaded = st.file_uploader("Upload depression_data.csv", type=["csv"])
    if uploaded:
        tmp_path = "/tmp/depression_data.csv"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())
        csv_found = tmp_path
    else:
        st.info("Alternatively, push `depression_data.csv` to the root of your GitHub repo and redeploy.")
        st.stop()

model, encodings, feature_cols = train_model(csv_found)


# ── Risk Helper ────────────────────────────────────────────────────────────────
def get_risk_info(prob):
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


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.markdown("### About")
    st.markdown("""
This tool uses a **Gradient Boosting** classifier trained on survey data to estimate
depression risk on a 0–100 % scale, bucketed into three bands:

| Band | Range | Meaning |
|------|-------|---------|
| 🟢 Low | 0–30 % | Low concern |
| 🟡 Moderate | 30–70 % | Some factors present |
| 🔴 High | 70–100 % | Professional help advised |
""")
    st.markdown("---")
    st.markdown("**Note:** Screening tool only — not a medical diagnosis.")


# ── Input Form ─────────────────────────────────────────────────────────────────
st.markdown("### 📋 Enter Your Details")
col1, col2, col3 = st.columns(3)

def opts(key, fallback):
    return list(encodings.get(key, {}).keys()) or fallback

with col1:
    st.markdown('<p class="section-header">👤 Demographics</p>', unsafe_allow_html=True)
    gender     = st.selectbox("Gender",     options=opts('Gender',     ['Male','Female']))
    age        = st.slider("Age", 16, 60, 22)
    city       = st.selectbox("City",       options=opts('City',       ['Mumbai','Delhi','Bangalore','Chennai','Other']))
    profession = st.selectbox("Profession", options=opts('Profession', ['Student','Engineer','Doctor','Teacher','Other']))
    degree     = st.selectbox("Degree",     options=opts('Degree',     ['B.Tech','M.Tech','MBA','MBBS','Other']))

with col2:
    st.markdown('<p class="section-header">📚 Academic & Work</p>', unsafe_allow_html=True)
    cgpa               = st.slider("CGPA",                    0.0, 10.0, 7.5, 0.1)
    academic_pressure  = st.slider("Academic Pressure (1–5)", 1, 5, 3)
    work_pressure      = st.slider("Work Pressure (1–5)",     1, 5, 3)
    study_satisfaction = st.slider("Study Satisfaction (1–5)",1, 5, 3)
    job_satisfaction   = st.slider("Job Satisfaction (1–5)",  1, 5, 3)
    work_study_hours   = st.slider("Work/Study Hours/Day",    0, 16, 8)

with col3:
    st.markdown('<p class="section-header">🏥 Lifestyle & Health</p>', unsafe_allow_html=True)
    sleep_dur      = st.selectbox("Sleep Duration",   options=opts('Sleep Duration',   ['Less than 5 hours','5-6 hours','7-8 hours','More than 8 hours']))
    dietary        = st.selectbox("Dietary Habits",   options=opts('Dietary Habits',   ['Healthy','Moderate','Unhealthy']))
    financial_stress = st.slider("Financial Stress (1–5)", 1, 5, 3)

    st.markdown('<p class="section-header">🧬 Mental Health History</p>', unsafe_allow_html=True)
    suicidal    = st.selectbox("Ever had suicidal thoughts?",      options=opts('Have you ever had suicidal thoughts ?', ['Yes','No']))
    family_hist = st.selectbox("Family History of Mental Illness", options=opts('Family History of Mental Illness',     ['Yes','No']))


# ── Predict ────────────────────────────────────────────────────────────────────
st.markdown("---")
btn_col, _ = st.columns([1, 3])
with btn_col:
    predict_btn = st.button("🔍 Predict Risk", use_container_width=True, type="primary")

if predict_btn:
    raw = {
        'Gender': gender, 'Age': age, 'City': city, 'Profession': profession,
        'Academic Pressure': academic_pressure, 'Work Pressure': work_pressure,
        'CGPA': cgpa, 'Study Satisfaction': study_satisfaction,
        'Job Satisfaction': job_satisfaction, 'Sleep Duration': sleep_dur,
        'Dietary Habits': dietary, 'Degree': degree,
        'Have you ever had suicidal thoughts ?': suicidal,
        'Work/Study Hours': work_study_hours,
        'Financial Stress': financial_stress,
        'Family History of Mental Illness': family_hist,
    }

    encoded = {}
    for col in feature_cols:
        val = raw.get(col, 0)
        encoded[col] = encodings[col].get(val, 0) if (col in encodings and isinstance(val, str)) else val

    input_df = pd.DataFrame([encoded])[feature_cols]
    prob     = model.predict_proba(input_df)[0][1]
    label, pct_str, css_class, emoji, desc = get_risk_info(prob)
    gauge_pct = prob * 100

    # Result card
    st.markdown("### 📊 Prediction Result")
    r1, r2, r3 = st.columns([2, 1, 1])

    with r1:
        st.markdown(f"""
        <div class="{css_class}">
            <div class="risk-title">{emoji} {label}</div>
            <div class="risk-pct">{pct_str}</div>
            <div class="risk-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

    with r2:
        st.metric("Depression Probability", f"{gauge_pct:.1f}%")

    with r3:
        confidence = abs(prob - 0.5) * 200
        st.metric("Model Confidence", f"{confidence:.0f}%")

    # Gauge bar
    bar_color = "#28a745" if gauge_pct < 30 else ("#ffc107" if gauge_pct < 70 else "#dc3545")
    st.markdown("#### Risk Gauge")
    st.markdown(f"""
    <div style="background:#e9ecef;border-radius:20px;height:28px;width:100%;overflow:hidden;">
        <div style="width:{gauge_pct:.1f}%;background:{bar_color};height:100%;border-radius:20px;
                    display:flex;align-items:center;justify-content:flex-end;padding-right:8px;">
            <span style="color:white;font-weight:700;font-size:0.85rem;">{gauge_pct:.1f}%</span>
        </div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#6c757d;margin-top:4px;">
        <span>0% (No Risk)</span><span>30%</span><span>70%</span><span>100% (High Risk)</span>
    </div>""", unsafe_allow_html=True)

    # Top features
    st.markdown("#### 🔑 Key Risk Factors in Your Input")
    feat_imp = pd.Series(model.feature_importances_, index=feature_cols).nlargest(6)
    f1, f2 = st.columns(2)
    for i, (feat, _) in enumerate(feat_imp.items()):
        val = input_df[feat].values[0]
        (f1 if i % 2 == 0 else f2).markdown(f"• **{feat}**: {val}")

    # Recommendations
    st.markdown("#### 💡 Recommendations")
    if gauge_pct < 30:
        st.success("✅ Low depression risk detected. Keep maintaining healthy habits, social connections, and stress management.")
    elif gauge_pct < 70:
        st.warning("⚠️ Some risk factors present. Consider talking to a counsellor, improving sleep, and reducing pressure where possible.")
    else:
        st.error("🚨 Multiple risk factors detected. **Please consult a mental health professional.** iCall (India): **9152987821** | Vandrevala Foundation: **1860-2662-345** (24/7)")

    st.markdown("""
    <div class="disclaimer">
        ⚕️ <strong>Medical Disclaimer:</strong> This is a screening tool only and does not constitute a medical diagnosis.
        If you are in crisis, please contact emergency services immediately.
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#adb5bd;font-size:0.8rem;'>Depression Risk Predictor · Powered by Gradient Boosting ML · For screening use only</p>", unsafe_allow_html=True)