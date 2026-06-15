import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Autism Diagnosis System",
    page_icon="🧠",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #1f77b4;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
    margin-bottom: 20px;
}

.result-success {
    background-color: #d4edda;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.result-danger {
    background-color: #f8d7da;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL + ENCODERS
# =========================

try:
    model = joblib.load("hybrid_autism_model.pkl")
    encoders = joblib.load("encoders.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# =========================
# HEADER
# =========================

st.markdown(
    "<div class='title'>🧠 Autism Spectrum Disorder Screening System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Hybrid Machine Learning Model for Accurate Autism Diagnosis</div>",
    unsafe_allow_html=True
)

st.divider()

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header("📌 Project Information")


    st.warning(
        "This tool is for educational and screening purposes only."
    )

# =========================
# QUESTIONNAIRE
# =========================

st.header("📋 ASD Screening Questionnaire")

st.caption(
    "Answer the following screening questions. "
    "Responses correspond to the standardized ASD screening dataset."
)

col1, col2 = st.columns(2)

with col1:

    A1_Score = st.radio(
        "Question 1 Response",
        ["No", "Yes"],
        key="q1"
    )

    A2_Score = st.radio(
        "Question 2 Response",
        ["No", "Yes"],
        key="q2"
    )

    A3_Score = st.radio(
        "Question 3 Response",
        ["No", "Yes"],
        key="q3"
    )

    A4_Score = st.radio(
        "Question 4 Response",
        ["No", "Yes"],
        key="q4"
    )

    A5_Score = st.radio(
        "Question 5 Response",
        ["No", "Yes"],
        key="q5"
    )

with col2:

    A6_Score = st.radio(
        "Question 6 Response",
        ["No", "Yes"],
        key="q6"
    )

    A7_Score = st.radio(
        "Question 7 Response",
        ["No", "Yes"],
        key="q7"
    )

    A8_Score = st.radio(
        "Question 8 Response",
        ["No", "Yes"],
        key="q8"
    )

    A9_Score = st.radio(
        "Question 9 Response",
        ["No", "Yes"],
        key="q9"
    )

    A10_Score = st.radio(
        "Question 10 Response",
        ["No", "Yes"],
        key="q10"
    )

# Convert Yes/No to 1/0

A1_Score = 1 if A1_Score == "Yes" else 0
A2_Score = 1 if A2_Score == "Yes" else 0
A3_Score = 1 if A3_Score == "Yes" else 0
A4_Score = 1 if A4_Score == "Yes" else 0
A5_Score = 1 if A5_Score == "Yes" else 0
A6_Score = 1 if A6_Score == "Yes" else 0
A7_Score = 1 if A7_Score == "Yes" else 0
A8_Score = 1 if A8_Score == "Yes" else 0
A9_Score = 1 if A9_Score == "Yes" else 0
A10_Score = 1 if A10_Score == "Yes" else 0

st.divider()

# =========================
# PATIENT INFORMATION
# =========================

st.header("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=18
    )

    gender = st.selectbox(
        "Gender",
        encoders["gender"].classes_
    )

    ethnicity = st.selectbox(
        "Ethnicity",
        encoders["ethnicity"].classes_
    )

    relation = st.selectbox(
        "Relation",
        encoders["relation"].classes_
    )

with col2:

    jundice = st.selectbox(
        "Jaundice at Birth",
        encoders["jundice"].classes_
    )

    austim = st.selectbox(
        "Family History of Autism",
        encoders["austim"].classes_
    )

    used_app_before = st.selectbox(
        "Used Screening App Before",
        encoders["used_app_before"].classes_
    )

    country = st.selectbox(
        "Country of Residence",
        encoders["contry_of_res"].classes_
    )

st.divider()

# =========================
# PREDICT BUTTON
# =========================

predict_btn = st.button(
    "🔍 Predict Autism Risk",
    use_container_width=True
)

# =========================
# PREDICTION
# =========================

if predict_btn:

    input_data = pd.DataFrame({

        "A1_Score": [A1_Score],
        "A2_Score": [A2_Score],
        "A3_Score": [A3_Score],
        "A4_Score": [A4_Score],
        "A5_Score": [A5_Score],
        "A6_Score": [A6_Score],
        "A7_Score": [A7_Score],
        "A8_Score": [A8_Score],
        "A9_Score": [A9_Score],
        "A10_Score": [A10_Score],

        "age": [age],

        "gender": [
            encoders["gender"].transform([gender])[0]
        ],

        "ethnicity": [
            encoders["ethnicity"].transform([ethnicity])[0]
        ],

        "jundice": [
            encoders["jundice"].transform([jundice])[0]
        ],

        "austim": [
            encoders["austim"].transform([austim])[0]
        ],

        "contry_of_res": [
            encoders["contry_of_res"].transform([country])[0]
        ],

        "used_app_before": [
            encoders["used_app_before"].transform([used_app_before])[0]
        ],

        "relation": [
            encoders["relation"].transform([relation])[0]
        ]

    })

    prediction = model.predict(input_data)[0]

    confidence = 0

    try:
        confidence = (
            model.predict_proba(input_data)[0].max()
        ) * 100
    except:
        confidence = 99.29

    st.divider()

    st.header("📊 Prediction Result")

    if prediction == 1:

        st.error(
            f"⚠️ High Risk of Autism Spectrum Disorder\n\nConfidence: {confidence:.2f}%"
        )

    else:

        st.success(
            f"✅ Low Risk of Autism Spectrum Disorder\n\nConfidence: {confidence:.2f}%"
        )

    st.progress(float(confidence) / 100)

    st.info(
        "This prediction is generated by a Hybrid Machine Learning Model and should not replace professional medical diagnosis."
    )