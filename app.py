# ============================================================
# AI-Based Mental Health Sentiment Monitoring System
# Professional Streamlit Application
# ============================================================

# =========================
# IMPORT LIBRARIES
# =========================

import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import nltk
import matplotlib.pyplot as plt
import gdown
import os

from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# =========================
# DOWNLOAD NLTK DATA
# =========================

nltk.download('stopwords')

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Mental Health AI Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# THEME TOGGLE
# ============================================================

theme = st.sidebar.toggle("🌙 Dark Mode")

# ============================================================
# LIGHT THEME CSS
# ============================================================

light_css = """
<style>

.stApp {
    background-color: #F4F7FC;
    color: #111111;
}

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    color: #512DA8;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #555555;
    font-size: 20px;
    margin-bottom: 30px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 25px;
    color: #111111;
}

.metric-card {
    background: linear-gradient(to right, #5E35B1, #7E57C2);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
}

.metric-card h2 {
    color: white !important;
}

textarea {
    background-color: white !important;
    color: black !important;
    border-radius: 12px !important;
    border: 2px solid #673AB7 !important;
    font-size: 18px !important;
}

textarea::placeholder {
    color: #666 !important;
}

.stButton>button {
    width: 100%;
    height: 3.5em;
    border-radius: 12px;
    border: none;
    background: linear-gradient(to right, #673AB7, #8E24AA);
    color: white;
    font-size: 18px;
    font-weight: bold;
}

.stMarkdown,
.stText,
label,
p,
div,
span {
    color: #111111 !important;
}

.footer {
    text-align: center;
    color: #666666;
    margin-top: 30px;
}

</style>
"""

# ============================================================
# DARK THEME CSS
# ============================================================

dark_css = """
<style>

.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    color: #BB86FC;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #D0D0D0;
    font-size: 20px;
    margin-bottom: 30px;
}

.card {
    background: #161B22;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 12px rgba(255,255,255,0.05);
    margin-bottom: 25px;
    color: #FAFAFA !important;
}

.metric-card {
    background: linear-gradient(to right, #7B1FA2, #9C27B0);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
}

.metric-card h2 {
    color: white !important;
}

textarea {
    background-color: #1F2937 !important;
    color: white !important;
    border-radius: 12px !important;
    border: 2px solid #BB86FC !important;
    font-size: 18px !important;
}

textarea::placeholder {
    color: #BBBBBB !important;
}

.stButton>button {
    width: 100%;
    height: 3.5em;
    border-radius: 12px;
    border: none;
    background: linear-gradient(to right, #7B1FA2, #AB47BC);
    color: white;
    font-size: 18px;
    font-weight: bold;
}

.stMarkdown,
.stText,
label,
p,
div,
span {
    color: #FAFAFA !important;
}

.footer {
    text-align: center;
    color: #CCCCCC;
    margin-top: 30px;
}

</style>
"""

# ============================================================
# APPLY THEME
# ============================================================

if theme:
    st.markdown(dark_css, unsafe_allow_html=True)
else:
    st.markdown(light_css, unsafe_allow_html=True)

# ============================================================
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# ============================================================

MODEL_FILE = "mental_health_rnn_model.h5"

if not os.path.exists(MODEL_FILE):

    file_id = "1m4xu6JUNdMEAYyeoAsvZCCsuPBmeewdN"

    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(
        url,
        MODEL_FILE,
        quiet=False
    )

# ============================================================
# LOAD MODEL
# ============================================================

model = load_model(MODEL_FILE)

# ============================================================
# LOAD TOKENIZER
# ============================================================

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# ============================================================
# LOAD LABEL ENCODER
# ============================================================

with open("label_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

# ============================================================
# VARIABLES
# ============================================================

stop_words = set(stopwords.words('english'))

max_length = 120

# ============================================================
# PREPROCESS FUNCTION
# ============================================================

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_emotion(text):

    cleaned_text = preprocess_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=max_length,
        padding='post'
    )

    prediction = model.predict(
        padded_sequence,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    predicted_label = encoder.inverse_transform(
        [predicted_index]
    )[0]

    return predicted_label, confidence, prediction[0]

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2785/2785819.png",
        width=100
    )

    st.title("🧠 Mental Health AI")

    st.markdown("""
    ### Features
    
    ✔ Emotion Detection  
    ✔ Sentiment Analysis  
    ✔ Confidence Score  
    ✔ Deep Learning Prediction  
    ✔ Emotional Guidance
    """)

    st.markdown("---")

    st.success("Model Accuracy: 94%")

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class='main-title'>
AI-Based Mental Health Monitoring
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='subtitle'>
Emotion Detection using NLP & Deep Learning
</div>
""", unsafe_allow_html=True)

# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class='metric-card'>
    <h2>94%</h2>
    Accuracy
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='metric-card'>
    <h2>7</h2>
    Emotions
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='metric-card'>
    <h2>RNN</h2>
    Deep Learning
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class='metric-card'>
    <h2>NLP</h2>
    Text Analysis
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns([2,1])

# ============================================================
# LEFT SECTION
# ============================================================

with left:

    st.markdown("""
    <div class='card'>
    """, unsafe_allow_html=True)

    st.subheader("✍️ Analyze Your Emotion")

    user_input = st.text_area(
        "",
        placeholder="Enter your thoughts or feelings here...",
        height=220
    )

    analyze = st.button("🔍 Analyze Emotion")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# RIGHT SECTION
# ============================================================

with right:

    st.markdown("""
    <div class='card'>
    """, unsafe_allow_html=True)

    st.subheader("💡 Wellness Tips")

    st.info("""
✔ Stay hydrated  
✔ Practice mindfulness  
✔ Sleep properly  
✔ Exercise regularly  
✔ Stay socially connected
""")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PREDICTION SECTION
# ============================================================

if analyze:

    if user_input.strip() == "":

        st.warning("⚠ Please enter some text.")

    else:

        emotion, confidence, probabilities = predict_emotion(user_input)

        # ====================================================
        # RESULT SECTION
        # ====================================================

        st.markdown("""
        <div class='card'>
        """, unsafe_allow_html=True)

        st.subheader("📊 Prediction Results")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Detected Emotion",
                emotion
            )

        with c2:
            st.metric(
                "Confidence",
                f"{round(confidence,2)}%"
            )

        with c3:

            if confidence >= 90:
                status = "Strong"

            elif confidence >= 70:
                status = "Moderate"

            else:
                status = "Low"

            st.metric(
                "Signal",
                status
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ====================================================
        # VISUALIZATION
        # ====================================================

        st.markdown("""
        <div class='card'>
        """, unsafe_allow_html=True)

        st.subheader("📈 Emotion Probability Distribution")

        class_labels = encoder.classes_

        fig, ax = plt.subplots(figsize=(10,4))

        bars = ax.bar(class_labels, probabilities)

        ax.set_xlabel("Emotions")

        ax.set_ylabel("Probability")

        ax.set_title("Prediction Confidence")

        plt.xticks(rotation=45)

        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

        # ====================================================
        # GUIDANCE
        # ====================================================

        st.markdown("""
        <div class='card'>
        """, unsafe_allow_html=True)

        st.subheader("💙 Emotional Guidance")

        emotion_lower = emotion.lower()

        if "depression" in emotion_lower or "sad" in emotion_lower:

            st.error("""
Take breaks, practice self-care,
and talk to trusted people.
""")

        elif "anxiety" in emotion_lower or "stress" in emotion_lower:

            st.warning("""
Practice mindfulness,
deep breathing, and relaxation techniques.
""")

        elif "happy" in emotion_lower or "positive" in emotion_lower:

            st.success("""
Positive emotional state detected.
Keep doing activities that motivate you.
""")

        else:

            st.info("""
Maintain healthy habits and emotional balance.
""")

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class='footer'>

<hr>

🧠 AI-Based Mental Health Monitoring System

Built with TensorFlow • NLP • RNN • Streamlit

</div>
""", unsafe_allow_html=True)
