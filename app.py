# ============================================================
# AI-Based Mental Health Sentiment Monitoring System
# Professional Streamlit Application with Dark/Light Theme
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
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Mental Health AI Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# THEME SELECTION
# ============================================================

theme = st.sidebar.selectbox(
    "🎨 Select Theme",
    ["Light Mode", "Dark Mode"]
)

# ============================================================
# LIGHT THEME CSS
# ============================================================

light_theme = """
<style>

.stApp {
    background: linear-gradient(to right, #eef2f3, #dfe9f3);
    color: black;
}

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #4A148C;
}

.sub-title {
    text-align: center;
    font-size: 22px;
    color: #6A1B9A;
    margin-bottom: 30px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    margin-bottom: 25px;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #6A1B9A, #8E24AA);
    color: white;
    border-radius: 12px;
    height: 3.5em;
    font-size: 20px;
    border: none;
    font-weight: bold;
}

textarea {
    border-radius: 15px !important;
    border: 2px solid #6A1B9A !important;
    padding: 15px !important;
    font-size: 18px !important;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    color: white;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    margin-top: 15px;
}

.footer {
    text-align: center;
    color: #555;
    padding-top: 20px;
}

</style>
"""

# ============================================================
# DARK THEME CSS
# ============================================================

dark_theme = """
<style>

.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #BB86FC;
}

.sub-title {
    text-align: center;
    font-size: 22px;
    color: #D1C4E9;
    margin-bottom: 30px;
}

.card {
    background-color: #1E1E1E;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 4px 15px rgba(255,255,255,0.1);
    margin-bottom: 25px;
    color: white;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #7B1FA2, #9C27B0);
    color: white;
    border-radius: 12px;
    height: 3.5em;
    font-size: 20px;
    border: none;
    font-weight: bold;
}

textarea {
    border-radius: 15px !important;
    border: 2px solid #BB86FC !important;
    padding: 15px !important;
    font-size: 18px !important;
    background-color: #2A2A2A !important;
    color: white !important;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    color: white;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    margin-top: 15px;
}

.footer {
    text-align: center;
    color: #E0E0E0;
    padding-top: 20px;
}

</style>
"""

# ============================================================
# APPLY THEME
# ============================================================

if theme == "Dark Mode":
    st.markdown(dark_theme, unsafe_allow_html=True)
else:
    st.markdown(light_theme, unsafe_allow_html=True)

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
# STOPWORDS
# ============================================================

stop_words = set(stopwords.words('english'))

# ============================================================
# MAX LENGTH
# ============================================================

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
        "https://cdn-icons-png.flaticon.com/512/3774/3774299.png",
        width=120
    )

    st.title("🧠 Mental Health AI")

    st.markdown("""
    ### Features
    ✔ Emotion Detection  
    ✔ Confidence Analysis  
    ✔ Probability Visualization  
    ✔ Wellness Guidance  
    ✔ Deep Learning Prediction  
    """)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-title">
AI-Based Mental Health Sentiment Monitoring System
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-title">
Emotion Detection using Deep Learning & Recurrent Neural Networks
</div>
""", unsafe_allow_html=True)

# ============================================================
# ABOUT SECTION
# ============================================================

st.markdown("""
<div class="card">

## 📘 About the Project

This AI-powered system analyzes emotional sentiment
from user text messages using NLP and Deep Learning.

### 🌟 Importance of Emotional AI
- Emotional Well-being Monitoring
- Sentiment Detection
- Mental Health Awareness
- Early Intervention Support

### 🤖 NLP Applications
- Sentiment Analysis
- AI Chatbots
- Recommendation Systems
- Mental Health Monitoring

### 🔁 Role of RNN
RNN processes text sequentially and remembers
previous words using hidden states.

</div>
""", unsafe_allow_html=True)

# ============================================================
# INPUT SECTION
# ============================================================

st.markdown("""
<div class="card">
""", unsafe_allow_html=True)

st.subheader("✍️ Enter Your Thoughts")

st.markdown("""
### 💬 Example Sentences

- I feel anxious and stressed today
- Nobody understands my feelings
- I am extremely happy and motivated
- I feel lonely and depressed
- Today was peaceful and relaxing
""")

user_input = st.text_area(
    "",
    placeholder="Enter your thoughts or feelings here...",
    height=220
)

analyze = st.button("🔍 Analyze Emotion")

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PREDICTION OUTPUT
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
        <div class="card">
        """, unsafe_allow_html=True)

        st.subheader("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(f"""
            <div class="result-box" style="background:#6A1B9A;">
            Emotion<br>{emotion}
            </div>
            """, unsafe_allow_html=True)

        with col2:

            st.markdown(f"""
            <div class="result-box" style="background:#00897B;">
            Confidence<br>{round(confidence,2)}%
            </div>
            """, unsafe_allow_html=True)

        with col3:

            if confidence >= 90:
                status = "Strong Signal"
                color = "#D32F2F"

            elif confidence >= 70:
                status = "Moderate Signal"
                color = "#F57C00"

            else:
                status = "Low Signal"
                color = "#1976D2"

            st.markdown(f"""
            <div class="result-box" style="background:{color};">
            Status<br>{status}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ====================================================
        # GRAPH SECTION
        # ====================================================

        st.markdown("""
        <div class="card">
        """, unsafe_allow_html=True)

        st.subheader("📈 Sentiment Confidence Visualization")

        class_labels = encoder.classes_

        fig, ax = plt.subplots(figsize=(12,5))

        ax.bar(class_labels, probabilities)

        ax.set_xlabel("Emotion Categories")

        ax.set_ylabel("Confidence")

        ax.set_title("Emotion Probability Distribution")

        plt.xticks(rotation=45)

        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

        # ====================================================
        # GUIDANCE SECTION
        # ====================================================

        st.markdown("""
        <div class="card">
        """, unsafe_allow_html=True)

        st.subheader("💡 Emotional Wellness Guidance")

        emotion_lower = emotion.lower()

        if "depression" in emotion_lower or "sad" in emotion_lower:

            st.error("""
### 🌼 Suggested Activities
- Take a short walk
- Listen to calming music
- Meditation
- Journaling
- Talk to trusted people
""")

        elif "anxiety" in emotion_lower or "stress" in emotion_lower:

            st.warning("""
### 🌿 Stress Management Tips
- Deep breathing exercises
- Yoga & Meditation
- Nature walks
- Reduce overthinking
- Focus on one task at a time
""")

        elif "happy" in emotion_lower or "positive" in emotion_lower:

            st.success("""
### ✨ Positive Emotional State
Great to hear positive emotions.

Continue activities that:
- Keep you motivated
- Improve productivity
- Maintain emotional balance
""")

        else:

            st.info("""
### 💖 General Wellness Tips
- Maintain proper sleep
- Exercise regularly
- Stay socially connected
- Practice self-care
""")

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

<hr>

🧠 AI-Based Mental Health Sentiment Monitoring System

Built using TensorFlow, NLP, Deep Learning & Streamlit

</div>
""", unsafe_allow_html=True)
