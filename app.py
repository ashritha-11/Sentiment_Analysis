# ============================================================
# AI-Based Mental Health Sentiment Monitoring System
# Professional Streamlit Web Application
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
import pandas as pd
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

# =========================
# CUSTOM CSS STYLING
# =========================

st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(to right, #eef2f3, #dfe9f3);
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #4A148C;
    margin-bottom: 5px;
}

/* Subtitle */
.sub-title {
    text-align: center;
    font-size: 22px;
    color: #6A1B9A;
    margin-bottom: 30px;
}

/* Cards */
.card {
    background-color: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    margin-bottom: 25px;
}

/* Text area */
textarea {
    border-radius: 15px !important;
    border: 2px solid #6A1B9A !important;
    padding: 15px !important;
    font-size: 18px !important;
}

/* Button */
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

/* Prediction boxes */
.result-box {
    padding: 20px;
    border-radius: 15px;
    color: white;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    margin-top: 15px;
}

/* Footer */
.footer {
    text-align: center;
    color: #555;
    padding-top: 20px;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

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

# =========================
# LOAD MODEL
# =========================

model = load_model(MODEL_FILE)

# =========================
# LOAD TOKENIZER
# =========================

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# =========================
# LOAD LABEL ENCODER
# =========================

with open("label_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

# =========================
# STOPWORDS
# =========================

stop_words = set(stopwords.words('english'))

# =========================
# MAX LENGTH
# =========================

max_length = 120

# ============================================================
# PREPROCESSING FUNCTION
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
    ✔ Sentiment Visualization  
    ✔ Wellness Guidance  
    ✔ Deep Learning Prediction  
    """)

    st.markdown("---")

    st.info("""
    This application uses:
    
    - TensorFlow
    - NLP
    - LSTM / RNN
    - Streamlit
    """)

# ============================================================
# HEADER SECTION
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
from user text messages using Natural Language Processing (NLP)
and Deep Learning techniques.

### 🌟 Importance of Emotional AI
- Monitor emotional well-being
- Detect negative sentiment patterns
- Support mental health awareness
- Assist counselors with early intervention

### 🤖 NLP Applications
- Sentiment Analysis
- Mental Health Monitoring
- Intelligent Chatbots
- Recommendation Systems

### 🔁 Role of RNN
Recurrent Neural Networks (RNNs) process text sequentially
and remember previous words using hidden states,
helping the model understand emotional context.

</div>
""", unsafe_allow_html=True)

# ============================================================
# USER INPUT SECTION
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
# PREDICTION SECTION
# ============================================================

if analyze:

    if user_input.strip() == "":

        st.warning("⚠ Please enter some text.")

    else:

        emotion, confidence, probabilities = predict_emotion(user_input)

        # ====================================================
        # RESULTS
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
        # VISUALIZATION
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
        # WELLNESS GUIDANCE
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
- Talk to a trusted friend
- Practice meditation
- Write your thoughts in a journal
""")

        elif "anxiety" in emotion_lower or "stress" in emotion_lower:

            st.warning("""
### 🌿 Stress Management Tips
- Try deep breathing exercises
- Focus on one task at a time
- Avoid overthinking
- Practice yoga or meditation
- Spend time outdoors
""")

        elif "happy" in emotion_lower or "positive" in emotion_lower:

            st.success("""
### ✨ Positive Emotional State
Wonderful! Continue activities that:
- Keep you motivated
- Improve your productivity
- Maintain emotional balance
""")

        elif "normal" in emotion_lower:

            st.info("""
### 🌸 Healthy Lifestyle Tips
- Maintain proper sleep
- Eat healthy food
- Stay socially connected
- Exercise regularly
""")

        else:

            st.write("""
### 💖 General Wellness Advice
Practice self-care, stay connected
with supportive people,
and engage in activities
that improve emotional well-being.
""")

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

<hr>

🧠 AI-Based Mental Health Sentiment Monitoring System

Built using TensorFlow, NLP, Deep Learning, LSTM & Streamlit

</div>
""", unsafe_allow_html=True)
