
# ============================================================
# AI-Based Mental Health Sentiment Monitoring System
# Streamlit Application
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
    page_title="Mental Health Sentiment Monitoring",
    page_icon="🧠",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #4A148C;
    text-align: center;
}

h2, h3 {
    color: #6A1B9A;
}

.stButton>button {
    width: 100%;
    background-color: #6A1B9A;
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
}

.stTextArea textarea {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================

model = load_model("mental_health_rnn_model.h5")

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
# HEADER SECTION
# ============================================================

st.markdown("""
# 🧠 AI-Based Mental Health Sentiment Monitoring System
""")

st.markdown("""
### Emotion Detection using Simple Recurrent Neural Networks
""")

st.markdown("---")

# ============================================================
# ABOUT PROJECT
# ============================================================

st.subheader("📘 About the Project")

st.write("""
This AI-powered system analyzes emotional sentiment
from user text messages using NLP and Deep Learning.

### Importance of Emotional AI
- Monitors emotional well-being
- Detects negative sentiment patterns
- Supports early intervention

### NLP Applications
- Sentiment Analysis
- Mental Health Monitoring
- Chatbots
- Recommendation Systems

### Role of RNN in Sequence Learning
RNN learns sequential text patterns and remembers
previous words using hidden states.
""")

st.markdown("---")

# ============================================================
# USER INPUT SECTION
# ============================================================

st.subheader("✍️ Enter Your Thoughts")

st.write("### Example Sentences")

st.write("""
- I feel anxious and stressed today
- Nobody understands my feelings
- I am extremely happy and motivated
- I feel lonely and depressed
- Today was peaceful and relaxing
""")

user_input = st.text_area(
    "Enter your thoughts or feelings here...",
    height=180
)

# ============================================================
# BUTTON
# ============================================================

analyze = st.button("🔍 Analyze Emotion")

# ============================================================
# PREDICTION SECTION
# ============================================================

if analyze:

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        emotion, confidence, probabilities = predict_emotion(user_input)

        st.markdown("---")

        st.subheader("📊 Prediction Output")

        st.success(f"Emotion Detected: {emotion}")

        st.info(f"Confidence Score: {round(confidence,2)}%")

        # Emotional Status
        if confidence >= 90:
            status = "Strong Emotional Signal"

        elif confidence >= 70:
            status = "Moderate Emotional Signal"

        else:
            status = "Low Emotional Signal"

        st.warning(f"Emotional Status: {status}")

        st.markdown("---")

        # ====================================================
        # VISUALIZATION
        # ====================================================

        st.subheader("📈 Sentiment Confidence Graph")

        class_labels = encoder.classes_

        fig, ax = plt.subplots(figsize=(10,5))

        ax.bar(class_labels, probabilities)

        ax.set_xlabel("Emotion Categories")

        ax.set_ylabel("Confidence")

        ax.set_title("Emotion Probability Distribution")

        plt.xticks(rotation=45)

        st.pyplot(fig)

        st.markdown("---")

        # ====================================================
        # EMOTIONAL GUIDANCE
        # ====================================================

        st.subheader("💡 Emotional Wellness Guidance")

        emotion_lower = emotion.lower()

        if "depression" in emotion_lower or "sad" in emotion_lower:

            st.error("""
Take a short break and talk to someone you trust.

Positive Activities:
- Walking
- Listening to music
- Meditation
- Journaling
""")

        elif "anxiety" in emotion_lower or "stress" in emotion_lower:

            st.warning("""
Try deep breathing exercises and relaxation techniques.

Positive Activities:
- Yoga
- Meditation
- Reading
- Nature walks
""")

        elif "happy" in emotion_lower or "positive" in emotion_lower:

            st.success("""
Great to hear positive emotions!

Continue activities that keep you motivated
and emotionally healthy.
""")

        elif "normal" in emotion_lower:

            st.info("""
Maintain a balanced lifestyle with:
- Proper sleep
- Healthy diet
- Exercise
- Social interaction
""")

        else:

            st.write("""
Practice self-care and stay connected
with supportive people.
""")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<center>

AI-Based Mental Health Sentiment Monitoring System

Built using TensorFlow, NLP, LSTM, and Streamlit

</center>
""", unsafe_allow_html=True)
