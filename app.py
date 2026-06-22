# Import Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import spacy
from spacy import load
import streamlit as st
import os
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Dropout
import pickle


# ── Text preprocessing ----
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    model_path = "spam_detector.keras"
    tok_path   = "tokenizer.pkl"
 
    if not os.path.exists(model_path):
        st.error(
            f"Model file **{model_path}** not found.\n\n")
        st.stop()
 
    if not os.path.exists(tok_path):
        st.error(f"Tokenizer file **{tok_path}** not found.\n\n")
        st.stop()
 
    model = load_model(model_path)

    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)
 
    return model, tokenizer



@st.cache_resource(show_spinner="Loading language model…")
def load_nlp():
    nlp = load("en_core_web_sm")
    nlp.disable_pipe("ner")
    return nlp
 
 
URL_PATTERN        = r"http\S+|www\S+|https\S+"
ALPHA_PATTERN      = r"[^a-zA-Z\s]"
WHITESPACE_PATTERN = r"\s+"
HTML_TAG_PATTERN   = r"<.*?>|&.*?;|\\n|\\xa0"
 
 
def process_text(text: str, nlp) -> str:
    text = re.sub(URL_PATTERN,        "", text)
    text = re.sub(HTML_TAG_PATTERN,   "", text)
    text = re.sub(ALPHA_PATTERN,      "", text)
    text = re.sub(WHITESPACE_PATTERN, " ", text)
 
    doc   = nlp(text)
    words = [w for w in doc if not w.is_stop and len(w) > 1]
    words = [w.lemma_ for w in words]
    return " ".join(words).lower()
 
 
def predict_email(email_text: str, model, tokenizer, nlp, maxlen: int = 3736):
 
    cleaned  = process_text(email_text, nlp)
    seq      = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(seq, maxlen=maxlen, padding="pre", truncating="post")
    prob     = float(model.predict(padded, verbose=0)[0][0])
    label    = "Spam" if prob > 0.5 else "Ham"
    return label, prob
 

# Streamlit APP

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="🛡️",
    layout="centered",
)

st.title("Email Spam Detector")
st.write("This is a simple Spam Detector.")

email = st.text_input("Enter Email")

if st.button("Predict"):
    st.write('Email submitted successfully')
    
# Load artifacts once
model, tokenizer = load_artifacts()
nlp = load_nlp()


# Text area
email_input = st.text_area(
    label="Email content",
    placeholder="Paste the full email text here (subject + body)…",
    height=220,
    label_visibility="collapsed",
)

# Analyse button
col1, col2 = st.columns([1, 3])
with col1:
    analyse = st.button("Analyse →", use_container_width=True, type="primary")
 

if analyse:
    if not email_input.strip():
        st.warning("Please enter some email text before analysing.")
    else:
        with st.spinner("Processing…"):
            label, prob = predict_email(email_input, model, tokenizer, nlp)
 
        spam_prob = prob
        ham_prob  = 1.0 - prob

        if label == "Spam":
            st.markdown(
                f'<div class="result-box result-spam">🚨 &nbsp; This email looks like <strong>SPAM</strong></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-box result-ham">✅ &nbsp; This email looks <strong>legitimate (HAM)</strong></div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
 
        # Confidence bars
        st.markdown('<p class="prob-label">Confidence breakdown</p>', unsafe_allow_html=True)
 
        c1, c2 = st.columns(2)
        with c1:
            st.metric("🚨 Spam probability", f"{spam_prob * 100:.1f}%")
            st.progress(spam_prob)
        with c2:
            st.metric("✅ Ham probability",  f"{ham_prob  * 100:.1f}%")
            st.progress(ham_prob)
 
        # Pipeline steps (informational)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="prob-label">Processing pipeline</p>',
            unsafe_allow_html=True,
        )
        steps = [
            ("1", "Remove URLs & HTML tags"),
            ("2", "Strip non-alphabetic characters"),
            ("3", "SpaCy stop-word removal"),
            ("4", "Lemmatization"),
            ("5", "Tokenize → pad to 3 736 tokens"),
            ("6", "LSTM inference"),
        ]
        for idx, desc in steps:
            st.markdown(
                f'<span class="step-badge">{idx}</span>{desc}',
                unsafe_allow_html=True,
            )