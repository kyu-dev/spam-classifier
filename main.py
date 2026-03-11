import pickle
import torch
import streamlit as st
from model import SpamClassifier


# ── Load (cached so it only runs once) ───────────────────────────────────────

@st.cache_resource
def load_model():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    input_size = len(vectorizer.vocabulary_)
    model = SpamClassifier(input_size=input_size)
    model.load_state_dict(torch.load('model.pth', weights_only=True))
    model.eval()
    return model, vectorizer


# ── App ───────────────────────────────────────────────────────────────────────

model, vectorizer = load_model()

st.title("Spam Classifier")
st.write("Type a message below to find out if it's spam or ham.")

message = st.text_area("Message", placeholder="Enter your message here…")

if st.button("Classify") and message.strip():
    vec = torch.tensor(
        vectorizer.transform([message]).toarray(), dtype=torch.float32
    )
    with torch.no_grad():
        prob = torch.sigmoid(model(vec)).item()

    label = "SPAM" if prob > 0.5 else "HAM"
    confidence = prob if label == "SPAM" else 1 - prob

    if label == "SPAM":
        st.error(f"SPAM — {confidence * 100:.0f}% confidence")
    else:
        st.success(f"HAM — {confidence * 100:.0f}% confidence")
