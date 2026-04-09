import streamlit as st
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model + tokenizer
model = tf.keras.models.load_model("spam_lstm_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200

# Keywords for explanation
spam_keywords = ["free", "win", "urgent", "click", "offer", "verify", "account", "prize", "money", "limited"]
ham_keywords = ["meeting", "project", "schedule", "attached", "thanks", "discussion", "team"]

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    return text

def explain_prediction(text):
    words = text.split()
    spam_hits = [w for w in words if w in spam_keywords]
    ham_hits = [w for w in words if w in ham_keywords]
    return spam_hits, ham_hits

# ---------- UI ---------- #

st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="wide")

st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    font-size: 20px;
    color: gray;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #1e1e1e;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>📧 Spam Email Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deep Learning based Spam Detection with Explainability</div>", unsafe_allow_html=True)

st.divider()

# Layout columns
col1, col2 = st.columns([2,1])

with col1:
    email = st.text_area("✉️ Enter Email Content", height=250)

with col2:
    st.info("💡 Tips:\n- Spam emails contain urgency\n- Look for suspicious links\n- Offers that sound too good")

st.divider()

if st.button("🚀 Analyze Email", use_container_width=True):

    if email.strip() == "":
        st.warning("⚠️ Please enter an email")
    else:
        cleaned = preprocess(email)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

        pred = model.predict(padded)[0][0]

        spam_hits, ham_hits = explain_prediction(cleaned)

        st.subheader("📊 Prediction Result")

        # Progress bar
        st.progress(float(pred))

        colA, colB = st.columns(2)

        with colA:
            if pred > 0.7:
                st.error(f"🚨 SPAM DETECTED\n\nConfidence: {pred:.2f}")
            else:
                st.success(f"✅ HAM (SAFE)\n\nConfidence: {pred:.2f}")

        with colB:
            st.metric("Spam Probability", f"{pred:.2f}")

        st.divider()

        # ---------- EXPLANATION ---------- #
        st.subheader("🧠 Why this prediction?")

        if pred > 0.7:
            if spam_hits:
                st.markdown("### 🚨 Detected Spam Indicators:")
                st.write(", ".join(spam_hits))
            else:
                st.write("Contains patterns similar to spam emails.")
        else:
            if ham_hits:
                st.markdown("### ✅ Legitimate Indicators:")
                st.write(", ".join(ham_hits))
            else:
                st.write("Email structure appears normal and safe.")

        # Highlight words
        st.subheader("🔍 Keyword Highlighting")

        highlighted = email
        for word in spam_hits:
            highlighted = highlighted.replace(word, f"🔴{word.upper()}🔴")
        for word in ham_hits:
            highlighted = highlighted.replace(word, f"🟢{word.upper()}🟢")

        st.markdown(highlighted)