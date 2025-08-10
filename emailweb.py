import pickle
import streamlit as st

# ===== Load model and vectorizer =====
feature_extraction = pickle.load(open('E:/feature_extraction.sav', 'rb'))
loaded_model = pickle.load(open('E:/trained_model.sav', 'rb'))

# ===== Prediction function =====
def emailspampredictor(input_mail):
    input_data_features = feature_extraction.transform(input_mail)
    prediction = loaded_model.predict(input_data_features)
    return "Ham" if prediction[0] == 1 else "Spam"

# ===== Streamlit App =====
st.set_page_config(page_title="📧 Email Spam Detector", page_icon="📩", layout="centered")

# ===== Custom CSS =====
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 42px !important;
        color: #4CAF50;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        font-size: 18px !important;
        color: #666666;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #999999;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# ===== Title and subtitle =====
st.markdown('<p class="title">📧 Email Spam Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect whether your email/message is Spam or Ham instantly!</p>', unsafe_allow_html=True)

# ===== Input Area =====
with st.container():
    input_mail = st.text_area("✉ Paste your email/message here:", height=150, placeholder="Type or paste your email text...")

# ===== Predict Button =====
if st.button("🔍 Predict", use_container_width=True):
    if not input_mail.strip():
        st.warning("⚠ Please enter some text before predicting.")
    else:
        result = emailspampredictor([input_mail])

        if result == "Ham":
            st.success("✅ This is a Ham mail (Not Spam).")
        else:
            st.error("🚫 This is a Spam mail! Be cautious.")

# ===== Footer =====
st.markdown('<p class="footer">Made with ❤️ using Streamlit</p>', unsafe_allow_html=True)
