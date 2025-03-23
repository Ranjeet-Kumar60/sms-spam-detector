import streamlit as st
import pickle
from utils import preprocess_text  # Import from utils.py


# Load trained model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Custom Styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .main-title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #4A90E2;
        }
        .subtext {
            font-size: 16px;
            text-align: center;
            color: #555;
        }
        .result-box {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            padding: 10px;
            border-radius: 10px;
        }
        .spam {
            background-color: #FF4B4B;
            color: white;
        }
        .not-spam {
            background-color: #28A745;
            color: white;
        }
        .input-box {
            font-size: 18px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown('<p class="main-title">📩 SMS Spam Classifier 🔍</p>', unsafe_allow_html=True)
st.markdown('<p class="subtext">Enter a message below and find out if it is spam or not! 🚀</p>', unsafe_allow_html=True)

# Input text area
input_sms = st.text_area("📝 Type your message here:", height=150)

# Predict button
if st.button('🚀 Detect Spam', help="Click to analyze the message"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # Preprocess the input
        transformed_sms = preprocess_text(input_sms)  # Use imported function
        # Vectorize input
        vector_input = tfidf.transform([transformed_sms])
        # Predict
        result = model.predict(vector_input)[0]

        # Display result with styling
        if result == 1:
            st.markdown('<p class="result-box spam">⚠️ Spam Message Detected!</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="result-box not-spam">✅ This is Not Spam.</p>', unsafe_allow_html=True)
