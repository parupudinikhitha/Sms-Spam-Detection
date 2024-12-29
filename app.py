
import nltk
import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from datetime import datetime

nltk.download('stopwords')
nltk.download('punkt')

# Initialize the Porter Stemmer
port_stemmer = PorterStemmer()

# Load pre-trained vectorizer and model
with open('vectorizer.pkl', 'rb') as vec_file:
    tfidf = pickle.load(vec_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    print("Model Type:",type(model))
    print("Model Details:",model)

# Load stopwords once
stop_words = set(stopwords.words('english'))

# Define the clean_text function
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and convert to lowercase
    text = [word for word in text.split() if word not in stop_words]  # Remove stopwords
    text = [port_stemmer.stem(word) for word in text]  # Apply stemming
    return ' '.join(text)

# Apply custom styles using CSS
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
        }
        .stButton>button {
            color: white;
            background: #333333; /* Dark background */
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background: #555555; /* Slightly lighter on hover */
        }
        .stHeader, .stSubheader {
            color: #2c3e50;
        }
        .message-history {
            font-family: "Arial", sans-serif;
            color: #34495e;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title('🚀 SMS Spam Classifier')
st.subheader('🌟 Enter your message below to check if it’s Spam or Not!')

# Input area for SMS message
input_sms = st.text_area("Enter the Message", placeholder="Type your message here...")

# Initialize message history if not already in session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Predict button
if st.button('Predict'):
    if not input_sms.strip():
        st.warning('Please enter a message to classify!')
    else:
        # Preprocess the input
        transform_text = clean_text(input_sms)
        
        # Vectorize the preprocessed text
        vector_input = tfidf.transform([transform_text])
        
        # Predict the result
        result = model.predict(vector_input)[0]
        
        # Determine classification label
        label = "Spam" if result == 1 else "Not Spam"
        # Capture the time of prediction for this message
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Add the message, result, and the time of prediction to history
        st.session_state.history.append((input_sms, label, current_time))

        # Display prediction and notification
        st.header(label)
        if label == "Spam":
            st.warning("⚠ This message is classified as Spam!")
        else:
            st.success("✅ This message is classified as Not Spam!")

        # Show the time the message was sent (fixed)
        st.write(f"Message sent at: {current_time}")

# Display the message history
if st.session_state.history:
    st.subheader("📜 Message History")
    for message, label, time in reversed(st.session_state.history):  # Show most recent at the top
        st.markdown(f"<div class='message-history'><strong>{label}:</strong> {message} <em>(Sent at: {time})</em></div>", unsafe_allow_html=True)
