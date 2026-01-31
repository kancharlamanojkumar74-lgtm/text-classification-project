import streamlit as st
import pandas as pd
import pickle
import os
from text_classification_system import TextClassificationSystem

st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📧",
    layout="wide"
)

@st.cache_resource
def load_trained_model():
    if os.path.exists('trained_classifier.pkl'):
        with open('trained_classifier.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        classifier = TextClassificationSystem()
        classifier.load_data('spam.csv')
        classifier.prepare_data()
        X_train_tfidf, X_test_tfidf = classifier.vectorize_text()
        classifier.train_models(X_train_tfidf, X_test_tfidf)
        with open('trained_classifier.pkl', 'wb') as f:
            pickle.dump(classifier, f)
        return classifier

def main():
    st.title("📧 SMS Spam Detection System")
    st.markdown("---")
    st.markdown("""
    This application uses machine learning to classify SMS messages as **spam** or **ham** (legitimate).
    
    **Features:**
    - Text preprocessing with NLTK
    - TF-IDF vectorization
    - Naive Bayes and Logistic Regression models
    - Real-time predictions
    """)
    
    with st.spinner("Loading trained model..."):
        classifier = load_trained_model()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Test Your Message")
        user_message = st.text_area(
            "Enter an SMS message to classify:",
            placeholder="Type your message here...",
            height=100
        )
        model_choice = st.selectbox(
            "Choose Model:",
            ["Logistic Regression", "Naive Bayes"],
            help="Select which model to use for prediction"
        )
        if st.button("🚀 Classify Message", type="primary"):
            if user_message.strip():
                model_key = 'lr' if model_choice == "Logistic Regression" else 'nb'
                with st.spinner("Analyzing message..."):
                    result = classifier.predict_new_message(user_message, model_key)
                prediction = "SPAM" if "SPAM" in result.upper() else "HAM"
                if prediction == "SPAM":
                    st.error(f"🚨 **SPAM DETECTED**")
                    st.markdown("This message appears to be spam.")
                else:
                    st.success(f"✅ **LEGITIMATE MESSAGE**")
                    st.markdown("This message appears to be legitimate (ham).")
                st.info(f"**Model Output:** {result}")
            else:
                st.warning("Please enter a message to classify.")
    
    with col2:
        st.subheader("📊 Model Information")
        st.markdown("""
        **Model Performance:**
        - Accuracy: ~97%
        - Precision: ~95%
        - Recall: ~90%
        - F1-Score: ~92%
        
        **Features:**
        - TF-IDF Vectorization
        - Text Preprocessing
        - Stopword Removal
        - 5000 Features
        """)
        st.subheader("🧪 Sample Messages")
        sample_messages = {
            "Spam Example 1": "Congratulations! You've won $1000! Click here to claim your prize now!",
            "Spam Example 2": "URGENT: Your account will be suspended. Call 123-456-7890 immediately!",
            "Ham Example 1": "Hey, are we still meeting for lunch tomorrow?",
            "Ham Example 2": "Thanks for the great presentation today."
        }
        for label, message in sample_messages.items():
            if st.button(f"Try: {label}", key=label):
                st.session_state.sample_message = message
        if 'sample_message' in st.session_state:
            st.rerun()
    
    st.markdown("---")
    with st.expander("ℹ️ About This System"):
        st.markdown("""
        **Text Classification Pipeline:**
        
        1. **Text Preprocessing:**
           - Convert to lowercase
           - Remove punctuation and numbers
           - Tokenization
           - Remove stopwords
        
        2. **Feature Extraction:**
           - TF-IDF (Term Frequency-Inverse Document Frequency)
           - Creates numerical features from text
        
        3. **Machine Learning Models:**
           - **Naive Bayes:** Fast, assumes feature independence
           - **Logistic Regression:** More flexible, handles feature interactions
        
        4. **Evaluation Metrics:**
           - Accuracy, Precision, Recall, F1-Score
           - Confusion Matrix analysis
        
        **Dataset:** SMS Spam Collection Dataset
        **Libraries:** scikit-learn, NLTK, pandas, numpy
        """)
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and scikit-learn")

if __name__ == "__main__":
    main()