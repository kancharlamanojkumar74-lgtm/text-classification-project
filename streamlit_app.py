import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📧",
    layout="wide"
)

@st.cache_data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    return True

@st.cache_resource
def load_and_train_model():
    download_nltk_data()
    
    try:
        data = pd.read_csv('spam.csv', encoding='latin-1')
        data = data.iloc[:, :2]
        data.columns = ['label', 'message']
        data = data.dropna()
        
        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
            return ' '.join(tokens)
        
        data['processed_message'] = data['message'].apply(preprocess_text)
        X = data['processed_message']
        y = data['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        vectorizer = TfidfVectorizer(
            max_features=5000, min_df=2, max_df=0.95, stop_words='english'
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        nb_model = MultinomialNB()
        nb_model.fit(X_train_tfidf, y_train)
        
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_tfidf, y_train)
        
        nb_pred = nb_model.predict(X_test_tfidf)
        lr_pred = lr_model.predict(X_test_tfidf)
        
        nb_f1 = f1_score(y_test, nb_pred, pos_label='spam')
        lr_f1 = f1_score(y_test, lr_pred, pos_label='spam')
        
        best_model = nb_model if nb_f1 > lr_f1 else lr_model
        best_name = "Naive Bayes" if nb_f1 > lr_f1 else "Logistic Regression"
        
        return {
            'vectorizer': vectorizer,
            'nb_model': nb_model,
            'lr_model': lr_model,
            'best_model': best_model,
            'best_name': best_name,
            'preprocess_func': preprocess_text,
            'dataset_size': len(data),
            'nb_f1': nb_f1,
            'lr_f1': lr_f1
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

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
    
    with st.spinner("Loading and training models..."):
        model_data = load_and_train_model()
    
    if model_data is None:
        st.error("Failed to load the model. Please check if spam.csv is available.")
        st.info("For deployment, make sure spam.csv is in your repository root.")
        return
    
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
            ["Best Model", "Naive Bayes", "Logistic Regression"],
            help="Select which model to use for prediction"
        )
        
        if st.button("🚀 Classify Message", type="primary"):
            if user_message.strip():
                with st.spinner("Analyzing message..."):
                    processed_msg = model_data['preprocess_func'](user_message)
                    msg_tfidf = model_data['vectorizer'].transform([processed_msg])
                    
                    if model_choice == "Naive Bayes":
                        model = model_data['nb_model']
                        model_name = "Naive Bayes"
                    elif model_choice == "Logistic Regression":
                        model = model_data['lr_model']
                        model_name = "Logistic Regression"
                    else:
                        model = model_data['best_model']
                        model_name = f"{model_data['best_name']} (Best)"
                    
                    prediction = model.predict(msg_tfidf)[0]
                    probabilities = model.predict_proba(msg_tfidf)[0]
                    confidence = max(probabilities)
                
                if prediction == "spam":
                    st.error("🚨 **SPAM DETECTED**")
                    st.markdown("This message appears to be spam.")
                else:
                    st.success("✅ **LEGITIMATE MESSAGE**")
                    st.markdown("This message appears to be legitimate (ham).")
                
                st.info(f"**Model:** {model_name} | **Confidence:** {confidence:.1%}")
            else:
                st.warning("Please enter a message to classify.")
    
    with col2:
        st.subheader("📊 Model Information")
        if model_data:
            st.markdown(f"""
            **Dataset:**
            - Total messages: {model_data['dataset_size']:,}
            - Training completed successfully
            
            **Model Performance:**
            - Naive Bayes F1: {model_data['nb_f1']:.3f}
            - Logistic Regression F1: {model_data['lr_f1']:.3f}
            - Best Model: {model_data['best_name']}
            
            **Features:**
            - TF-IDF Vectorization
            - Text Preprocessing
            - Stopword Removal
            - 5000 Features Max
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
                st.session_state.user_message = message
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
           - Model comparison and selection
        
        **Dataset:** SMS Spam Collection Dataset
        **Libraries:** scikit-learn, NLTK, pandas, numpy
        """)
    
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and scikit-learn")

if __name__ == "__main__":
    main()
