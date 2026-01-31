
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
print("NLTK data downloaded successfully!")

class TextClassificationSystem:
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.nb_model = None
        self.lr_model = None
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self, filepath):
        print("Loading dataset...")
        self.data = pd.read_csv(filepath, encoding='latin-1')
        self.data = self.data.iloc[:, :2]
        self.data.columns = ['label', 'message']
        self.data = self.data.dropna()
        
        print(f"Dataset loaded successfully!")
        print(f"Total samples: {len(self.data)}")
        print(f"Spam messages: {len(self.data[self.data['label'] == 'spam'])}")
        print(f"Ham messages: {len(self.data[self.data['label'] == 'ham'])}")
        
        return self.data
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        return ' '.join(tokens)
    
    def prepare_data(self):
        print("\nPreprocessing text data...")
        self.data['processed_message'] = self.data['message'].apply(self.preprocess_text)
        X = self.data['processed_message']
        y = self.data['label']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Training samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
        
    def vectorize_text(self):
        print("\nConverting text to numerical features using TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        X_test_tfidf = self.vectorizer.transform(self.X_test)
        print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
        print(f"Number of features: {len(self.vectorizer.get_feature_names_out())}")
        return X_train_tfidf, X_test_tfidf
    
    def train_models(self, X_train_tfidf, X_test_tfidf):
        print("\nTraining machine learning models...")
        print("Training Naive Bayes model...")
        self.nb_model = MultinomialNB()
        self.nb_model.fit(X_train_tfidf, self.y_train)
        print("Training Logistic Regression model...")
        self.lr_model = LogisticRegression(random_state=42, max_iter=1000)
        self.lr_model.fit(X_train_tfidf, self.y_train)
        print("Models trained successfully!")
        return X_test_tfidf
    
    def evaluate_model(self, model, X_test_tfidf, model_name):
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, pos_label='spam')
        recall = recall_score(self.y_test, y_pred, pos_label='spam')
        f1 = f1_score(self.y_test, y_pred, pos_label='spam')
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred
        }
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n{model_name} Confusion Matrix:")
        print(f"True Negatives (Ham as Ham): {cm[0,0]}")
        print(f"False Positives (Ham as Spam): {cm[0,1]}")
        print(f"False Negatives (Spam as Ham): {cm[1,0]}")
        print(f"True Positives (Spam as Spam): {cm[1,1]}")
    
    def compare_models(self, nb_results, lr_results):
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        comparison_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Naive Bayes': [
                f"{nb_results['accuracy']:.4f}",
                f"{nb_results['precision']:.4f}",
                f"{nb_results['recall']:.4f}",
                f"{nb_results['f1_score']:.4f}"
            ],
            'Logistic Regression': [
                f"{lr_results['accuracy']:.4f}",
                f"{lr_results['precision']:.4f}",
                f"{lr_results['recall']:.4f}",
                f"{lr_results['f1_score']:.4f}"
            ]
        }
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        nb_f1 = nb_results['f1_score']
        lr_f1 = lr_results['f1_score']
        print("\n" + "="*60)
        print("OBSERVATIONS AND ANALYSIS")
        print("="*60)
        if lr_f1 > nb_f1:
            better_model = "Logistic Regression"
            difference = lr_f1 - nb_f1
        else:
            better_model = "Naive Bayes"
            difference = nb_f1 - lr_f1
        print(f"Best performing model: {better_model}")
        print(f"F1-Score difference: {difference:.4f}")
        print("\nDetailed Analysis:")
        print("• Accuracy: Overall correctness of predictions")
        print("• Precision: Of all spam predictions, how many were actually spam")
        print("• Recall: Of all actual spam messages, how many were correctly identified")
        print("• F1-Score: Harmonic mean of precision and recall (balanced metric)")
        print(f"\nNaive Bayes Performance:")
        print(f"• Accuracy: {nb_results['accuracy']:.1%} - Good overall performance")
        print(f"• Precision: {nb_results['precision']:.1%} - {self._interpret_precision(nb_results['precision'])}")
        print(f"• Recall: {nb_results['recall']:.1%} - {self._interpret_recall(nb_results['recall'])}")
        print(f"\nLogistic Regression Performance:")
        print(f"• Accuracy: {lr_results['accuracy']:.1%} - Good overall performance")
        print(f"• Precision: {lr_results['precision']:.1%} - {self._interpret_precision(lr_results['precision'])}")
        print(f"• Recall: {lr_results['recall']:.1%} - {self._interpret_recall(lr_results['recall'])}")
        
    def _interpret_precision(self, precision):
        if precision >= 0.95:
            return "Excellent - Very few false positives"
        elif precision >= 0.90:
            return "Very good - Low false positive rate"
        elif precision >= 0.80:
            return "Good - Acceptable false positive rate"
        else:
            return "Needs improvement - High false positive rate"
    
    def _interpret_recall(self, recall):
        if recall >= 0.95:
            return "Excellent - Catches almost all spam"
        elif recall >= 0.90:
            return "Very good - Catches most spam messages"
        elif recall >= 0.80:
            return "Good - Catches majority of spam"
        else:
            return "Needs improvement - Missing too many spam messages"
    
    def explain_model_choice(self):
        print("\n" + "="*60)
        print("WHY THESE MODELS ARE SUITABLE FOR TEXT CLASSIFICATION")
        print("="*60)
        print("\n1. NAIVE BAYES CLASSIFIER:")
        print("   ✓ Assumes feature independence (works well with TF-IDF)")
        print("   ✓ Handles high-dimensional sparse data efficiently")
        print("   ✓ Fast training and prediction")
        print("   ✓ Works well with small datasets")
        print("   ✓ Naturally handles multi-class problems")
        print("   ✓ Less prone to overfitting")
        print("   ✗ Strong independence assumption may not hold in reality")
        print("\n2. LOGISTIC REGRESSION:")
        print("   ✓ No assumptions about feature independence")
        print("   ✓ Provides probability estimates")
        print("   ✓ Less sensitive to outliers")
        print("   ✓ Can handle feature interactions better")
        print("   ✓ Interpretable coefficients")
        print("   ✓ Regularization helps prevent overfitting")
        print("   ✗ Can be slower with very large feature spaces")
        print("\n3. TF-IDF VECTORIZATION:")
        print("   ✓ Captures word importance in documents")
        print("   ✓ Reduces impact of common words")
        print("   ✓ Creates meaningful numerical features")
        print("   ✓ Handles variable-length text well")
        print("   ✓ Widely used and proven effective")
        
    def predict_new_message(self, message, model_choice='best'):
        processed_message = self.preprocess_text(message)
        message_tfidf = self.vectorizer.transform([processed_message])
        if model_choice == 'nb':
            model = self.nb_model
            model_name = "Naive Bayes"
        elif model_choice == 'lr':
            model = self.lr_model
            model_name = "Logistic Regression"
        else:
            model = self.lr_model
            model_name = "Logistic Regression (Best Model)"
        prediction = model.predict(message_tfidf)[0]
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(message_tfidf)[0]
            confidence = max(probabilities)
            return f"Prediction: {prediction.upper()} (Confidence: {confidence:.2%}) - {model_name}"
        else:
            return f"Prediction: {prediction.upper()} - {model_name}"

def main():
    print("="*60)
    print("TEXT CLASSIFICATION SYSTEM FOR SPAM DETECTION")
    print("="*60)
    classifier = TextClassificationSystem()
    data = classifier.load_data('spam.csv')
    classifier.prepare_data()
    X_train_tfidf, X_test_tfidf = classifier.vectorize_text()
    classifier.train_models(X_train_tfidf, X_test_tfidf)
    print("\nEvaluating models...")
    nb_results = classifier.evaluate_model(classifier.nb_model, X_test_tfidf, "Naive Bayes")
    lr_results = classifier.evaluate_model(classifier.lr_model, X_test_tfidf, "Logistic Regression")
    print("\nDisplaying confusion matrices...")
    classifier.plot_confusion_matrix(classifier.y_test, nb_results['predictions'], "Naive Bayes")
    classifier.plot_confusion_matrix(classifier.y_test, lr_results['predictions'], "Logistic Regression")
    classifier.compare_models(nb_results, lr_results)
    classifier.explain_model_choice()
    print("\n" + "="*60)
    print("TESTING WITH SAMPLE MESSAGES")
    print("="*60)
    sample_messages = [
        "Congratulations! You've won a $1000 gift card. Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account will be suspended. Call 123-456-7890 immediately!",
        "Thanks for the great presentation today. Looking forward to working together."
    ]
    for message in sample_messages:
        print(f"\nMessage: '{message}'")
        print(classifier.predict_new_message(message, 'nb'))
        print(classifier.predict_new_message(message, 'lr'))
    print("\n" + "="*60)
    print("SYSTEM READY FOR USE!")
    print("="*60)
    return classifier

if __name__ == "__main__":
    classifier = main()
    print("\nWould you like to test with your own messages? (y/n)")
    print("Interactive mode skipped for demonstration.")
    print("\nTo test with custom messages, use:")
    print("classifier.predict_new_message('Your message here', 'lr')")