import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

print('='*60)
print('TEXT CLASSIFICATION SYSTEM DEMO')
print('='*60)

print('Loading NLTK data...')
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    print('✅ NLTK data already available')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('✅ NLTK data downloaded')

print('\n📊 Loading dataset...')
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data.iloc[:, :2]
data.columns = ['label', 'message']
data = data.dropna()

print(f'✅ Dataset loaded successfully!')
print(f'   Total messages: {len(data)}')
print(f'   Spam messages: {len(data[data["label"] == "spam"])}')
print(f'   Ham messages: {len(data[data["label"] == "ham"])}')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

print('\n🔧 Preprocessing text data...')
data['processed_message'] = data['message'].apply(preprocess_text)

X = data['processed_message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f'   Training samples: {len(X_train)}')
print(f'   Testing samples: {len(X_test)}')

print('\n🔢 Creating TF-IDF features...')
vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f'   TF-IDF matrix shape: {X_train_tfidf.shape}')

print('\n🤖 Training models...')
print('   Training Naive Bayes...')
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

print('   Training Logistic Regression...')
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

print('\n📈 Evaluating models...')
nb_pred = nb_model.predict(X_test_tfidf)
lr_pred = lr_model.predict(X_test_tfidf)

nb_acc = accuracy_score(y_test, nb_pred)
nb_prec = precision_score(y_test, nb_pred, pos_label='spam')
nb_rec = recall_score(y_test, nb_pred, pos_label='spam')
nb_f1 = f1_score(y_test, nb_pred, pos_label='spam')

lr_acc = accuracy_score(y_test, lr_pred)
lr_prec = precision_score(y_test, lr_pred, pos_label='spam')
lr_rec = recall_score(y_test, lr_pred, pos_label='spam')
lr_f1 = f1_score(y_test, lr_pred, pos_label='spam')

print('\n' + '='*60)
print('MODEL PERFORMANCE RESULTS')
print('='*60)
print(f'{"Metric":<12} {"Naive Bayes":<15} {"Logistic Regression":<20}')
print('-' * 50)
print(f'{"Accuracy":<12} {nb_acc:<15.4f} {lr_acc:<20.4f}')
print(f'{"Precision":<12} {nb_prec:<15.4f} {lr_prec:<20.4f}')
print(f'{"Recall":<12} {nb_rec:<15.4f} {lr_rec:<20.4f}')
print(f'{"F1-Score":<12} {nb_f1:<15.4f} {lr_f1:<20.4f}')

nb_cm = confusion_matrix(y_test, nb_pred)
lr_cm = confusion_matrix(y_test, lr_pred)

print('\n📊 Confusion Matrices:')
print('\nNaive Bayes:')
print(f'   True Negatives (Ham as Ham): {nb_cm[0,0]}')
print(f'   False Positives (Ham as Spam): {nb_cm[0,1]}')
print(f'   False Negatives (Spam as Ham): {nb_cm[1,0]}')
print(f'   True Positives (Spam as Spam): {nb_cm[1,1]}')

print('\nLogistic Regression:')
print(f'   True Negatives (Ham as Ham): {lr_cm[0,0]}')
print(f'   False Positives (Ham as Spam): {lr_cm[0,1]}')
print(f'   False Negatives (Spam as Ham): {lr_cm[1,0]}')
print(f'   True Positives (Spam as Spam): {lr_cm[1,1]}')

better_model = 'Naive Bayes' if nb_f1 > lr_f1 else 'Logistic Regression'
print(f'\n🏆 Best performing model: {better_model}')
print(f'   F1-Score difference: {abs(nb_f1 - lr_f1):.4f}')

print('\n' + '='*60)
print('TESTING WITH SAMPLE MESSAGES')
print('='*60)

samples = [
    "Congratulations! You've won a $1000 gift card. Click here now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account will be suspended. Call immediately!",
    "Thanks for the great presentation today."
]

best_model = nb_model if nb_f1 > lr_f1 else lr_model
model_name = 'Naive Bayes' if nb_f1 > lr_f1 else 'Logistic Regression'

for i, msg in enumerate(samples, 1):
    processed = preprocess_text(msg)
    msg_tfidf = vectorizer.transform([processed])
    pred = best_model.predict(msg_tfidf)[0]
    prob = best_model.predict_proba(msg_tfidf)[0]
    conf = max(prob)
    
    print(f'\n{i}. Message: "{msg[:50]}{"..." if len(msg) > 50 else ""}"')
    print(f'   Prediction: {pred.upper()} (Confidence: {conf:.1%}) - {model_name}')

print('\n' + '='*60)
print('✅ TEXT CLASSIFICATION SYSTEM COMPLETED!')
print('='*60)
print('🚀 Key Achievements:')
print(f'   • Processed {len(data)} SMS messages')
print(f'   • Achieved {max(nb_acc, lr_acc):.1%} accuracy')
print(f'   • Best model: {better_model}')
print('   • Successfully classified sample messages')
print('\n💡 Why these models work for text classification:')
print('   • Naive Bayes: Fast, assumes word independence, great for text')
print('   • Logistic Regression: Flexible, handles feature interactions')
print('   • TF-IDF: Captures word importance, reduces noise')