import numpy as np
import pandas as pd
import pickle
from utils import preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, classification_report,confusion_matrix
# Load dataset
def load_data(filepath):
    df = pd.read_csv(filepath, encoding='latin1')
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
    df = df.drop_duplicates(keep='first')
    return df

# Text Vectorization
def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()
    return X_train, X_test, vectorizer

# Train and Evaluate Model
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = BernoulliNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Confusion Matrix :\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model

# Main Function
def main():
    filepath = 'spam.csv'
    df = load_data(filepath)
    df['transformed_text'] = df['text'].apply(preprocess_text)  # Use imported function
    
    # Encode target variable
    encoder = LabelEncoder()
    df['target'] = encoder.fit_transform(df['target'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['transformed_text'], df['target'], test_size=0.2, random_state=2, stratify=df['target']
    )
    
    # Vectorization
    X_train, X_test, vectorizer = vectorize_text(X_train, X_test)
    
    # Train and evaluate model
    model = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    # Save model and vectorizer
    pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
    pickle.dump(model, open('model.pkl', 'wb'))
    print("Model and vectorizer saved successfully!")

if __name__ == "__main__":
    main()
