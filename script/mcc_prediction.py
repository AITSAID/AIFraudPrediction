# Filename: mcc_prediction_mcc_codes_enhanced.py

import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
import sys
import warnings

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, ISRIStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Ensure NLTK packages are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def main():
    print("Starting enhanced MCC prediction script...")

    # 1. Load the Dataset
    print("Loading dataset...")
    try:
        data = pd.read_csv('sysplexData1.csv', encoding='utf-8')  # Ensure UTF-8 encoding
    except FileNotFoundError:
        print("Error: Dataset file 'sysplexData1.csv' not found.")
        sys.exit(1)

    # Check if required columns are present
    if 'post' not in data.columns or 'mcc' not in data.columns:
        print("Error: Dataset must contain 'post' and 'mcc' columns.")
        sys.exit(1)

    print(f"Dataset loaded successfully. Number of records: {len(data)}")

    # 2. Data Preprocessing
    print("Preprocessing text data...")
    tqdm.pandas(desc="Text Preprocessing")
    data['cleaned_post'] = data['post'].progress_apply(preprocess_text)
    print("Text preprocessing completed.")

    # 3. Feature Extraction
    print("Extracting features using TF-IDF vectorization...")
    X_features, tfidf_vectorizer = extract_features(data['cleaned_post'])
    print(f"Feature extraction completed. Number of features: {X_features.shape[1]}")

    # 4. Encode the Labels
    print("Encoding MCC labels...")
    y_labels, label_encoder = encode_labels(data['mcc'])
    print(f"Labels encoded. Number of classes: {len(label_encoder.classes_)}")

    # Analyze class distribution
    unique, counts = np.unique(y_labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print("Class counts:", class_counts)

    # Handle classes with insufficient samples
    min_samples = 5  # Increase the minimum number of samples for robustness
    valid_classes = [cls for cls, count in class_counts.items() if count >= min_samples]
    if len(valid_classes) < len(label_encoder.classes_):
        print(f"Removing classes with less than {min_samples} samples.")
        # Create a mask for valid samples
        valid_mask = np.isin(y_labels, valid_classes)
        # Filter your dataset
        X_features = X_features[valid_mask]
        y_labels = y_labels[valid_mask]
        data = data.iloc[valid_mask].reset_index(drop=True)
        # Update label encoder
        y_labels, label_encoder = encode_labels(label_encoder.inverse_transform(y_labels))
        print(f"Updated number of classes: {len(label_encoder.classes_)}")

    # 5. Split the Dataset (include post data for test set)
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test, data_train, data_test = train_test_split(
        X_features, y_labels, data, test_size=0.2, random_state=42, stratify=y_labels
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # 6. Train the Model
    print("Training the model with cross-validation...")
    model = train_model(X_train, y_train)
    print("Model training completed.")

    # 7. Evaluate the Model (pass test post data)
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test, label_encoder, data_test)
    print("Model evaluation completed.")

    # 8. Test the Model with New Input
    print("You can now test the model with a new input.")
    test_new_input(model, tfidf_vectorizer, label_encoder)

def preprocess_text(text):
    # Convert to string and lowercase
    text = str(text).lower()

    # Remove URLs, mentions, hashtags, extra spaces
    text = re.sub(r'http\S+|@\S+|#\S+', '', text)
    text = re.sub(r'\s+', ' ', text.strip())  # Remove extra spaces

    # Remove special characters but keep Arabic and English characters
    text = re.sub(r'[^\w\s\u0600-\u06FF]', '', text)

    # Detect language (simple heuristic based on character range)
    if re.search(r'[\u0600-\u06FF]', text):
        # Arabic text
        stemmer = ISRIStemmer()
        stop_words = set(stopwords.words('arabic'))

        # Tokenize using custom Arabic tokenizer
        words = arabic_tokenizer(text)
    else:
        # Assume English
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))

        # Tokenize the text
        words = word_tokenize(text)

    # Remove stopwords and perform stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]

    # Join the words back into one string
    cleaned_text = ' '.join(words)

    return cleaned_text

def arabic_tokenizer(text):
    # Define a regex pattern for Arabic words
    pattern = r'[\u0600-\u06FF]+'
    tokenizer = RegexpTokenizer(pattern)
    tokens = tokenizer.tokenize(text)
    return tokens

def extract_features(text_data):
    tfidf_vectorizer = TfidfVectorizer(max_features=7000)  # Increased features for better representation
    X_features = tfidf_vectorizer.fit_transform(text_data)
    return X_features, tfidf_vectorizer

def encode_labels(labels):
    # Ensure labels are strings to handle leading zeros
    labels = labels.astype(str)
    label_encoder = LabelEncoder()
    y_labels = label_encoder.fit_transform(labels)
    return y_labels, label_encoder

def train_model(X_train, y_train):
    # Hyperparameter tuning using Grid Search
    parameters = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}  # Expanded search space
    mnb = MultinomialNB()
    grid_search = GridSearchCV(mnb, parameters, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print("Best parameters found:", grid_search.best_params_)

    # Cross-validation performance
    scores = cross_val_score(best_model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {scores}")
    print(f"Average cross-validation accuracy: {np.mean(scores):.4f}")

    return best_model

def evaluate_model(model, X_test, y_test, label_encoder, data_test):
    y_pred = model.predict(X_test)
    labels = np.arange(len(label_encoder.classes_))
    target_names = label_encoder.classes_  # MCC codes as strings

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, labels=labels, target_names=target_names, zero_division=0
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plot_confusion_matrix(cm, target_names)

    # Log failed predictions (misclassifications)
    misclassified_indices = np.where(y_test != y_pred)[0]
    misclassified_posts = data_test.iloc[misclassified_indices]

    print(f"\nNumber of misclassified posts: {len(misclassified_posts)}")

    # Display the failed posts with their true and predicted MCCs
    for idx in misclassified_indices:
        true_label = label_encoder.inverse_transform([y_test[idx]])[0]
        predicted_label = label_encoder.inverse_transform([y_pred[idx]])[0]
        post = data_test['post'].iloc[idx]
        print(f"Post: {post}")
        print(f"True MCC: {true_label}, Predicted MCC: {predicted_label}")
        print("-" * 80)

    # Optionally save misclassified posts to a file
    misclassified_posts['True MCC'] = label_encoder.inverse_transform(y_test[misclassified_indices])
    misclassified_posts['Predicted MCC'] = label_encoder.inverse_transform(y_pred[misclassified_indices])
    misclassified_posts.to_csv('misclassified_posts.csv', index=False)

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j, i, format(cm[i, j], 'd'),
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=8
        )

    plt.tight_layout()
    plt.ylabel('True MCC')
    plt.xlabel('Predicted MCC')
    plt.show()

def test_new_input(model, tfidf_vectorizer, label_encoder):
    while True:
        new_post = input("\nEnter a new social media post (or type 'exit' to quit): ")
        if new_post.lower() == 'exit':
            break

        cleaned_post = preprocess_text(new_post)
        X_new = tfidf_vectorizer.transform([cleaned_post])
        y_pred_prob = model.predict_proba(X_new)[0]
        mcc_labels = label_encoder.classes_  # MCC codes as strings

        # Sort probabilities and labels in descending order
        sorted_indices = np.argsort(y_pred_prob)[::-1]
        sorted_probs = y_pred_prob[sorted_indices]
        sorted_labels = mcc_labels[sorted_indices]

        # Display top 5 predicted MCCs
        print("\nTop predicted MCCs:")
        for i in range(min(5, len(sorted_labels))):
            print(f"MCC {sorted_labels[i]}: Probability {sorted_probs[i]:.4f}")

        # Optionally, plot the spider chart
        plot_spider_chart(y_pred_prob, mcc_labels)

def plot_spider_chart(probabilities, mcc_labels):
    # Plot only top N MCCs to make the chart readable
    N = min(5, len(mcc_labels))  # Adjust N as needed
    sorted_indices = np.argsort(probabilities)[::-1]
    top_indices = sorted_indices[:N]
    top_probs = probabilities[top_indices]
    top_labels = mcc_labels[top_indices]

    # Prepare data for spider chart
    categories = list(top_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    top_probs = np.concatenate((top_probs, [top_probs[0]]))  # Close the circle
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, top_probs, 'o-', linewidth=2)
    ax.fill(angles, top_probs, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_title('Top MCC Predictions', y=1.1)
    ax.set_ylim(0, 1)
    plt.show()

if __name__ == '__main__':
    main()
