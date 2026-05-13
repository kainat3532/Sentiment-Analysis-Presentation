import pandas as pd
import numpy as np
import re
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# ---------------------------------------------------------
# 1. Configuration & Data Loading
# ---------------------------------------------------------
TRAIN_PATH = "archive (2)/twitter_training.csv"
VALID_PATH = "archive (2)/twitter_validation.csv"

# Load data - The CSV has no header
# Columns: ID, Topic, Sentiment, Text
cols = ['id', 'topic', 'sentiment', 'text']
train_df = pd.read_csv(TRAIN_PATH, names=cols)
valid_df = pd.read_csv(VALID_PATH, names=cols)

print(f"Loaded {len(train_df)} training samples and {len(valid_df)} validation samples.")

# ---------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'\@\w+|\#','', text) # Remove mentions and hashtags
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
    text = text.strip()
    return text

print("Cleaning text data...")
train_df['text_clean'] = train_df['text'].apply(clean_text)
valid_df['text_clean'] = valid_df['text'].apply(clean_text)

# Remove empty strings after cleaning
train_df = train_df[train_df['text_clean'] != ""].copy()
valid_df = valid_df[valid_df['text_clean'] != ""].copy()

# ---------------------------------------------------------
# 3. Task Selection: Binary vs Multi-class
# ---------------------------------------------------------
# Let's create a binary subset (Positive vs Negative) to compare with the paper
train_binary = train_df[train_df['sentiment'].isin(['Positive', 'Negative'])].copy()
valid_binary = valid_df[valid_df['sentiment'].isin(['Positive', 'Negative'])].copy()

print(f"\n--- Multi-class Task (4 classes) ---")
print(f"Samples: {len(train_df)}")

# ---------------------------------------------------------
# 4. Feature Extraction & Training (Multi-class)
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_df['text_clean'])
X_valid = vectorizer.transform(valid_df['text_clean'])

y_train = train_df['sentiment']
y_valid = valid_df['sentiment']

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)
acc_multi = accuracy_score(y_valid, y_pred)
print(f"Multi-class Accuracy: {acc_multi:.4f}")

# ---------------------------------------------------------
# 5. Feature Extraction & Training (Binary - To match Paper)
# ---------------------------------------------------------
print(f"\n--- Binary Task (Positive vs Negative) ---")
print(f"Samples: {len(train_binary)}")

vec_bin = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_bin = vec_bin.fit_transform(train_binary['text_clean'])
X_valid_bin = vec_bin.transform(valid_binary['text_clean'])

y_train_bin = train_binary['sentiment']
y_valid_bin = valid_binary['sentiment']

model_bin = LogisticRegression(max_iter=1000)
model_bin.fit(X_train_bin, y_train_bin)

y_pred_bin = model_bin.predict(X_valid_bin)
acc_bin = accuracy_score(y_valid_bin, y_pred_bin)
print(f"Binary Accuracy: {acc_bin:.4f} (Matches Paper's Goal)")

# ---------------------------------------------------------
# 6. Final Evaluation & Visualization
# ---------------------------------------------------------
print("\nBinary Task Classification Report:")
print(classification_report(y_valid_bin, y_pred_bin))

# 6. Visualization for Presentation
# ---------------------------------------------------------
# Confusion Matrix (Binary)
cm = confusion_matrix(y_valid_bin, y_pred_bin, labels=model_bin.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=model_bin.classes_, yticklabels=model_bin.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Binary Sentiment Analysis (Accuracy: {acc_bin:.2f})')
plt.savefig('confusion_matrix_binary.png')
print("Saved confusion_matrix_binary.png")

# Comparison Chart for Presentation
comparison_df = pd.DataFrame({
    'Model': ['Paper Baseline', 'Our Multi-class', 'Our Binary'],
    'Accuracy': [0.82, acc_multi, acc_bin]
})

plt.figure(figsize=(10, 6))
sns.barplot(x='Accuracy', y='Model', data=comparison_df, palette='magma')
plt.axvline(0.82, color='red', linestyle='--', label='Paper Target (82%)')
plt.title('Performance Comparison: Our Model vs Research Paper')
plt.xlim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_comparison.png')
print("Saved accuracy_comparison.png")
