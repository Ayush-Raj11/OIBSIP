# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (replace 'spam_data.csv' with your dataset)
data = pd.read_csv('spam_data.csv')

# Data Preprocessing
# Clean and preprocess text data (remove HTML tags, punctuation, etc.)

# Feature Extraction (using TF-IDF vectorization)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['email_text'])
y = data['label']  # Assuming 'label' column denotes spam or non-spam

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
