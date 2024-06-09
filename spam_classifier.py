import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import string
from nltk.corpus import stopwords

# Load the dataset
data = pd.read_csv('emails.csv')

# Display the first few rows of the dataset
print(data.head())

# Data preprocessing
data['length'] = data['text'].apply(len)

# Plot the distribution of email lengths
data['length'].plot(bins=100, kind='hist')
plt.show()

# Data cleaning function
def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

# Convert text into numerical data
vectorizer = CountVectorizer(analyzer=text_process)
X = vectorizer.fit_transform(data['text'])

# Split the dataset into features and target variable
y = data['spam'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
