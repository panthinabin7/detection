import warnings
import sklearn
import numpy as np
import pandas as pd
import re
import nltk
import joblib
from sklearn.exceptions import DataConversionWarning
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask


warnings.filterwarnings(action='ignore', category=DataConversionWarning)
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv("D:\\desktop\\terrorism dataset\\RAND_Database_of_Worldwide_Terrorism_Incidents.csv", encoding='ISO-8859-1')
df = df.dropna(subset=['Description']) #removes rows or columns that have missing values by default.

# Preprocessing
stop_words = stopwords.words('english')
stop_words.extend(['wa','two','one','iraq','false','frequent','near','baghdad','three','al','people','group','four','responsibiliity','reported','district','others','patrol','day','car','claimed','thailand','device',''])
lemmatizer = WordNetLemmatizer()

# Use sentiment analyzer to assign sentiment scores to each incident description
sia = SentimentIntensityAnalyzer()

def get_sentiment_scores(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['neg'], sentiment['neu'], sentiment['pos']

df['Negative_Score'], df['Neutral_Score'], df['Positive_Score'] = zip(*df['Description'].apply(get_sentiment_scores))

# Label each incident as 'terrorism' or 'not terrorism' based on a threshold for the negative sentiment score
threshold = 0.2
df['Label'] = np.where(df['Negative_Score'] >= threshold, 'terrorism', 'not terrorism')
corpus = []


for i in range(len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['Description'][i])
    review = review.lower().split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stop_words)]
    review = ' '.join(review)
    corpus.append(review)

# Feature extraction
cv = TfidfVectorizer(ngram_range=(1,1))
corpus1 = cv.fit_transform(corpus)
avg = corpus1.mean(axis=0)
avg = pd.DataFrame(avg, columns=cv.get_feature_names_out())
avg = avg.T
avg = avg.rename(columns={0:'score'})
avg['word'] = avg.index
avg = avg.sort_values('score',ascending=False)
avg = avg[:500]

cv_bigram = TfidfVectorizer(ngram_range=(3,3))
corpus1_bigram = cv_bigram.fit_transform(corpus)
avg_bigram = corpus1_bigram.mean(axis=0)
avg_bigram = pd.DataFrame(avg_bigram, columns=cv_bigram.get_feature_names_out())
avg_bigram = avg_bigram.T
avg_bigram = avg_bigram.rename(columns={0:'score'})
avg_bigram['word'] = avg_bigram.index
avg_bigram = avg_bigram.sort_values('score',ascending=False)

unigrams_list= avg['word'].tolist()
bigrams_list = avg_bigram['word'].tolist()

def convert(lst):
    return([item.split() for item in lst])

bigrams_split = convert(bigrams_list)

check = pd.DataFrame(columns=['topic','subtopic'])

for i in unigrams_list:
    counter = 0
    for j in bigrams_split:
        if counter < 5 and (i == j[0] or i == j[1] or i == j[2]):
            bigram_words = ' '.join(j)
            check = pd.concat([check, pd.concat([pd.Series(i,name='topic'), pd.Series(bigram_words,name='subtopic')],axis=1)],axis=0)
            counter += 1

# Group subtopics by topic and concatenate into a comma-separated list
check_new = check.groupby(['topic'], as_index=False, sort=False).agg({'subtopic':', '.join})

#Transform text data into numerical features
vectorizer = TfidfVectorizer(vocabulary=unigrams_list + bigrams_list)
X = vectorizer.fit_transform(corpus)
y = df['Label']

# Save trained vectorizer
joblib.dump(vectorizer, 'trained_vectorizer.pkl')

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train model using SVM algorithm
clf = SVC(kernel='linear', C=1, random_state=42,probability=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#evaluate the model on the test data 
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Assuming y_test contains the true labels and y_pred contains the predicted labels
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the precision, recall, and F1 score
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

#Save trained model
joblib.dump(clf, 'terrorism_detection_model.pkl')



