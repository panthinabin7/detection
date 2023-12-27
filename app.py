from flask import Flask, render_template, request
import joblib
import requests
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load the trained model and vectorizer

clf = joblib.load('terrorism_detection_model.pkl') 
vectorizer = joblib.load('trained_vectorizer.pkl') 

# Define stop words and lemmatizer
stop_words = stopwords.words('english')
stop_words.extend(['wa','two','one','iraq','false','frequent','near','baghdad','three','al','people','group','four','responsibiliity','reported','district','others','patrol','day','car','claimed','thailand','device',''])
lemmatizer = WordNetLemmatizer()

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for handling form submission
@app.route('/result', methods=['POST'])
def check():
    # Get URL input from form
    url = request.form['url']

    # Scrape text from website
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # extract the text from each tag
    title = soup.title.text if soup.title else ""
    h1 = soup.h1.text if soup.h1 else ""
    h2 = soup.h2.text if soup.h2 else ""
    h3 = soup.h3.text if soup.h3 else ""
    h4 = soup.h4.text if soup.h4 else ""
    h5 = soup.h5.text if soup.h5 else ""
    p_list =  [p.text for p in soup.find_all('p')] 

    # combine all the text into a single list
    all_text = [title, h1, h2, h3,h4,h5]
    all_text.extend(p_list)
    all_text = ' '.join(str(s) for s in all_text)

    # Preprocess the scraped text
    review = re.sub('[^a-zA-Z]', ' ', all_text)
    review = review.lower()
    review = review.split() # convert sentence to list of words
    review = [lemmatizer.lemmatize(word) for word in review] # lemmatize words in list
    review = [word for word in review if not word in set(stop_words)] # remove stop words

    if isinstance(review, list):
      review = ' '.join(review) 

    # Create BoW representation of the text
    X = vectorizer.transform([review])

    # Predict whether the text is terrorism-related or not
    label = clf.predict(X)

    if label == 'terrorism':
     suspect_sentences = []
     sentences = all_text.split('.')
     for sentence in sentences:
        sentence = sentence.strip()
        # Only consider sentences that have at least one word
        # and that are not already in the suspect_sentences list
        if len(sentence) > 0 and sentence not in suspect_sentences:
            # Create BoW representation of the sentence
            sentence_X = vectorizer.transform([sentence])
            # Predict whether the sentence is related to terrorism or not
            sentence_label = clf.predict(sentence_X)
            if sentence_label == 'terrorism':
                suspect_sentences.append(sentence)
        if len(suspect_sentences) > 0:
         print('The text is related to terrorism.')
        suspect_scores = []
        for sentence in suspect_sentences:
            # Create BoW representation of the sentence
            sentence_X = vectorizer.transform([sentence])
            # Get the probability score of the sentence being related to terrorism
            sentence_score = clf.predict_proba(sentence_X)[0][1]
            suspect_scores.append((sentence, sentence_score))
        # Sort the suspect sentences by their probability score
        sorted_suspect_sentences = sorted(suspect_scores, key=lambda x: x[1], reverse=True)
        print('The top five sentences responsible for spreading terrorism are:')
        for sentence, score in sorted_suspect_sentences[:5]:
            print(f'{sentence}  ({score:.2f} )\n')
        result = 'Terrorism related content is detected and related sentences are:'
        # prediction = ', '.join([s[0] for s in sorted_suspect_sentences[:5]])
        prediction = ', '.join([f'{s[0]} ({s[1]:.2f})' for s in sorted_suspect_sentences[:5]])

    else:
     result = "Terrorism related content is not detected!."
     prediction="Not spreading terrorism"

    return render_template('result.html', url=url, prediction=prediction, result=result)


    

    

if __name__ == '__main__':
    app.run(debug=True)

