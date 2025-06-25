#Part 2 - Political Speech Classification to predict the political party from a speech

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#import sklearn pip install scikit-learn nltk pandas
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score

#download the nltk resources
nltk.download('stopwords')
nltk.download('wordnet')

#Part 2A - Data loading + cleaning
def load_and_clean_data():
    # Load the csv file
    from pathlib import Path
    path = Path.cwd() / "hansard40000.csv"
    df = pd.read_csv(path)

    #rename labour coop to just labour
    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')

    # top 4 parties and remove speaker
    df = df[df["party"] != "Speaker"]
    top_parties = df['party'].value_counts().nlargest(4).index 
    df = df[df['party'].isin(top_parties)]

    #rows where the speech_class is 'speech'
    df = df[df['speech_class'] == 'Speech'] #capital S

    #get rid of shorter speeches
    df = df[df['speech'].str.len() > 1000] #nummber 1000 charaters

    #now print the speeches in rows/columns
    print("Clean DataFrame:", df.shape) 

    return df

#Part 2B - Vecotrization + train/test split

def vectorize_and_split_speech_data(df):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features = 3000) #3000 top words
    
    #now transform speeches into number matrices
    X = vectorizer.fit_transform(df['speech'])
    Y = df['party'] #the acc prediction

    #split data 80/20 for training/testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size= 0.2,
                                                        random_state = 26,
                                                        stratify=Y)
    
    return X_train, X_test, Y_train, Y_test, vectorizer                                                 


#Part C: Train the classifier and evaluate

def train_classifier_and_evaluate(X_train, X_test, Y_train, Y_test):
    #this trains the model + gives us the performance reports

    all_classifiers = {'random forest': RandomForestClassifier(n_estimators = 300, random_state = 26),
                       'SVM': SVC(kernel='linear', random_state=26)}

    #rf.fit(X_train, Y_train)
    #rf_pred = rf.predict(X_test)
    #print(classification_report(Y_test, rf_pred, digits=3))

    #second model = SVM
    #svm = SVC(kernel='linear', random_state=26)
    #svm.fit(X_train, Y_train)
    #svm_pred = svm.predict(X_test)
    #print("SVM performance")

    for model, classifer in all_classifiers.items():
        classifer.fit(X_train, Y_train)
        model_prediction = classifer.predict(X_test)
        print(f"Model Results for {model}: ")
        print(classification_report(Y_test, model_prediction, digits=3))
        print(f"Accuracy Score{accuracy_score(Y_test, model_prediction, digits=3)}")
        print(f"F1 Score{f1_score(Y_test, model_prediction, average="macro")}")



#Part D: Improve with the N-Grams

def use_n_grams(df):
    n_gram_vectorizer = TfidfVectorizer(
        stop_words= 'english',
        ngram_range=(1,3),
        max_features=3000
    )

    X_ngrams = n_gram_vectorizer.fit_transform(df['speech'])
    Y = df['party']

    X_train_ngrams, X_test_ngrams, Y_train_ngrams, Y_test_ngrams = train_test_split(
        X_ngrams, Y,
        test_size=0.2,
        random_state=26,
        stratify=Y
    )

    #check model again with ngrams
    print("Performance with ngrams")
    #train_classifier_and_evaluate(X_train_ngrams, X_test_ngrams, Y_train_ngrams, Y_test_ngrams)

    all_classifiers = {'random forest': RandomForestClassifier(n_estimators = 300, random_state = 26),
                       'SVM': SVC(kernel='linear', random_state=26)}


    for model, classifer in all_classifiers.items():
        classifer.fit(X_train_ngrams, Y_train_ngrams)
        model_prediction = classifer.predict(X_test_ngrams)
        print(f"Model Results for {model}: ")
        print(classification_report(Y_test_ngrams, model_prediction, digits=3))
        print(f"NGrams Accuracy Score{accuracy_score(Y_test_ngrams, model_prediction, digits=3)}")
        print(f"NGrams F1 Score{f1_score(Y_test_ngrams, model_prediction, average='macro')}")

    #Part E: custom tokenizer for better performance

    #make it all lowker case
def advanced_tokenizer(text):
    text = text.lower()

    #get rid of all special characters (?) + keep the spaces
    text = re.sub(r'[^a-z\s]', '', text) #assitance from copilot for this line

    words = text.split()

    stopwords = set(stopwords.words('english'))
    words = [w for w in words if w not in stopwords]

    #now lemmatize and convert it to root words like in class
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    return words




def custom_tokenizer_evaluation(df):
    #check the modesl with smart tokenizer/custom cleaning

    custom_vectorizer = TfidfVectorizer(
        stop_words = 'english',
        tokenizer = advanced_tokenizer,
        max_features = 3000
    )

    X_custom = custom_vectorizer.fit_transform(df['speech'])
    Y = df['party']

    X_train_custom, X_test_custom, Y_train_custom, Y_test_custom = train_test_split(
        X_custom, Y,
        test_size = 0.2,
        random_state = 26,
        stratify=Y
    )

    #Custom tokeniser for all classiferrs

    all_classifiers = {'random forest': RandomForestClassifier(n_estimators = 300, random_state = 26),
                        'SVM': SVC(kernel='linear', random_state=26)}


    for model, classifer in all_classifiers.items():
        classifer.fit(X_train_custom, Y_train_custom)
        model_prediction = classifer.predict(X_test_custom)
        print(f"Custom Model Results for {model}: ")
        print(classification_report(Y_test_custom, model_prediction, digits=3))
        print(f"Custom Accuracy Score{accuracy_score(Y_test_custom, model_prediction, digits=3)}")
        print(f"Custom F1 Score{f1_score(Y_test_custom, model_prediction, average='macro')}")


#MAIN

if __name__ == "__main__": #copied from partone.py becuase of errors when I typed it up

    #A: first load/clean data
    df = load_and_clean_data()

    #B: vectorization + split 
    X_train, X_test, Y_train, Y_test, vectorizer = vectorize_and_split_speech_data(df)

    #C: baseline models check
    train_classifier_and_evaluate(X_train, X_test, Y_train, Y_test)

    #D: nngrams check
    use_n_grams(df)

    #E: custom tokenizer check final
    custom_tokenizer_evaluation(df)