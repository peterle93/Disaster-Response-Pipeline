# import libraries
import sys
import re
import numpy as np
import pandas as pd
import pickle
from time import time

# import nltk libraries
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# import sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
from sklearn.multioutput import MultiOutputClassifier


from sqlalchemy import create_engine


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('disaster_table', con=engine)

    X = df['message'] # reads message column
    y = df.iloc[:,4:] # reads all rows, all column starting at [4] ->
    pass


def tokenize(text):
        stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove the stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]    
    
    return tokens
    pass


def build_model():
    # Pipeline 3 with AdaboostClf
    pipeline3 = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer(smooth_idf=False)),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))

    parameters3 = {'clf__estimator__n_estimators': [50, 100], 
                'clf__estimator__learning_rate': [1, 2]}


    cv3 = GridSearchCV(estimator=pipeline3, param_grid=parameters3, verbose=3)
    pass


def evaluate_model(model, X_test, y_test, category_names):
    cv3.fit(X_train, y_train)
    y_pred3 = cv3.predict(X_test)
    print(classification_report(y_test, y_pred3, target_names=y.columns))
    pass


def save_model(model, model_filepath):
    with open('classifier.pkl', 'wb') as file:
    pickle.dump(cv3, file)
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()