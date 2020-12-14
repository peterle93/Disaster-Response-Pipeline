# Import libraries
import sys
import re
import numpy as np
import pandas as pd
import pickle

# Import nltk libraries
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
    ''' 
    function loads data from sql database
        database_filepath: path of database
    Returns:
        X: message column
        Y: categories
    category_names: names of categories
    '''    
    # load data from database
    table_name = 'disaster_table'
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table(table_name, con=engine)

    X = df['message'] # reads message column
    Y = df.iloc[:,4:] # reads all rows, all column starting at [4] ->
    category_names = Y.columns
   
   return X, Y, category_names

def tokenize(text):
    '''
    function to tokenize the text
        Input: text
        Output: list of clean tokens 
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Detect urls with url_regex
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # Normalizes and tokenizes (removes punctuations and converts to lowercase)
    tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]

    # lemmatize and remove stop words
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stopwords.words("english"):
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    ML pipeline that takes the message column and passes it through the classfier
    to place into the most accurate category of the 36 in dataset
    Returns:
        cv3: the machine learning model
    '''
    # Best pipeline settings and parameters
    pipeline3 = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))])
    
	parameters3 = {'clf__estimator__n_estimators': [10, 40], 
            	'clf__estimator__learning_rate': [1,2]}

    cv3 = GridSearchCV(pipeline, param_grid=parameters, verbose=3)    
    return cv3

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Provides a classifcation report (precision, recall and f1 score) for all categories
        model: model to be evaluated
        X_test: test dataset (messages)
        Y_test: categories for messages of X_test
        category_names: list of categories for messages for classification
    '''
    Y_pred3 = model.predict(X_test)

    # cv3 best parameters
    print(classification_report(Y_test, Y_pred3, target_names=Y.columns)) 
    pass

def save_model(model, model_filepath):
    '''
    Saves best parameters of ML model as pickle file
        model: ML pipeline model
        model_filepath: file path for saving model
    '''
    pickle.dump(cv3, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
   