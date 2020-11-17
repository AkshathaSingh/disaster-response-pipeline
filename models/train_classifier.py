import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


import re
import nltk
nltk.download(['stopwords','wordnet','punkt'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def load_data(database_filepath):
    '''
    Load the data from the database
    INPUT: 
          database_filepath - path to SQL database where the data is stored
    OUPUT:
          X - independent features (messages data)
          Y - dependent features (categories data)
          Colunm names - names of the 36 features
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Disaster_messages', engine)
    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis=1)
    column_names = list(df.columns[4:])
    
    return X, Y, column_names


def tokenize(text):
    '''
    clean the text and generate words
    INPUT:
          text - the text which needs to be processed
    OUTPUT:
           processed text as words
    '''
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


def build_model():
    '''
    Build a ML model with tf-idf, countvectorizer, random forest and grid search
    INPUT:
          None
    OUTPUT:
           ML model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'tfidf__use_idf': (True, False),
              'clf__estimator__n_estimators': [30, 60], 
              } 
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate model performance using test data
    INPUT:
          model - trained model to be evaluated
          X_test - features of the test dataset
          y_test - independent or target features of test dataset
          category_names - names of the features
    OUTPUT:
          classification report by features
    '''
    Y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))


def save_model(model, model_filepath):
    '''
    save the model as a pickle file
    INPUT:
          model - trained model to be saved
          model_filepath - molde file path
    OUTPUT:
           pickle file of the model
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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