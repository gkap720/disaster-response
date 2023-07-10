import sys
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import pandas as pd
import re
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    """
    Loads data from a given database
    Args:
        database_filepath: path to an SQLite database
    Returns:
        X: the unaltered text messages
        y: all of the categories related to those messages
        columns: list of all of the category names
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("message", engine)
    X = df["message"]
    y = df.iloc[:, 4:]
    return X, y, list(y.columns)

def tokenize(text: str) -> list(str):
    """
    Function which cleans raw text and tokenizes it
    Args:
        text: raw text to clean
    Returns:
        text: cleaned text
    """
    # make everything lower case
    text = text.lower()
    # remove punctuation
    text = re.sub(r'[!"#$%&\'()\*\+,-./:;<=>?@\[\\\]^_`{|}~]', " ", text)
    # tokenize
    text = word_tokenize(text)
    # remove stop words
    stop_words = list(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = [word for word in text if word not in stop_words]
    for pos in ['v', 'a', 'r', 's']:
        text = [lemmatizer.lemmatize(word, pos) for word in text]
    # output result
    return text


def build_model():
    """
    Returns a pipeline which vectorizes text and contains a classifier
    Args:
        None
    Returns:
        pipeline: untrained sklearn pipeline object
    """
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(tokenizer=tokenize, min_df=5, ngram_range=(2,2))),
        ('classifier', MultiOutputClassifier(XGBClassifier(n_estimators=2000, eta=0.2, min_child_weight=2)))
    ])
    # Better performance model, but takes a long time to train
    # MultiOutputClassifier(XGBClassifier(n_estimators=2000, eta=0.2, min_child_weight=2))
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates model and prints out metrics based on each category
    Args:
        model: trained pipeline
        X_test: test data
        Y_test: test targets
        category_names: list of all categories
    Returns:
        None
    """
    preds = model.predict(X_test)
    for i, cat in enumerate(category_names):
        print("Metrics for ", cat)
        print(classification_report(Y_test.iloc[:, i], preds[:, i], zero_division=0))


def save_model(model, model_filepath):
    """
    Saves trained model to disk
    Args:
        model: trained sklearn pipeline
        model_filepath: path to save model to (should have .pkl extension)
    Returns:
        None
    """
    joblib.dump(model, model_filepath)


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