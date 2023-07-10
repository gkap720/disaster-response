import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import plotly.express as px
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    cats = df.iloc[:, 4:]
    category_counts = cats.sum(axis=0)
    category_names = list(cats.columns)

    unlabeled_count = (cats.sum(axis=1) == 0).sum()
    underrepresented_cats = category_counts[category_counts < 500]
    underrepresented_cat_names = list(underrepresented_cats.index)
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    fig1 = px.bar(category_counts, color=range(0, len(category_counts)), 
                color_continuous_scale='Viridis',
                title="Distribution of Message Categories", 
                labels={'index': 'Category', 'value': 'Count'}
    )
    fig1 = fig1.update_layout(showlegend=False)
    fig1.update(layout_coloraxis_showscale=False)
    fig2 = px.pie([unlabeled_count, len(df) - unlabeled_count], values=0, 
       names=["Unlabeled", "Labeled"], color=["Unlabeled", "Labeled"], 
       color_discrete_map={'Unlabeled':'#440154', 'Labeled':'#22948B'},
       title="Share of Unlabeled Messages"
    )
    fig3 = px.bar(underrepresented_cats, color=range(0, len(underrepresented_cats)), 
                color_continuous_scale='Viridis',
                title='Underrepresented Categories', 
                labels={'index': 'Category', 'value': 'Count'}
    )
    fig3 = fig3.update_layout(showlegend=False)
    fig3.update(layout_coloraxis_showscale=False)
    graphs = [
        fig1,
        fig2,
        fig3
    ]
    classes = ['col-12', 'col-6', 'col-6']
    # encode plotly graphs in JSON
    ids = [["graph-{}".format(i), el[1]] for i, el in enumerate(zip(graphs, classes))]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()