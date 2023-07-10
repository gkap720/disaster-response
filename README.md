# Disaster Response Pipeline Project
### Overview:
In this project, I first cleaned a database of text messages to prepare them for a classification model (lower case, remove stop words, lemmatization). I then created a pipeline for vectorizing the text before passing it to a model for training. After performing a GridSearch on a few different model architectures, I settled on XGBoost as tree-based (and boosted) models generally perform better on a sparse dataset (which was the case for my text-based dataset here). This repo contains the final weights of the trained pipeline in `models/classifier.pkl`. 

You can see this model in action by following the usage instructions (skip step 1 if you don't want to re-train the model). This will open up a local web server where you can view some visualizations related to the app as well as make some predictions based on your own text.

### Challenges

### Usage Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
