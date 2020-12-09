# Disaster Response Pipeline Project
A machine learning pipeline to categorize emergency messages according to the needs communicated by the sender

This is a project done for Udacity Data Scientist Nano degree program. This projects requires to create a machine learning model to predict classification for messages sent during disaster in right category. 
The data set is Provided by Figure Eight team and contains real messages set during disaster events.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
