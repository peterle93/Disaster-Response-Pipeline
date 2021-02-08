## Table of Contents
1. [Project Motivation](#motivation)
2. [Summary of Results](#results)
3. [Instructions](#instructions)
4. [Libraries Used](#libraries)
5. [File Descriptions](#descriptions)
6. [Acknowledgements](#acknowledgements)

## Project Motivation <a name="motivation"></a>
### Disaster Response Pipeline Project
The goal of the project is to classify legitimate messages sent during disaster events from a dataset provided by Figure Eight. 
It requires us to build machine learning pipeline to categorize emergency messages according to the needs communicated by the sender on a real time basis.
The specific machine model is a Natural Language Processing (NLP) model.

The project is divided into three main sections:

1. Building an ETL pipeline to extract data, cleaning the data and storing it into a SQlite Database.
2. Building a ML model to train the classifier to place the messages in the most accurate categories.
3. Running the app to display the models accuracy and results on real time basis.

## Summary of Results <a name="results"></a>
The web app is able to classify the messages sent and to place them into the most appropriate category.

![Disaster Response Pipeline](https://raw.githubusercontent.com/peterle93/Disaster-Response-Pipeline/master/images/Disaster%20Gif.gif)

By entering a message and pressing 'Classify message', the message will be classified into the all the categories listed that are suitable for its description.

## Instructions: <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Libraries: <a name="libraries"></a>
Requires Python 3.5+

Machine Learning Libraries
- Numpy
- Pandas
- Scikit Learn
- Matplotlib
- NLTK (Natural Language Processing)
- pickle (Model Loading and Saving Library)
- sqlalchemy (SQLlite Database)

Web App and Data Visualization
- Flask 
- plotly
- json

## File Descriptions <a name="descriptions"></a>
1. app
- run.py: Launches the web app 
2. data 
- DisasterResponse.db: Contains the SQL database 
- ETL Pipeline Preparation.ipynb: Extract, Transfer, Load Pipeline preparation code
- disaster_categories.csv: categories dataset
- disaster_messages.csv: messages dataset
- process_data.py: Processes the categories and messages dataset
3. models 
- classifier.pkl - classifier using pickle
- ML Pipeline Preparation.ipynb: Machine learning preparation code
- train_classifier.py: classification code 
4. images - contains the images and gifs for presentation
    
## Acknowledgements <a name="acknowledgements"></a>

1. https://www.figure-eight.com/
2. https://www.udacity.com/
