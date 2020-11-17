## The Disaster Response Pipeline Project
### Overview
This project was done as a part of the Udacity's Data Scientist Nanodegree program in collaboartion with Figure Eight. The aim of the project is to build a model for an API that would classify the disaster resposne messages into categories using Natural Language Processing and Machine Learning and return the results in real time. The model is built using the data engineering techniques and the data has been taken from Figure Eight.

The project is divided into three main components, the ETL Pipeline, ML Pipeline and the Web app:

### 1. ETL Pipeline: file process_data.py contains the ETL pipeline
- Load the datasets 'messages' and 'categories'
- Merge the datasets
- Clean the dataset
- Save the cleaned data to sqlite database

### 2. ML Pipeline: file train_classifier.py contains the ML pipeline
- Load the data from sqlite database
- Preprocess the data using Natural Language processing
- Build a machine learning pipeline
- Split the data into train and test sets
- Train the model and evaluate the predictions
- Improve the model using Grid Search CV
- Export the model as a pickle file

### Flask Web App:
The user can enter the messages in the app and the app returns the classified categories in real time.

### Instructions to run the project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
