import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads message and categories
    Input: 
        messages_filepath: path to messages.csv 
        categories_filepath: path to categories.csv
    Output:
        df: Joined dataset of messages and categories 
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='outer', on=['id'])
    return df

def clean_data(df):
    '''
        Input:
        df: Merged dataset in load_data
        Output:
        df: Dataset after cleaning process
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True) 
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # In first row [0], separate the string and the value by '-' using .split(), then using lambda x: x[0], take the first string, and replace "_" with " "
    category_colnames = row.str.split('-').apply(lambda x:x[0].replace("_", " "))
    # rename the columns of `categories` using category_colnames    
    categories.columns = category_colnames
    
    for column in categories:
        '''
        In: Unclean data
        Out: Clean data  
        '''
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
        # converting number greater than 1 to 1 to fit multioutputclassification
        categories[column] = categories[column].apply(lambda x: 1 if x > 1 else x)
    
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    
    # concatenate the original data frame with the new `categories` data frame
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates(subset=['message'])
    return df

def save_data(df, database_filename):
    '''
    df: Clean data
    database_filename: Path of SQL database stored
    '''
    engine = f'sqlite:///{database_filename}'
    df.to_sql('Disaster_Response_Database', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()