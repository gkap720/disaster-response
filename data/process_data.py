import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function takes in two csv file paths and returns a combined DataFrame
    Args:
        messages_filepath: path to a csv containing messages
        categories_filepath: path to a csv containing categories about messages
    Returns:
        df: new DataFrame with combined input csv's
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id", how="outer")
    return df


def clean_data(df):
    """
    This function takes in a dataframe and cleans it so that the categories are
    in the desired format.
    Args:
        df: the DataFrame to be cleaned
    Returns:
        df: the cleaned DataFrame
    """
    columns = [cat[:-2] for cat in df.loc[0, "categories"].split(";")]
    values = df["categories"].str.split(';').map(lambda x: [val[-1:] for val in x]).tolist()
    categories = pd.DataFrame(values, columns=columns).apply(pd.to_numeric, axis=0)
    # some values were greater than 1, clip these to 1
    categories = categories.applymap(lambda x: 1 if x > 1 else x)
    df.drop(columns="categories", inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Function saves dataframe into a database.
    Args:
        df: DataFrame to save
        database_filename: location of SQLite Database to save to
    Returns:
        None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('message', engine, index=False, if_exists="replace")


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