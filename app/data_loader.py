import pandas as pd

# Assuming you have global variables for file paths
PERSONALITY_DATA_FILE = 'datasets/2018-personality-data.csv'
RATINGS_DATA_FILE = 'datasets/2018_ratings.csv'
MOVIES_DATA_FILE = 'datasets/movies.csv'

def load_data():
    personality_data = pd.read_csv(PERSONALITY_DATA_FILE)
    ratings_data = pd.read_csv(RATINGS_DATA_FILE)
    movies_data = pd.read_csv(MOVIES_DATA_FILE)
    return personality_data, ratings_data, movies_data

def save_new_user_data(new_user_data):
    personality_data = pd.read_csv(PERSONALITY_DATA_FILE)
    # Convert new_user_data to DataFrame
    new_user_df = pd.DataFrame([new_user_data])
    # Append new_user_df to personality_data
    updated_personality_data = pd.concat([personality_data, new_user_df], ignore_index=True)
    updated_personality_data.to_csv(PERSONALITY_DATA_FILE, index=False)
    return updated_personality_data



def update_user_data(updated_user_data):
    personality_data = pd.read_csv(PERSONALITY_DATA_FILE)
    user_id = updated_user_data['userid']
    personality_data.loc[personality_data['userid'] == user_id, updated_user_data.keys()] = updated_user_data.values()
    personality_data.to_csv(PERSONALITY_DATA_FILE, index=False)


def get_user_personality_data(user_id):
    personality_data = pd.read_csv(PERSONALITY_DATA_FILE)
    user_data = personality_data[personality_data['userid'] == user_id]
    if not user_data.empty:
        return user_data.iloc[0].to_dict()
    return None

