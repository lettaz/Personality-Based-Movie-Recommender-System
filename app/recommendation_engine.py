import logging
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from .data_loader import load_data, save_new_user_data
from .config import MIN_RATINGS_COUNT

# Setup logging
logging.basicConfig(level=logging.INFO)

# Preprocessing functions
def preprocess_personality_data(data):
    data.columns = [col.strip() for col in data.columns]
    relevant_columns = ['userid', 'openness', 'agreeableness', 'emotional_stability', 'conscientiousness', 'extraversion']
    data = data[relevant_columns]
    data = data.drop_duplicates(subset=['userid'])
    return data

def preprocess_ratings_data(data):
    data.columns = data.columns.str.strip()
    return data.groupby(['useri', 'movie_id']).agg({'rating': 'mean'}).reset_index()

# Load and preprocess initial data
personality_data, ratings_data, movies_data = load_data()
personality_data = preprocess_personality_data(personality_data)
ratings_data = preprocess_ratings_data(ratings_data)

def create_user_profiles():
    user_profiles = personality_data.set_index('userid')
    return user_profiles

def calculate_similarity(user_profiles, k):
    correlation_matrix = pd.DataFrame(squareform(pdist(user_profiles, metric='correlation')), columns=user_profiles.index, index=user_profiles.index)
    top_k_similarities = correlation_matrix.apply(lambda row: row.nlargest(k+1).iloc[1:], axis=1)
    return top_k_similarities

def find_unrated_movies(user_id, ratings_data, movies_data):
    rated_movie_ids = ratings_data.loc[ratings_data['useri'] == user_id, 'movie_id']
    unrated_movies = movies_data[~movies_data['movieId'].isin(rated_movie_ids)]
    return unrated_movies

def predict_movie_ratings(user_id, unrated_movies, top_k_similarities, ratings_data, k):
    top_k_users = top_k_similarities.loc[user_id].index
    filtered_ratings = ratings_data[ratings_data['useri'].isin(top_k_users)]
    predictions = []
    for movie_id in unrated_movies['movieId']:
        relevant_ratings = filtered_ratings[filtered_ratings['movie_id'] == movie_id]['rating']
        if not relevant_ratings.empty:
            predicted_rating = relevant_ratings.mean()
        else:
            all_users_relevant_ratings = ratings_data[ratings_data['movie_id'] == movie_id]['rating']
            predicted_rating = all_users_relevant_ratings.mean() if not all_users_relevant_ratings.empty else np.nan
        if not np.isnan(predicted_rating):
            predictions.append((movie_id, predicted_rating))
    return predictions

def filter_movies_by_rating_count(ratings_data, movies_data):
    ratings_count = ratings_data['movie_id'].value_counts()
    frequently_rated_movies = ratings_count[ratings_count >= MIN_RATINGS_COUNT].index
    return movies_data[movies_data['movieId'].isin(frequently_rated_movies)]

def recommend_movies_for_new_user(new_user_id, new_user_data, updated_personality_data, ratings_data, movies_data, k, top_n=10):
    # Use the updated personality data directly to create user profiles
    user_profiles = updated_personality_data.set_index('userid')

    # No need to add the new user data to the personality data again
    # as it is already included in updated_personality_data

    # Filter movies based on rating count
    filtered_movies_data = filter_movies_by_rating_count(ratings_data, movies_data)

    # Find unrated movies for the new user
    unrated_movies = find_unrated_movies(new_user_id, ratings_data, filtered_movies_data)
    top_k_similarities = calculate_similarity(user_profiles, k)

    # Predict ratings for unrated movies
    predicted_ratings = predict_movie_ratings(new_user_id, unrated_movies, top_k_similarities, ratings_data, k)

    # Sort the predicted ratings and select the top_n recommendations
    top_predictions = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:top_n]
    user_recommendations = [
        ( 
            int(movie_id),
            filtered_movies_data[filtered_movies_data['movieId'] == movie_id]['title'].iloc[0], 
            rating
        ) 
        for movie_id, rating in top_predictions
    ]
    
    return user_recommendations


def recommend_movies_for_old_user(user_id, personality_data, ratings_data, movies_data, k, top_n=10):
    if user_id in personality_data['userid'].values:
        existing_user_data = personality_data[personality_data['userid'] == user_id].iloc[0].to_dict()
        # Pass the original personality data as it is for existing users
        return recommend_movies_for_new_user(user_id, existing_user_data, personality_data, ratings_data, movies_data, k, top_n)
    else:
        logging.error(f"User ID {user_id} not found in personality data.")
        return None



def get_top_genres_from_movies(recommended_movies, movies_data, num_genres):
    # Adjust the unpacking to match the tuple structure
    recommended_movie_ids = [movie_id for movie_id, _, _ in recommended_movies]
    recommended_movie_genres = movies_data[movies_data['movieId'].isin(recommended_movie_ids)]['genres']
    genre_counts = recommended_movie_genres.str.split('|').explode().value_counts()
    top_genres = genre_counts.head(num_genres).index.tolist()
    return top_genres


def recommend_genres_for_old_user(user_id, personality_data, ratings_data, movies_data, k, num_genres):
    recommended_movies = recommend_movies_for_old_user(user_id, personality_data, ratings_data, movies_data, k, num_genres)
    if recommended_movies:
        return get_top_genres_from_movies(recommended_movies, movies_data, num_genres)
    else:
        return None
