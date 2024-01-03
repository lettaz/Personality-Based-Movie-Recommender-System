from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
import uuid
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

# Initialize data

k = 5


# Read CSV files
personality_data = pd.read_csv('datasets/2018-personality-data.csv')
ratings_data = pd.read_csv('datasets/2018_ratings.csv')
movies_data = pd.read_csv('datasets/movies.csv')

'''
    This is for all the functions from the jupyter notebook
'''

# Dummy implementation of notebook functions
def recommend_movies(user_id, personality_data, ratings_data, movies_data, k, top_n):
    # Implement your notebook logic here
    return [("Movie1", 5), ("Movie2", 4.5)]

# Preprocess personality data from the CSV
def preprocess_personality_data(data):
    
    # Adjusting column names by removing leading spaces
    data.columns = [col.strip() for col in personality_data.columns]
    
    # Keeping only relevant columns
    relevant_columns = ['userid', 'openness', 'agreeableness', 'emotional_stability', 'conscientiousness', 'extraversion']
    data = data[relevant_columns]

    # Removing duplicates
    data = data.drop_duplicates(subset=['userid'])

    return data

# Preprocess rating data from the CSV
def preprocess_ratings_data(data):
    # Strip whitespace from column names
    data.columns = data.columns.str.strip()

    # Aggregate duplicate ratings
    return data.groupby(['useri', 'movie_id']).agg({'rating': 'mean'}).reset_index()

# Preprocess each dataset
personality_data = preprocess_personality_data(personality_data)
ratings_data = preprocess_ratings_data(ratings_data)

# Create user profiles
def create_user_profiles(personality_data):
    user_profiles = personality_data.set_index('userid')
    return user_profiles

user_profiles = create_user_profiles(personality_data)

# Calculate similarities for movie recommendation
def calculate_movie_similarity(user_profiles, k):
    # Compute the pairwise Pearson correlation coefficient between all user profiles
    correlation_matrix = pd.DataFrame(squareform(pdist(user_profiles, metric='correlation')), columns=user_profiles.index, index=user_profiles.index)

    # For each user, find the top k similar users (excluding self)
    top_k_similarities = correlation_matrix.apply(lambda row: row.nlargest(k+1).iloc[1:], axis=1)
    
    return top_k_similarities

top_k_similarities = calculate_movie_similarity(user_profiles, k)

# Get User Personality Data
def GetUserPersonalityData(user_id, personality_data):
    user_data = personality_data[personality_data['userid'] == user_id]
    if not user_data.empty:
        user_data_dict = user_data.iloc[0].to_dict()
        print("User Data from GetUserPersonalityData:\n", user_data_dict)  # Debug print
        return user_data_dict
    else:
        print("No user data found for ID:", user_id)  # Debug print
        return None

# Generate new movie dataset of only movies with 'n' number of rating count
def filter_movies_by_rating_count(ratings_data, movies_data, min_ratings_count):
    # Count the number of ratings each movie received
    ratings_count = ratings_data['movie_id'].value_counts()

    # Filter movies that have been rated at least 'min_ratings_count' times
    frequently_rated_movies = ratings_count[ratings_count >= min_ratings_count].index

    # Filter the movies_data DataFrame
    filtered_movies_data = movies_data[movies_data['movieId'].isin(frequently_rated_movies)]

    return filtered_movies_data

# Calling the function
min_ratings_count = 5  # Define the threshold
movies_data = filter_movies_by_rating_count(ratings_data, movies_data, min_ratings_count)

# Find Unrated Movies
def find_unrated_movies(user_id, ratings_data, movies_data):
    rated_movie_ids = ratings_data.loc[ratings_data['useri'] == user_id, 'movie_id']
    unrated_movies = movies_data[~movies_data['movieId'].isin(rated_movie_ids)]
    print("Unrated Movies from find_unrated_movies:\n", unrated_movies.head())  # Debug print
    return unrated_movies

# Predict Movie Ratings
def predict_movie_ratings(user_id, unrated_movies, top_k_similarities, ratings_data, k):
    top_k_users = top_k_similarities.loc[user_id].index
    filtered_ratings = ratings_data[ratings_data['useri'].isin(top_k_users)]

    predictions = []
    for movie_id in unrated_movies['movieId']:
        # First, try to predict based on similar users
        relevant_ratings = filtered_ratings[filtered_ratings['movie_id'] == movie_id]['rating']
        if not relevant_ratings.empty:
            predicted_rating = relevant_ratings.mean()
        else:
            # Fallback: Use average rating of the movie by all users
            all_users_relevant_ratings = ratings_data[ratings_data['movie_id'] == movie_id]['rating']
            predicted_rating = all_users_relevant_ratings.mean() if not all_users_relevant_ratings.empty else np.nan
        
        if not np.isnan(predicted_rating):
            predictions.append((movie_id, predicted_rating))
    
    
    return predictions


# Predict top n movies for a new user
# This will return a list of the top 10 recommended movies (with titles) and their predicted ratings for a user, will also output it to csv.
def recommend_movies_for_new_user(new_user_id, new_user_data, personality_data, ratings_data, movies_data, k, top_n=10):
    
    # Check if user is new
    if new_user_id not in personality_data['userid'].values:
        
        # Add new user to personality data
        new_user_df = pd.DataFrame([new_user_data], columns=personality_data.columns)
        personality_data = pd.concat([personality_data, new_user_df], ignore_index=True)
    
    # Preprocess data
    personality_data = preprocess_personality_data(personality_data)

    # Create user profiles
    user_profiles = create_user_profiles(personality_data)
    
    unrated_movies = find_unrated_movies(new_user_id, ratings_data, movies_data)
    predicted_ratings = predict_movie_ratings(new_user_id, unrated_movies, user_profiles, ratings_data, k)
    top_predictions = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:top_n]
    # Include movieId in the output
    user_predictions = [(movies_data[movies_data['movieId'] == movie_id].iloc[0]['movieId'], 
                         movies_data[movies_data['movieId'] == movie_id].iloc[0]['title'], 
                         rating) 
                        for movie_id, rating in top_predictions]

    return user_predictions  # Output format: [(movieId, title, rating), ...]

# Old User Integration and Recommendation    
def recommend_movies_for_old_user(existing_user_id, personality_data, ratings_data, movies_data, k, top_n = 10):
    # Get existing user data
    existing_user_data = GetUserPersonalityData(existing_user_id, personality_data)

    # Check if the user data exists
    if existing_user_data is not None:
        # Get recommended movies for the existing user
        recommended_movies = recommend_movies_for_new_user(
            existing_user_id, 
            existing_user_data, 
            personality_data, 
            ratings_data, 
            movies_data, 
            k,
            top_n
        )
        return recommended_movies
    else:
        return "User not found in the dataset."
    


    
'''
    This is the function for genre recommendation
'''

def get_top_genres_from_movies(recommended_movies, movies_data, num_genres):
    # Extract movie IDs from recommended_movies
    recommended_movie_ids = [movie_id for movie_id, _, _ in recommended_movies]
    # Filter movies_data to only include recommended movies
    recommended_movie_genres = movies_data[movies_data['movieId'].isin(recommended_movie_ids)]['genres']

    # Split genres into individual labels
    all_genres = []
    for genre_list in recommended_movie_genres:
        all_genres.extend(genre_list.split('|'))  # Split and add to the list

    # Count genre occurrences
    genre_counts = pd.Series(all_genres).value_counts()
    
    # Select top genres
    top_genres = genre_counts.head(num_genres).index.tolist()
    
    return top_genres




def recommend_genres_for_new_user(new_user_id, new_user_data, personality_data, ratings_data, movies_data, k, num_genres):
    print("New User ID:", new_user_id)  # Debug print
    top_movies = recommend_movies_for_new_user(new_user_id, new_user_data, personality_data, ratings_data, movies_data, k, num_genres)
    print("Top Movies:", top_movies)  # Debug print
    top_genres = get_top_genres_from_movies(top_movies, movies_data, num_genres)
    print("Top Genres:", top_genres)  # Debug print
    return top_genres


def recommend_genres_for_old_user(user_id, personality_data, ratings_data, movies_data, k=5, num_genres=5):
    # Get existing user data
    existing_user_data = GetUserPersonalityData(user_id, personality_data)

    # Check if the user data exists
    if existing_user_data is not None:
        # Get recommended genres for the existing user
        recommended_genres = recommend_genres_for_new_user(
            user_id, 
            existing_user_data, 
            personality_data, 
            ratings_data, 
            movies_data, 
            k,
            num_genres
        )
        return recommended_genres
    else:
        return "User not found in the dataset."


'''
    This is for all the API endpoints
'''

# Endpoint to create a new user and provide movie recommendations
@app.route('/api/users/new', methods=['POST'])
def create_new_user():
    new_user_data = request.json
    global personality_data

    # Generate a new user_id and remove dashes
    user_id = str(uuid.uuid4()).replace('-', '')

    print("Generated new user_id:", user_id)
    
    # Add the generated user_id to the new user data
    new_user_data['userid'] = user_id

    # Ensure the new user data is formatted correctly as a DataFrame
    new_user_df = pd.DataFrame([new_user_data], columns=personality_data.columns)

    # Concatenate the new user data with the existing personality data
    personality_data = pd.concat([personality_data, new_user_df], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    personality_data.to_csv('datasets/2018-personality-data.csv', index=False)

    return jsonify({"user_id": user_id}), 201


# Endpoint to get user personality data
@app.route('/api/personality/<user_id>', methods=['GET'])
def get_personality(user_id):
    user_data = GetUserPersonalityData(user_id, personality_data)
    print("user_id:", user_id)
    print("personality_data:", personality_data.tail())
    if user_data:
        return jsonify(user_data)
    else:
        return jsonify({"error": "User not found"}), 404



# Endpoint to update user personality data
@app.route('/api/personality', methods=['POST'])
def update_personality():
    global personality_data  # Declare global variable

    user_data = request.json
    user_id = user_data.get("userid")

    if user_id in personality_data['userid'].values:
        for key, value in user_data.items():
            if key in personality_data.columns:
                personality_data.loc[personality_data['userid'] == user_id, key] = value

        personality_data.to_csv('2018-personality-data.csv', index=False)
        return jsonify({"message": "Personality updated"}), 200
    else:
        return jsonify({"error": "User not found"}), 404



# Endpoint to recommend movies to an existing user
@app.route('/api/recommendations/movies', methods=['POST'])
def get_movie_recommendations():
    req_data = request.json
    user_id = req_data.get("user_id")
    top_n = req_data.get("top_n", 10)
    k_value = req_data.get("k", 50)
    
    existing_user_data = GetUserPersonalityData(user_id, personality_data)
    if existing_user_data:
        recommendations = recommend_movies_for_old_user(user_id, personality_data, ratings_data, movies_data, k_value, top_n)
        return jsonify({"recommendations": recommendations})
    else:
        return jsonify({"error": "User not found"}), 404


# Endpoint to recommend genres to an existing user
@app.route('/api/recommendations/genre', methods=['POST'])
def get_genre_recommendations():
    req_data = request.json
    user_id = req_data.get("user_id")
    num_genres = req_data.get("num_genres", 5)
    k_value = req_data.get("k", 50)
    
    existing_user_data = GetUserPersonalityData(user_id, personality_data)
    if existing_user_data:
        recommendations = recommend_genres_for_old_user(user_id, personality_data, ratings_data, movies_data, k_value, num_genres)
        return jsonify({"recommendations": recommendations})
    else:
        return jsonify({"error": "User not found"}), 404



if __name__ == "__main__":
    app.run(debug=True)
