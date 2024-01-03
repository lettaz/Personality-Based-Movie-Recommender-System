from flask import jsonify, request
from .data_loader import load_data
from .recommendation_engine import recommend_movies_for_old_user, recommend_genres_for_old_user, preprocess_personality_data, preprocess_ratings_data
from .data_loader import get_user_personality_data, save_new_user_data, update_user_data
import uuid



def init_routes(app):
    @app.route('/api/users/new', methods=['POST'])
    def create_new_user():
        new_user_data = request.json
        # Generate a new user_id and remove dashes
        user_id = str(uuid.uuid4()).replace('-', '')
        new_user_data['userid'] = user_id
        save_new_user_data(new_user_data)
        return jsonify({"user_id": user_id}), 201

    @app.route('/api/personality/<user_id>', methods=['GET'])
    def get_personality(user_id):
        user_data = get_user_personality_data(user_id)
        if user_data:
            return jsonify(user_data)
        else:
            return jsonify({"error": "User not found"}), 404

    @app.route('/api/personality', methods=['POST'])
    def update_personality():
        updated_user_data = request.json
        update_user_data(updated_user_data)
        return jsonify({"message": "Personality updated"}), 200

    @app.route('/api/recommendations/movies', methods=['POST'])
    def get_movie_recommendations():
        req_data = request.json
        user_id = req_data.get("user_id")
        top_n = req_data.get("top_n", 10)
        k_value = req_data.get("k", 50)

        # Load and preprocess data
        personality_data, ratings_data, movies_data = load_data()
        personality_data = preprocess_personality_data(personality_data)
        ratings_data = preprocess_ratings_data(ratings_data)

        recommendations = recommend_movies_for_old_user(user_id, personality_data, ratings_data, movies_data, k_value, top_n)
        if recommendations is not None:
            return jsonify({"recommendations": recommendations})
        else:
            return jsonify({"error": "User not found or no recommendations available"}), 404

    @app.route('/api/recommendations/genre', methods=['POST'])
    def get_genre_recommendations():
        req_data = request.json
        user_id = req_data.get("user_id")
        num_genres = req_data.get("num_genres", 5)
        k_value = req_data.get("k", 50)

        # Load and preprocess data
        personality_data, ratings_data, movies_data = load_data()
        personality_data = preprocess_personality_data(personality_data)
        ratings_data = preprocess_ratings_data(ratings_data)

        recommendations = recommend_genres_for_old_user(user_id, personality_data, ratings_data, movies_data, k_value, num_genres)
        if recommendations is not None:
            return jsonify({"recommendations": recommendations})
        else:
            return jsonify({"error": "User not found or no recommendations available"}), 404