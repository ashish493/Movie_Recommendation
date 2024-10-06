from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
import bcrypt
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Enable CORS for angular 
CORS(app)
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})
# JWT Secret Key
app.config['JWT_SECRET_KEY'] = 'super-secret-key'
jwt = JWTManager(app)

# In-memory users database
users_db = {}

# Load MovieLens 100K Dataset
ratings_file = 'dataset/ml-100k/u.data'
ratings_df = pd.read_csv(ratings_file, sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])

# Create a user-item matrix
user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Perform matrix factorization using Truncated SVD
svd = TruncatedSVD(n_components=50)
latent_matrix = svd.fit_transform(user_item_matrix)

# Compute cosine similarity between users
user_similarity = cosine_similarity(latent_matrix)

# User Registration
@app.route('/register', methods=['POST'])
def register():
    username = request.json.get('username')
    password = request.json.get('password')

    if username in users_db:
        return jsonify({"msg": "Username already exists"}), 400

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users_db[username] = hashed_password
    print(users_db,"users db ")
    return jsonify({"msg": "User registered successfully"}), 200


# User Login
@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')

    if username not in users_db:
        return jsonify({"msg": "Invalid username"}), 401

    if not bcrypt.checkpw(password.encode('utf-8'), users_db[username]):
        return jsonify({"msg": "Invalid password"}), 401

    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token), 200


# Rate a movie (JWT required)
@app.route('/rate', methods=['POST'])
@jwt_required()
def rate_movie():
    current_user = get_jwt_identity()
    user_id = int(request.json.get('user_id'))
    movie_id = int(request.json.get('movie_id'))
    rating = float(request.json.get('rating'))

    # Optionally update ratings_df and user_item_matrix with this new rating
    # ratings_df.append({'userId': user_id, 'movieId': movie_id, 'rating': rating}, ignore_index=True)
    new_row = pd.DataFrame([{'userId': user_id, 'movieId': movie_id, 'rating': rating}])
    ratings_df = pd.concat([ratings_df, new_row], ignore_index=True)
    return jsonify({"msg": f"User {current_user} rated movie {movie_id} with {rating}"}), 200


# Get movie recommendations (JWT required)
@app.route('/recommendations/<int:user_id>', methods=['GET'])
@jwt_required()
def get_recommendations(user_id):
    print(user_id, "userid received ")
    
    # Get similarity scores for the current user
    similar_users = list(enumerate(user_similarity[user_id - 1]))  # Adjust user_id index
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)  # Sort by similarity score

    # Get top similar users
    top_similar_users = similar_users[1:11]  # Exclude the user themselves

    # Gather recommendations from top similar users
    movie_scores = {}
    
    for similar_user_id, similarity_score in top_similar_users:
        similar_user_ratings = ratings_df[ratings_df['userId'] == (similar_user_id + 1)]
        
        for _, row in similar_user_ratings.iterrows():
            movie_id = int(row['movieId'])  # Convert to int for JSON compatibility
            rating = float(row['rating'])  # Convert to float for JSON compatibility
            
            if movie_id not in movie_scores:
                movie_scores[movie_id] = rating * similarity_score
            else:
                movie_scores[movie_id] += rating * similarity_score

    # Sort movies by predicted score
    recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print(recommendations, "recommendations received ")

    # Prepare the recommendations for output
    recommendations_list = [{'movie_id': movie[0], 'predicted_rating': movie[1]} for movie in recommendations]
    
    # No need for .to_dict() here, as we're already creating a list of dictionaries
    return jsonify({'recommendations': recommendations_list}), 200

if __name__ == '__main__':
    app.run(debug=True)
