from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
import bcrypt
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from models import db, Rating, User

app = Flask(__name__)

# DB Config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///moviedb.sqlite'  # Path to your SQLite database file
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'your_jwt_secret_key'  # Change this to a strong secret key
db.init_app(app)

# with app.app_context():
#     db.create_all() 

# Enable CORS for angular 
CORS(app)
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})
# JWT Secret Key
app.config['JWT_SECRET_KEY'] = 'super-secret-key'
jwt = JWTManager(app)

# In-memory users database
users_db = {}

# Load MovieLens 100K Dataset
def load_movie_names(filepath):
    movie_names = {}
    with open(filepath, encoding='latin-1') as f:
        for line in f:
            fields = line.split('|')
            movie_id = int(fields[0])
            movie_name = fields[1]
            movie_names[movie_id] = movie_name
    return movie_names

movie_names = load_movie_names('dataset/ml-100k/u.item')

ratings_file = 'dataset/ml-100k/u.data'
ratings_df = pd.read_csv(ratings_file, sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])

# Populate data from existing dataset to our DB
with app.app_context():
    # Step 1: Populate the 'user' table with unique users from the dataset
    unique_users = ratings_df['userId'].unique()

    for user_id in unique_users:
        try:
            # Cast the user_id to a Python int to avoid datatype mismatch
            user_id = int(user_id)

            # Create a simple username and a hashed password for each user
            username = f"user_{user_id}"
            password = bcrypt.hashpw('default_password'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            # Check if user ID already exists in the database before adding
            if not User.query.filter_by(id=user_id).first():
                new_user = User(id=user_id, username=username, password=password)
                db.session.add(new_user)
        except Exception as e:
            print(f"Error adding user {user_id}: {e}")
            db.session.rollback()  # Rollback the session if an error occurs
            continue  # Skip to the next user

    db.session.commit()
    print(f"Populated user table with {len(unique_users)} users.")

    # Step 2: Populate the 'rating' table with data from the dataset
    for _, row in ratings_df.iterrows():
        try:
            user_id = int(row['userId'])  # Ensure user_id is cast to Python int
            movie_id = int(row['movieId'])
            rating = float(row['rating'])

            # Add each rating to the 'rating' table
            new_rating = Rating(user_id=user_id, movie_id=movie_id, rating=rating)
            db.session.add(new_rating)
        except Exception as e:
            print(f"Error adding rating for user {user_id} and movie {movie_id}: {e}")
            db.session.rollback()  # Rollback the session if an error occurs
            continue  # Skip to the next rating

    db.session.commit()
    print(f"Populated rating table with {len(ratings_df)} ratings.")

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

    if User.query.filter_by(username=username).first():
        return jsonify({"msg": "Username already exists"}), 400

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    new_user = User(username=username, password=hashed_password.decode('utf-8'))  # Decode to string
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({"msg": "User registered successfully"}), 201


# User Login
@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')

    user = User.query.filter_by(username=username).first()
    if not user or not bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
        return jsonify({"msg": "Invalid username or password"}), 401

    access_token = create_access_token(identity=user.id)
    return jsonify(access_token=access_token), 200


# Rate a movie (JWT required)
@app.route('/rate', methods=['POST'])
@jwt_required()
def rate_movie():
    current_user = get_jwt_identity()
    user_id = current_user  # Get the current user's ID from JWT
    movie_id = int(request.json.get('movie_id'))
    rating = float(request.json.get('rating'))

    # Add rating to the database
    new_rating = Rating(user_id=user_id, movie_id=movie_id, rating=rating)
    db.session.add(new_rating)
    db.session.commit()

    movie_name = movie_names.get(movie_id, "Unknown Movie")  # Fetch movie name

    return jsonify({"msg": f"User {user_id} rated movie '{movie_name}' with {rating}"}), 200

# Get movie recommendations (JWT required)
@app.route('/recommendations/<int:user_id>', methods=['GET'])
@jwt_required()
def get_recommendations_SVD_Factorization(user_id):
    # Get similarity scores for the current user
    similar_users = list(enumerate(user_similarity[user_id - 1]))  # Adjust user_id index
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)

    # Get top similar users
    top_similar_users = similar_users[1:11]  # Exclude the user themselves

    # Gather recommendations from top similar users
    movie_scores = {}
    
    for similar_user_id, similarity_score in top_similar_users:
        similar_user_ratings = Rating.query.filter_by(user_id=(similar_user_id + 1)).all()
        
        for rating in similar_user_ratings:
            movie_id = rating.movie_id
            rating_value = rating.rating
            
            if movie_id not in movie_scores:
                movie_scores[movie_id] = rating_value * similarity_score
            else:
                movie_scores[movie_id] += rating_value * similarity_score

    # Sort movies by predicted score
    recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    # Prepare the recommendations for output with movie names
    recommendations_list = [{'movie_name': movie_names.get(movie[0], "Unknown Movie"), 'predicted_rating': movie[1]} for movie in recommendations]

    return jsonify({'recommendations': recommendations_list}), 200

@app.route('/movies', methods=['GET'])
def get_movie_list():
    movie_list = [{'movie_id': mid, 'movie_name': name} for mid, name in movie_names.items()]
    return jsonify(movie_list), 200

if __name__ == '__main__':
    app.run(debug=True)
