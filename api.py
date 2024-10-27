from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
import bcrypt
import pandas as pd
from prometheus_client import Counter, Gauge, generate_latest
from prometheus_client.core import CollectorRegistry
from prometheus_flask_exporter import PrometheusMetrics
from models import db, Rating, User
from sklearn.model_selection import train_test_split
import torch
from torch_movie import MatrixFactorization  # Import your PyTorch model class
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# DB Config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///moviedb.sqlite?check_same_thread=False'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'your_jwt_secret_key'  # Change this to a strong secret key
db.init_app(app)

# Enable CORS for Angular
CORS(app)
jwt = JWTManager(app)

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

# Split the dataset into training and test sets
train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)

# PyTorch model configuration
n_users, n_items = ratings_df['userId'].nunique(), ratings_df['movieId'].nunique()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and load saved state
def load_model(n_users, n_items):
    model = MatrixFactorization(n_users, n_items).to(device)
    model.load_state_dict(torch.load('matrix_factorization_model.pth', map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model(n_users=n_users, n_items=n_items) 

# Predict rating function
def predict_rating(model, user_id, movie_id):
    user = torch.tensor([user_id], dtype=torch.long, device=device)
    item = torch.tensor([movie_id], dtype=torch.long, device=device)
    with torch.no_grad():
        prediction = model(user, item).item()
    return prediction

# Training function
def train_model(model, train_data, epochs=1, batch_size=32, lr=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(train_data), batch_size):
            batch = train_data.iloc[i:i+batch_size]
            users = torch.tensor(batch['userId'].values, dtype=torch.long).to(device)
            movies = torch.tensor(batch['movieId'].values, dtype=torch.long).to(device)
            ratings = torch.tensor(batch['rating'].values, dtype=torch.float).to(device)
            optimizer.zero_grad()
            predictions = model(users, movies)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_data)}')
    # Save the trained model
    torch.save(model.state_dict(), 'matrix_factorization_model.pth')

# Prometheus metrics setup
registry = CollectorRegistry()
RMSE_GAUGE = Gauge('model_rmse', 'Root Mean Squared Error of the Model', registry=registry)
MAE_GAUGE = Gauge('model_mae', 'Mean Absolute Error of the Model', registry=registry)
PREDICTION_COUNTER = Counter('predictions_made', 'Total number of movie rating predictions', registry=registry)

# User Registration and Authentication

def populate_database():
    with app.app_context():
        if User.query.first() is None:  # Only populate if no users exist
            unique_users = ratings_df['userId'].unique()
            for user_id in unique_users:
                try:
                    user_id = int(user_id)
                    username = f"user_{user_id}"
                    password = bcrypt.hashpw('default_password'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

                    # Check if user ID already exists in the database before adding
                    if not User.query.filter_by(id=user_id).first():
                        new_user = User(id=user_id, username=username, password=password)
                        db.session.add(new_user)
                except Exception as e:
                    print(f"Error adding user {user_id}: {e}")
                    db.session.rollback()
                    continue

            db.session.commit()
            print(f"Populated user table with {len(unique_users)} users.")

        if Rating.query.first() is None:  # Only populate if no ratings exist
            for _, row in ratings_df.iterrows():
                try:
                    user_id = int(row['userId'])
                    movie_id = int(row['movieId'])
                    rating = float(row['rating'])

                    # Add each rating to the 'rating' table
                    new_rating = Rating(user_id=user_id, movie_id=movie_id, rating=rating)
                    db.session.add(new_rating)
                except Exception as e:
                    print(f"Error adding rating for user {user_id} and movie {movie_id}: {e}")
                    db.session.rollback()
                    continue

            db.session.commit()
            print(f"Populated rating table with {len(ratings_df)} ratings.")

# Call the populate function if needed
populate_database()

@app.route('/register', methods=['POST'])
def register():
    username = request.json.get('username')
    password = request.json.get('password')
    if User.query.filter_by(username=username).first():
        return jsonify({"msg": "Username already exists"}), 400
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    new_user = User(username=username, password=hashed_password.decode('utf-8'))
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({"msg": "User registered successfully"}), 201

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')
    user = User.query.filter_by(username=username).first()
    if not user or not bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
        return jsonify({"msg": "Invalid username or password"}), 401
    access_token = create_access_token(identity=user.id)
    return jsonify({'access_token': access_token, 'user_id': user.id}), 200

# Rate a movie
@app.route('/rate', methods=['POST'])
@jwt_required()
def rate_movie():
    # with app.app_context():  # Ensure we have an application context
    global ratings_df
    current_user = get_jwt_identity()
    user_id = current_user
    movie_id = int(request.json.get('movie_id'))
    rating = float(request.json.get('rating'))

    new_rating = Rating(user_id=user_id, movie_id=movie_id, rating=rating)
    db.session.add(new_rating)
    db.session.commit()

    # # Check if db.session.bind is available
    # if db.session.bind is None:
    #     return jsonify({"error": "Database connection is not available"}), 500

    # Retrain the model with the updated ratings
    # Update the rating in the DataFrame
    ratings_df.loc[(ratings_df['userId'] == user_id) & (ratings_df['movieId'] == movie_id), 'rating'] = rating

    # If the rating does not exist in the DataFrame, add it
    if not ((ratings_df['userId'] == user_id) & (ratings_df['movieId'] == movie_id)).any():
        new_row = pd.DataFrame({'userId': [user_id], 'movieId': [movie_id], 'rating': [rating], 'timestamp': [pd.Timestamp.now().timestamp()]})
        ratings_df = pd.concat([ratings_df, new_row], ignore_index=True)

    # Use the updated DataFrame for training
    updated_ratings_df = ratings_df
    train_data, _ = train_test_split(updated_ratings_df, test_size=0.2, random_state=42)
    model = load_model(n_users, n_items)
    train_model(model, train_data)
    movie_name = movie_names.get(movie_id, "Unknown Movie")
    return jsonify({"msg": f"User {current_user} rated movie '{movie_name}' with {rating}"}), 200

# Get movie recommendations using PyTorch model
@app.route('/recommendations/<int:user_id>', methods=['GET'])
@jwt_required()
def get_recommendations(user_id):
    movie_scores = {}
    # Load pre-trained model
    model = load_model(n_users=n_users, n_items=n_items)
    # Predict ratings for all movies in the test set
    for _, row in test_data.iterrows():
        predicted_rating = predict_rating(model, row['userId'], row['movieId'])
        movie_scores[row['movieId']] = predicted_rating
    # Get top 10 recommended movies
    recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    recommendations_list = [{'movie_name': movie_names.get(movie[0], "Unknown Movie"), 'predicted_rating': movie[1]} for movie in recommendations]
    PREDICTION_COUNTER.inc()
    return jsonify({'recommendations': recommendations_list}), 200

@app.route('/movies', methods=['GET'])
def get_movie_list():
    movie_list = [{'movie_id': mid, 'movie_name': name} for mid, name in movie_names.items()]
    return jsonify(movie_list), 200

def add_predictions_to_test_data(test_data, model):
    predictions = []
    for _, row in test_data.iterrows():
        predicted_rating = predict_rating(model, row['userId'], row['movieId'])
        predictions.append(predicted_rating)
    test_data['predicted_rating'] = predictions

# Add predictions to test_data
add_predictions_to_test_data(test_data, model)

@app.route('/evaluate_model', methods=['GET'])
def evaluate_model():
    try:
        # Check if 'predicted_rating' column exists
        if 'predicted_rating' not in test_data.columns:
            raise ValueError("The 'predicted_rating' column is missing from test_data")

        # Ensure 'rating' and 'predicted_rating' columns are numeric
        if not pd.api.types.is_numeric_dtype(test_data['rating']):
            raise ValueError("The 'rating' column must be numeric")
        if not pd.api.types.is_numeric_dtype(test_data['predicted_rating']):
            raise ValueError("The 'predicted_rating' column must be numeric")

        # Calculate RMSE and MAE
        rmse = np.sqrt(mean_squared_error(test_data['rating'], test_data['predicted_rating']))
        mae = mean_absolute_error(test_data['rating'], test_data['predicted_rating'])

        # Update Prometheus metrics
        RMSE_GAUGE.set(rmse)
        MAE_GAUGE.set(mae)

        return jsonify({'RMSE': rmse, 'MAE': mae}), 200
    except Exception as e:
        # Log the exception
        logging.error("Error in evaluate_model: %s", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(registry), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)