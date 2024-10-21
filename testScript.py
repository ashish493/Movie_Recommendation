import requests
import random

# URL for API endpoints
LOGIN_URL = "http://localhost:5000/login"  # Change this to your login URL
RATE_MOVIE_URL = "http://localhost:5000/rate"  # Change this to your rating API URL

# Sample users (you may have more or use real users from your database)
users = [
    {"username": "user_1", "password": "default_password"},
    {"username": "user_2", "password": "default_password"},
    {"username": "user_3", "password": "default_password"},
    {"username": "user_4", "password": "default_password"},
    {"username": "user_5", "password": "default_password"},
    {"username": "user_6", "password": "default_password"},
    {"username": "user_7", "password": "default_password"},
    {"username": "user_8", "password": "default_password"},
    {"username": "user_9", "password": "default_password"},
    {"username": "user_10", "password": "default_password"},
    # Add more users as needed
]

# List of movie IDs (assuming you have a list of movie IDs)
movies = list(range(1, 1683))  # Assuming movie IDs are from 1 to 1682

def login_and_rate(user):
    # Log in to get user token (if your API uses tokens for authentication)
    login_payload = {
        "username": user["username"],
        "password": user["password"]
    }
    
    response = requests.post(LOGIN_URL, json=login_payload)
    if response.status_code == 200:
        print(f"User {user['username']} logged in successfully.")
        token = response.json().get("token")
        
        # Rate random movies
        for _ in range(10):  # Each user rates 5 movies randomly
            movie_id = random.choice(movies)
            rating = random.randint(1, 5)  # Random rating between 1 and 5
            
            rate_payload = {
                "movie_id": movie_id,
                "rating": rating
            }
            
            # Set headers if you are using a token for authentication
            headers = {
                "Authorization": f"Bearer {token}"
            }
            
            rate_response = requests.post(RATE_MOVIE_URL, json=rate_payload, headers=headers)
            if rate_response.status_code == 200:
                print(f"User {user['username']} rated movie {movie_id} with {rating} stars.")
            else:
                print(f"Failed to rate movie {movie_id}. Status code: {rate_response.status_code}")
    else:
        print(f"Login failed for {user['username']}. Status code: {response.status_code}")

def main():
    # Simulate random user logins and movie ratings
    for user in users:
        login_and_rate(user)

if __name__ == "__main__":
    main()
