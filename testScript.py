import requests
import random
import time

# API URL (change to your Flask API's base URL)
API_URL = 'http://localhost:5000'

# User credentials for login
USERNAME = 'user_1'  # Example user
PASSWORD = 'default_password'

# Total number of movies and the range for random ratings
MOVIE_ID_RANGE = (1, 100)  # Adjust based on your actual movie IDs
RATING_RANGE = (1, 5)  # Ratings between 1 and 5

# Get JWT token by logging in
def login(username, password):
    url = f'{API_URL}/login'
    payload = {'username': username, 'password': password}
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json()['access_token']
    else:
        print(f"Login failed: {response.text}")
        return None

# Rate a random movie
def rate_random_movie(token):
    headers = {'Authorization': f'Bearer {token}'}
    movie_id = random.randint(*MOVIE_ID_RANGE)
    rating = random.randint(*RATING_RANGE)

    url = f'{API_URL}/rate'
    payload = {'movie_id': movie_id, 'rating': rating}

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        print(f"Successfully rated movie {movie_id} with rating {rating}")
    else:
        print(f"Failed to rate movie: {response.text}")

# Function to repeatedly rate random movies
def repeatedly_rate_movies(username, password, num_ratings=100, delay=1):
    token = login(username, password)

    if token:
        for i in range(num_ratings):
            rate_random_movie(token)
            time.sleep(delay)  # Add delay between requests to avoid overloading the server
    else:
        print("Unable to obtain JWT token, aborting.")

if __name__ == '__main__':
    # Number of ratings to submit and delay between each request
    NUMBER_OF_RATINGS = 50  # Adjust as needed
    DELAY_BETWEEN_RATINGS = 2  # 2 seconds delay between requests

    repeatedly_rate_movies(USERNAME, PASSWORD, num_ratings=NUMBER_OF_RATINGS, delay=DELAY_BETWEEN_RATINGS)
