import { Component } from '@angular/core';
import { MovieService } from '../movie.service';

@Component({
  selector: 'app-movie-rating',
  templateUrl: './movie-rating.component.html',
  styleUrls: ['./movie-rating.component.css']
})
export class MovieRatingComponent {
  username: string = '';
  password: string = '';
  movieId: number = 0;
  rating: number = 0;
  recommendations: any[] = [];
  loggedIn: boolean = false;
  errorMessage: string = "";
  movieList: any[] = [];  // Store the list of movies

  randomUsers = [
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
{"username": "user_11", "password": "default_password"},
{"username": "user_12", "password": "default_password"},
{"username": "user_13", "password": "default_password"},
{"username": "user_14", "password": "default_password"},
{"username": "user_15", "password": "default_password"},
{"username": "user_16", "password": "default_password"},
{"username": "user_17", "password": "default_password"},
{"username": "user_18", "password": "default_password"},
{"username": "user_19", "password": "default_password"},
{"username": "user_20", "password": "default_password"},
{"username": "user_21", "password": "default_password"},
{"username": "user_22", "password": "default_password"},
{"username": "user_23", "password": "default_password"},
{"username": "user_24", "password": "default_password"},
{"username": "user_25", "password": "default_password"},
{"username": "user_26", "password": "default_password"},
{"username": "user_27", "password": "default_password"},
{"username": "user_28", "password": "default_password"},
{"username": "user_29", "password": "default_password"},
{"username": "user_30", "password": "default_password"}
]
  userId: number = 1; 
  constructor(private movieService: MovieService) {
    this.loadMovies();  // Load the list of movies when component is initialized
    this.startAutomation();
  }

  // Login and get JWT token
  login() {
    this.movieService.login(this.username, this.password).subscribe(
      response => {
        this.movieService.setToken(response.access_token);
        this.userId = response.user_id;
        this.loggedIn = true;
      },
      error => {
        console.error('Login failed:', error);
        this.errorMessage = 'Login failed. Please check your credentials.';
      }
    );
  }

  // Load the list of movies from the API
  loadMovies() {
    this.movieService.getMovieList().subscribe(
      (data: any[]) => {
        this.movieList = data.map(movie => ({
          id: movie.movie_id,
          name: movie.movie_name
        }));
      },
      error => {
        console.error('Error loading movies:', error);
      }
    );
  }

  // Submit a movie rating
  rateMovie() {
    if (this.userId && this.movieId && this.rating) {
      this.movieService.rateMovie(this.userId, this.movieId, this.rating).subscribe(
        response => {
          console.log('Movie rated successfully:', response);
          this.getRecommendations(); // Fetch recommendations after rating
        },
        error => {
          console.error('Error rating the movie:', error);
          this.errorMessage = 'Could not rate the movie. Please try again.';
        }
      );
    } else {
      this.errorMessage = 'Please provide valid movie details.';
    }
  }

  // Get movie recommendations based on rating
//   getRecommendations() {
//     this.movieService.getRecommendations(this.userId).subscribe(
//       (data) => {
//         this.recommendations = data.recommendations;  
//         this.movieService.evaluateModel().subscribe(response => {
//           console.log('Model evaluation:', response);
//         });// Update the recommendation list
//       },
//       (error) => {
//         console.error('Error fetching recommendations:', error);
//         this.errorMessage = 'Could not fetch recommendations. Please try again later.';
//       }
//     );
//   }
// }

getRecommendations() {
  if (this.userId) {
    this.movieService.getRecommendations(this.userId).subscribe(
      (data) => {
        this.recommendations = data.recommendations; // Update the recommendation list
        this.movieService.evaluateModel().subscribe(response => {
                    console.log('Model evaluation:', response);
                  });// Update the recommendation list
      },
      (error) => {
        console.error('Error fetching recommendations:', error);
        this.errorMessage = 'Could not fetch recommendations. Please try again later.';
      }
    );
  }
}
randomLogin() {
  const randomUser = this.randomUsers[Math.floor(Math.random() * this.randomUsers.length)];
  this.username = randomUser.username;
  this.password = randomUser.password;

  this.movieService.login(this.username, this.password).subscribe(
    response => {
      this.movieService.setToken(response.access_token);
      this.userId = response.user_id;
      this.loggedIn = true;
      console.log('Logged in successfully as:', this.username);
      this.rateRandomMovie();  // Rate a random movie after login
    },
    error => {
      console.error('Login failed:', error);
      this.errorMessage = 'Login failed for ' + this.username;
    }
  );
}

// Rate a random movie with a random rating
rateRandomMovie() {
  const randomMovie = this.movieList[Math.floor(Math.random() * this.movieList.length)];
  const randomRating = Math.floor(Math.random() * 5) + 1; // Random rating between 1 and 5

  if (this.userId && randomMovie && randomRating) {
    this.movieService.rateMovie(this.userId, randomMovie.id, randomRating).subscribe(
      response => {
        console.log(`Rated movie "${randomMovie.name}" with ${randomRating} stars.`);
        this.getRecommendations();  // Fetch recommendations after rating
      },
      error => {
        console.error('Error rating the movie:', error);
        this.errorMessage = 'Could not rate the movie.';
      }
    );
  }
}
startAutomation() {
  const automationInterval = 1000;  // 5 seconds interval between actions (adjust as needed)
  
  setInterval(() => {
    this.randomLogin();  // Perform a random login and rating every interval
  }, automationInterval);
}

}
