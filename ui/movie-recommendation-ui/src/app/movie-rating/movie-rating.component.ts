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

  userId: number = 1; 
  constructor(private movieService: MovieService) {
    this.loadMovies();  // Load the list of movies when component is initialized
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
}
