import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class MovieService {
  private apiUrl = 'http://localhost:5000';  // Flask API URL
  private token: string = '';  // JWT Token for authentication

  constructor(private http: HttpClient) { }

  // Set token for authenticated requests
  setToken(token: string) {
    this.token = token;
  }

  // Login to get JWT token
  login(username: string, password: string): Observable<any> {
    const url = `${this.apiUrl}/login`;
    return this.http.post(url, { username, password });
  }

  // Register a new user
  register(username: string, password: string): Observable<any> {
    const url = `${this.apiUrl}/register`;
    return this.http.post(url, { username, password });
  }
  
  // Submit rating for a movie
  rateMovie(userId: number, movieId: number, rating: number): Observable<any> {
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`,
      'Content-Type': 'application/json'
    });
    const url = `${this.apiUrl}/rate`;
    const body = { user_id: userId, movie_id: movieId, rating: rating };
    return this.http.post<any>(url, body, { headers });
  }
  evaluateModel(): Observable<any> {
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`,
      'Content-Type': 'application/json'
    });
    const url = `${this.apiUrl}/evaluate_model`;
    // const body = { user_id: userId, movie_id: movieId, rating: rating };
    return this.http.get<any>(url, { headers });
  }


  // Get recommendations for a user
  getRecommendations(userId: number): Observable<any> {
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });
    const url = `${this.apiUrl}/recommendations/${userId}`;
    return this.http.get<any>(url, { headers });
  }

  // Get the list of movies (movie_id and movie_name) for the dropdown
  getMovieList(): Observable<any[]> {
    const url = `${this.apiUrl}/movies`;
    return this.http.get<any[]>(url);
  }
}
