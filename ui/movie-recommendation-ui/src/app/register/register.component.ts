import { Component } from '@angular/core';
import { MovieService } from '../movie.service';

@Component({
  selector: 'app-register',
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.css']
})
export class RegisterComponent {
  username: string = '';
  password: string = '';
  registrationMessage: string = '';

  constructor(private movieService: MovieService) { }

  // Register a new user
  registerUser() {
    this.movieService.register(this.username, this.password).subscribe(
      response => {
        this.registrationMessage = 'Registration successful!';
      },
      error => {
        console.error('Error registering user:', error);
        this.registrationMessage = 'Error: User already exists or invalid input';
      }
    );
  }
}