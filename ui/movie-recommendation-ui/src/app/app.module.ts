import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';  // FormsModule for ngModel
import { HttpClientModule } from '@angular/common/http';  // HttpClientModule for API calls

import { AppComponent } from './app.component';
import { MovieRatingComponent } from './movie-rating/movie-rating.component';
import { RouterModule } from '@angular/router';
import { AppRoutingModule } from './app-routing.module';
import { RegisterComponent } from './register/register.component';

@NgModule({
  declarations: [
    AppComponent,
    MovieRatingComponent,
    RegisterComponent
  ],
  imports: [
    BrowserModule,
    FormsModule,  // Add FormsModule here
    HttpClientModule,  // Add HttpClientModule here
    RouterModule,
    AppRoutingModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }