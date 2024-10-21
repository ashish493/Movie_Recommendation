import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { MovieRatingComponent } from './movie-rating/movie-rating.component';
import { RegisterComponent } from './register/register.component';

const routes: Routes = [
  { path: '', redirectTo: '/rate-movie', pathMatch: 'full' },
  { path: 'rate-movie', component: MovieRatingComponent },
  { path: 'register', component: RegisterComponent }  // Add the register route here
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
