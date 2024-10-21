# Step 1: Build the Angular app
FROM node:18-alpine 

# Set the working directory to /app
WORKDIR /app

# Copy package.json and package-lock.json to /app
COPY package*.json ./

# Install app dependencies
RUN npm install

# Copy app source code to /app
COPY . .

# Build the app for production
RUN npm run build --prod

# Expose port 80 for the container
EXPOSE 4200

# Start the app
CMD ["ng", "serve", "--host", "0.0.0.0"]