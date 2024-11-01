# 🎬 MovieScout: Real-Time Movie Recommendation System

**MovieScout** is a recommendation system that predicts movie ratings using collaborative filtering, deep learning, and real-time monitoring. This solution is built with PyTorch, Prometheus, and Angular for a complete development experience.

## 🌟 Key Features
- **Collaborative Filtering**: Recommendations using matrix factorization.
- **Prometheus Monitoring**: Real-time tracking with Flask metrics.
- **Angular Frontend**: Intuitive, interactive user interface.

## 🛠️ Tech Stack
- **Backend**: Python, Flask, PyTorch
- **Frontend**: Angular
- **Monitoring**: Prometheus, Grafana

---

## 🚀 Getting Started (Manual Setup)

### 1. Clone the Repository
```bash
git clone https://github.com/ashish493/MovieScout.git
cd MovieScout
```

### 2. Set Up the Backend

1. **Python Environment**  
   Create and activate a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate  # on Windows, use `env\Scripts\activate`
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Flask API**  
   Run the Flask application:
   ```bash
   export FLASK_APP=api.py
   flask run
   ```

### 3. Set Up the Frontend

1. **Install Angular CLI** (if not installed):
   ```bash
   npm install -g @angular/cli
   ```

2. **Navigate to the `ui/` Directory** and install dependencies:
   ```bash
   cd ui
   npm install
   ```

3. **Serve the Angular App**:
   ```bash
   ng serve
   ```
   Access the app at `http://localhost:4200`.

### 4. Set Up Prometheus Monitoring

1. **Install Prometheus** from [Prometheus Downloads](https://prometheus.io/download/).
2. **Configure Prometheus**  
   Update `prometheus.yml` to include:
   ```yaml
   - job_name: 'flask_app'
     static_configs:
       - targets: ['localhost:5000']
   ```

3. **Start Prometheus**:
   ```bash
   ./prometheus --config.file=prometheus.yml
   ```
   Access Prometheus at `http://localhost:9090`.

## 📜 License
This project is licensed under the MIT License.