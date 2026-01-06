# 🎵 Music Recommendation System

A production-ready music recommendation system using **collaborative filtering** with k-nearest neighbors (k-NN) and cosine similarity. This system provides personalized song recommendations based on user listening patterns.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Format](#data-format)
- [Usage](#usage)
  - [Command Line](#command-line)
  - [Python API](#python-api)
  - [Streamlit Web App](#streamlit-web-app)
  - [Jupyter Notebook](#jupyter-notebook)
- [How It Works](#how-it-works)
- [Examples](#examples)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **User-Based Recommendations**: Recommend songs to users based on similar users' preferences
- **Item-Based Recommendations**: Find songs similar to a given track
- **Efficient Sparse Matrix Operations**: Uses scipy sparse matrices for memory efficiency
- **Clean, Modular Code**: Well-structured, production-ready codebase with type hints
- **Interactive Web UI**: Beautiful Streamlit interface for easy interaction
- **Comprehensive Documentation**: Detailed docstrings and examples

## 📁 Project Structure

```
Music Recommendation System/
│
├── data_loader.py          # Data loading and cleaning
├── preprocess.py           # User-item matrix construction
├── model.py                # k-NN recommender model
├── recommend.py            # High-level recommendation interface
├── app.py                  # Streamlit web application
├── prepare_data.py         # Convert lyrics data to interaction format
├── main.ipynb              # Demo Jupyter notebook
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
└── data/                   # Data directory
    ├── music_data.csv      # Your music interaction data (generated)
    └── spotify_millsongdata.csv  # Lyrics dataset (optional)
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**:
   
   **Option A: Use existing interaction data**
   - Create a `data/` directory
   - Place your CSV file with music interactions in `data/music_data.csv`
   - See [Data Format](#data-format) for required columns
   
   **Option B: Convert lyrics/song metadata to interaction data**
   - If you have a lyrics dataset (like Spotify lyrics) with columns: `artist`, `song`
   - Run the data preparation script:
     ```bash
     python prepare_data.py data/your_lyrics_file.csv data/music_data.csv
     ```
   - This will generate synthetic user interactions from your song catalog
   - Default: `python prepare_data.py` (uses `data/spotify_millsongdata.csv`)

## 📊 Data Format

Your CSV file should contain the following columns:

### Required Columns
- `user_id`: Unique identifier for each user (string or numeric)
- `track_id`: Unique identifier for each track/song (string or numeric)
- `rating` OR `play_count`: User's interaction with the track
  - `rating`: Numeric rating (typically 1-5)
  - `play_count`: Number of times user played the track

### Optional Columns
- `song_name`: Name of the song
- `artist_name`: Name of the artist

### Example CSV

```csv
user_id,track_id,rating,song_name,artist_name
user1,track1,5,Bohemian Rhapsody,Queen
user1,track2,4,Stairway to Heaven,Led Zeppelin
user2,track1,5,Bohemian Rhapsody,Queen
user2,track3,3,Hotel California,The Eagles
user3,track2,5,Stairway to Heaven,Led Zeppelin
user3,track3,4,Hotel California,The Eagles
```

## 💻 Usage

### Command Line

#### Quick Recommendations

```bash
# Get recommendations for a user
python recommend.py data/music_data.csv user1

# Find similar tracks
python recommend.py data/music_data.csv track1
```

#### View System Statistics

```bash
python recommend.py data/music_data.csv
```

### Python API

#### Basic Usage

```python
from recommend import RecommendationEngine

# Initialize the engine (loads data and trains model)
engine = RecommendationEngine("data/music_data.csv", n_neighbors=10)

# Get recommendations for a user
recommendations = engine.recommend_for_user("user1", n_recommendations=10)
print(recommendations)

# Find similar tracks
similar_tracks = engine.recommend_similar_tracks("track1", n_recommendations=10)
print(similar_tracks)

# Get system statistics
stats = engine.get_user_stats()
print(stats)
```

#### Advanced Usage

```python
from data_loader import load_and_clean_data
from preprocess import build_user_item_matrix
from model import MusicRecommender

# Step 1: Load and clean data
df = load_and_clean_data("data/music_data.csv", min_interactions=2)

# Step 2: Build user-item matrix
matrix, user_map, track_map, metadata = build_user_item_matrix(df)

# Step 3: Train model
recommender = MusicRecommender(n_neighbors=10, metric='cosine')
recommender.fit(matrix, user_map, track_map, metadata)

# Step 4: Get recommendations
user_recs = recommender.recommend_for_user("user1", n_recommendations=10)
track_recs = recommender.recommend_similar_tracks("track1", n_recommendations=10)

# Step 5: Save model for later use
recommender.save("model.pkl")

# Step 6: Load saved model
new_recommender = MusicRecommender()
new_recommender.load("model.pkl")
```

### Streamlit Web App

Launch the interactive web interface:

```bash
streamlit run app.py
```

The app will open in your browser. You can:
- Upload a CSV file or use an existing file path
- Get recommendations for users
- Find similar tracks
- View system statistics
- Download recommendations as CSV

### Jupyter Notebook

Open `main.ipynb` in Jupyter to explore the system interactively:

```bash
jupyter notebook main.ipynb
```

The notebook includes:
- Step-by-step data loading and exploration
- Model training visualization
- Recommendation examples
- Batch processing examples

## 🔄 Data Preparation

If you have a lyrics dataset (with `artist` and `song` columns) instead of user interactions, use the `prepare_data.py` script:

```bash
# Basic usage (uses default paths)
python prepare_data.py

# Custom input/output files
python prepare_data.py data/your_lyrics.csv data/output_interactions.csv

# With custom parameters (in Python)
python -c "from prepare_data import prepare_interaction_data; prepare_interaction_data('data/lyrics.csv', 'data/music_data.csv', n_users=2000, min_interactions_per_user=10)"
```

The script generates realistic synthetic user interactions with:
- Configurable number of users (default: 1000)
- Variable interactions per user (default: 5-50)
- Realistic rating distribution based on song popularity
- Preserves song and artist metadata

**Example**: If you have `data/spotify_millsongdata.csv` with lyrics data, running `python prepare_data.py` will create `data/music_data.csv` with 1000 users and their interactions.

## 🔧 How It Works

### Collaborative Filtering

The system uses **collaborative filtering**, which recommends items based on patterns of user behavior:

1. **User-Item Matrix**: Converts user-track interactions into a sparse matrix
2. **Similarity Calculation**: Uses cosine similarity to find similar users/tracks
3. **k-Nearest Neighbors**: Finds k most similar users/tracks
4. **Recommendation Generation**: Aggregates preferences from similar users/tracks

### Algorithm Details

#### User-Based Recommendations
1. Find k users most similar to the target user
2. Identify tracks these similar users liked
3. Score tracks based on similarity and interaction strength
4. Return top-N recommendations

#### Item-Based Recommendations
1. Find k tracks most similar to the target track
2. Similarity based on users who interacted with both tracks
3. Return top-N most similar tracks

### Technical Implementation

- **Sparse Matrices**: Uses `scipy.sparse.csr_matrix` for memory efficiency
- **k-NN**: Implemented using `sklearn.neighbors.NearestNeighbors`
- **Cosine Similarity**: Default metric for measuring similarity
- **Scalable**: Handles large datasets efficiently

## 📝 Examples

### Example 1: Basic User Recommendations

```python
from recommend import RecommendationEngine

engine = RecommendationEngine("data/music_data.csv")
recs = engine.recommend_for_user("user1", n_recommendations=5)

print("Top 5 Recommendations:")
for track_id, score in recs:
    print(f"  {track_id}: {score:.4f}")
```

### Example 2: Find Similar Songs

```python
similar = engine.recommend_similar_tracks("track1", n_recommendations=5)
print("Similar Tracks:")
for track_id, score in similar:
    print(f"  {track_id}: {score:.4f}")
```

### Example 3: Batch Processing

```python
user_ids = ["user1", "user2", "user3"]
all_recommendations = {}

for user_id in user_ids:
    recs = engine.recommend_for_user(user_id, n_recommendations=10)
    all_recommendations[user_id] = recs
```

## ⚙️ Configuration

### Model Parameters

```python
recommender = MusicRecommender(
    n_neighbors=10,      # Number of neighbors (default: 10)
    metric='cosine',     # Similarity metric: 'cosine', 'euclidean', etc.
    algorithm='brute'    # Algorithm: 'brute', 'ball_tree', 'kd_tree'
)
```

### Data Cleaning Parameters

```python
df = load_and_clean_data(
    "data/music_data.csv",
    min_interactions=1,      # Minimum interactions per user/track
    rating_column='rating'   # Name of rating column (auto-detected if None)
)
```

### Preprocessing Options

```python
matrix, user_map, track_map, metadata = build_user_item_matrix(
    df,
    rating_column='rating',  # Rating column name
    normalize=False          # Normalize ratings per user (0-1 scaling)
)
```

## 🐛 Troubleshooting

### Common Issues

#### 1. File Not Found Error
```
FileNotFoundError: File not found: data/music_data.csv
```
**Solution**: Ensure your CSV file exists in the `data/` directory, or update the file path.

#### 2. No Recommendations Found
```
No recommendations found for this user.
```
**Possible causes**:
- User has no interactions in the dataset
- User is too unique (no similar users found)
- Dataset is too small

**Solutions**:
- Reduce `min_interactions` parameter
- Increase `n_neighbors` parameter
- Add more data to your dataset

#### 3. Memory Issues with Large Datasets
**Solutions**:
- Use `min_interactions` to filter sparse users/tracks
- Process data in batches
- Consider using a more powerful machine or cloud computing

#### 4. Import Errors
```
ModuleNotFoundError: No module named 'pandas'
```
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Performance Tips

1. **Filter Sparse Data**: Use `min_interactions` to remove users/tracks with few interactions
2. **Adjust k**: Smaller `n_neighbors` = faster, but potentially less accurate
3. **Use Sparse Matrices**: The system automatically uses sparse matrices for efficiency
4. **Save Models**: Train once and save the model to avoid retraining

## 📚 Additional Resources

- [scikit-learn NearestNeighbors Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
- [Collaborative Filtering Explained](https://en.wikipedia.org/wiki/Collaborative_filtering)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available for educational and commercial use.

## 👤 Author

Created as a production-ready music recommendation system using collaborative filtering.

---

**Happy Recommending! 🎵**

