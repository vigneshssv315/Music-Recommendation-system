"""
Script to convert Spotify lyrics dataset into user-item interaction format.

This script takes a lyrics dataset (artist, song) and generates synthetic
user interactions for collaborative filtering.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_interaction_data(
    lyrics_file: str,
    output_file: str = "data/music_data.csv",
    n_users: int = 1000,
    min_interactions_per_user: int = 5,
    max_interactions_per_user: int = 50,
    rating_range: tuple = (1, 5),
    seed: int = 42
) -> pd.DataFrame:
    """
    Convert lyrics dataset into user-item interaction format.
    
    This function generates synthetic user interactions based on the songs
    in the lyrics dataset. It simulates realistic user behavior where:
    - Some songs are more popular (more users interact with them)
    - Users have varying numbers of interactions
    - Ratings follow a realistic distribution
    
    Args:
        lyrics_file: Path to the lyrics CSV file
        output_file: Path to save the interaction data
        n_users: Number of synthetic users to create
        min_interactions_per_user: Minimum interactions per user
        max_interactions_per_user: Maximum interactions per user
        rating_range: Tuple of (min_rating, max_rating)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with user_id, track_id, rating, song_name, artist_name
    """
    logger.info(f"Loading lyrics data from {lyrics_file}...")
    
    # Load lyrics data
    try:
        lyrics_df = pd.read_csv(lyrics_file)
        logger.info(f"Loaded {len(lyrics_df)} songs")
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        raise
    
    # Check required columns
    required_cols = ['artist', 'song']
    missing_cols = [col for col in required_cols if col not in lyrics_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean and prepare song data
    lyrics_df = lyrics_df.dropna(subset=['artist', 'song'])
    lyrics_df = lyrics_df.drop_duplicates(subset=['artist', 'song'])
    
    # Create track_id (unique identifier for each song)
    lyrics_df['track_id'] = lyrics_df.apply(
        lambda row: f"{row['artist'].strip().replace(' ', '_')}_{row['song'].strip().replace(' ', '_')}",
        axis=1
    )
    lyrics_df['track_id'] = lyrics_df['track_id'].str.replace(r'[^\w\-_]', '', regex=True)
    
    # Remove duplicates based on track_id
    lyrics_df = lyrics_df.drop_duplicates(subset=['track_id'])
    
    logger.info(f"Prepared {len(lyrics_df)} unique tracks")
    
    # Set random seed
    np.random.seed(seed)
    
    # Generate synthetic user interactions
    logger.info(f"Generating interactions for {n_users} users...")
    
    interactions = []
    track_ids = lyrics_df['track_id'].values
    n_tracks = len(track_ids)
    
    # Create popularity distribution (some songs are more popular)
    # Use a power law distribution to simulate realistic popularity
    popularity_weights = np.power(np.arange(1, n_tracks + 1), -0.7)
    popularity_weights = popularity_weights / popularity_weights.sum()
    
    for user_id in range(1, n_users + 1):
        # Random number of interactions for this user
        n_interactions = np.random.randint(min_interactions_per_user, max_interactions_per_user + 1)
        
        # Sample tracks based on popularity
        sampled_track_indices = np.random.choice(
            n_tracks,
            size=min(n_interactions, n_tracks),
            replace=False,
            p=popularity_weights
        )
        
        for track_idx in sampled_track_indices:
            track_id = track_ids[track_idx]
            
            # Generate rating (higher ratings for more popular songs, with some randomness)
            base_rating = 3 + (1 - popularity_weights[track_idx] / popularity_weights.max()) * 1.5
            rating = np.clip(
                np.random.normal(base_rating, 0.8),
                rating_range[0],
                rating_range[1]
            )
            rating = int(np.round(rating))
            
            interactions.append({
                'user_id': f'user_{user_id}',
                'track_id': track_id,
                'rating': rating
            })
    
    # Create interactions DataFrame
    interactions_df = pd.DataFrame(interactions)
    
    # Merge with song metadata
    track_metadata = lyrics_df[['track_id', 'song', 'artist']].rename(columns={
        'song': 'song_name',
        'artist': 'artist_name'
    })
    
    final_df = interactions_df.merge(track_metadata, on='track_id', how='left')
    
    # Reorder columns
    final_df = final_df[['user_id', 'track_id', 'rating', 'song_name', 'artist_name']]
    
    # Remove duplicates (same user-track pair)
    final_df = final_df.drop_duplicates(subset=['user_id', 'track_id'], keep='first')
    
    logger.info(f"Generated {len(final_df)} interactions")
    logger.info(f"  Unique users: {final_df['user_id'].nunique()}")
    logger.info(f"  Unique tracks: {final_df['track_id'].nunique()}")
    logger.info(f"  Average interactions per user: {len(final_df) / final_df['user_id'].nunique():.2f}")
    
    # Save to CSV
    final_df.to_csv(output_file, index=False)
    logger.info(f"Saved interaction data to {output_file}")
    
    return final_df


if __name__ == "__main__":
    import sys
    
    # Default paths
    lyrics_file = "data/spotify_millsongdata.csv"
    output_file = "data/music_data.csv"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        lyrics_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    try:
        df = prepare_interaction_data(
            lyrics_file=lyrics_file,
            output_file=output_file,
            n_users=1000,
            min_interactions_per_user=5,
            max_interactions_per_user=50
        )
        
        print("\nData preparation complete!")
        print(f"\nSample data:")
        print(df.head(10))
        print(f"\nStatistics:")
        print(f"  Total interactions: {len(df)}")
        print(f"  Unique users: {df['user_id'].nunique()}")
        print(f"  Unique tracks: {df['track_id'].nunique()}")
        print(f"  Rating distribution:")
        print(df['rating'].value_counts().sort_index())
        
    except FileNotFoundError:
        print(f"Error: File not found: {lyrics_file}")
        print(f"Please ensure the file exists in the data directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

