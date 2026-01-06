"""
Preprocessing module for building user-item matrices.

This module creates sparse matrices for collaborative filtering.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_user_item_matrix(df: pd.DataFrame, 
                          rating_column: Optional[str] = None,
                          normalize: bool = False) -> Tuple[csr_matrix, Dict[str, int], Dict[str, int], pd.DataFrame]:
    """
    Build a sparse user-item interaction matrix.
    
    Args:
        df: DataFrame with columns user_id, track_id, and rating/play_count
        rating_column: Name of rating column (auto-detected if None)
        normalize: Whether to normalize ratings per user (0-1 scaling)
        
    Returns:
        Tuple of:
        - user_item_matrix: Sparse matrix (users x tracks)
        - user_to_idx: Dictionary mapping user_id to matrix index
        - track_to_idx: Dictionary mapping track_id to matrix index
        - metadata_df: DataFrame with track metadata (if available)
    """
    logger.info("Building user-item matrix...")
    
    # Auto-detect rating column
    if rating_column is None:
        if 'rating' in df.columns:
            rating_column = 'rating'
        elif 'play_count' in df.columns:
            rating_column = 'play_count'
        else:
            raise ValueError("Could not find rating or play_count column")
    
    # Create mappings
    unique_users = df['user_id'].unique()
    unique_tracks = df['track_id'].unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(sorted(unique_users))}
    track_to_idx = {track: idx for idx, track in enumerate(sorted(unique_tracks))}
    
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_track = {idx: track for track, idx in track_to_idx.items()}
    
    logger.info(f"Found {len(unique_users)} unique users and {len(unique_tracks)} unique tracks")
    
    # Create matrix indices
    user_indices = df['user_id'].map(user_to_idx).values
    track_indices = df['track_id'].map(track_to_idx).values
    ratings = df[rating_column].values
    
    # Normalize ratings per user if requested
    if normalize:
        user_ratings = df.groupby('user_id')[rating_column]
        user_max = user_ratings.max()
        user_min = user_ratings.min()
        
        def normalize_rating(row):
            user = row['user_id']
            rating = row[rating_column]
            max_r = user_max[user]
            min_r = user_min[user]
            if max_r == min_r:
                return 1.0
            return (rating - min_r) / (max_r - min_r)
        
        ratings = df.apply(normalize_rating, axis=1).values
    
    # Build sparse matrix
    user_item_matrix = csr_matrix(
        (ratings, (user_indices, track_indices)),
        shape=(len(unique_users), len(unique_tracks))
    )
    
    logger.info(f"Created sparse matrix: {user_item_matrix.shape} with {user_item_matrix.nnz} non-zero entries")
    
    # Extract metadata if available
    metadata_columns = ['track_id', 'song_name', 'artist_name']
    available_metadata = [col for col in metadata_columns if col in df.columns]
    
    if available_metadata:
        metadata_df = df[available_metadata].drop_duplicates(subset=['track_id']).set_index('track_id')
    else:
        metadata_df = pd.DataFrame(index=unique_tracks)
    
    return user_item_matrix, user_to_idx, track_to_idx, metadata_df


def get_user_vector(user_id: str, 
                   user_to_idx: Dict[str, int],
                   user_item_matrix: csr_matrix) -> Optional[np.ndarray]:
    """
    Get the interaction vector for a specific user.
    
    Args:
        user_id: User identifier
        user_to_idx: User to index mapping
        user_item_matrix: User-item matrix
        
    Returns:
        User's interaction vector (1D array) or None if user not found
    """
    if user_id not in user_to_idx:
        return None
    
    user_idx = user_to_idx[user_id]
    return user_item_matrix[user_idx].toarray().flatten()


def get_track_vector(track_id: str,
                    track_to_idx: Dict[str, int],
                    user_item_matrix: csr_matrix) -> Optional[np.ndarray]:
    """
    Get the interaction vector for a specific track (item-based).
    
    Args:
        track_id: Track identifier
        track_to_idx: Track to index mapping
        user_item_matrix: User-item matrix
        
    Returns:
        Track's interaction vector (1D array) or None if track not found
    """
    if track_id not in track_to_idx:
        return None
    
    track_idx = track_to_idx[track_id]
    # Transpose to get item-based view
    item_user_matrix = user_item_matrix.T
    return item_user_matrix[track_idx].toarray().flatten()


if __name__ == "__main__":
    # Example usage
    try:
        from data_loader import load_and_clean_data
        
        df = load_and_clean_data("data/music_data.csv")
        matrix, user_map, track_map, metadata = build_user_item_matrix(df)
        
        print(f"Matrix shape: {matrix.shape}")
        print(f"Number of users: {len(user_map)}")
        print(f"Number of tracks: {len(track_map)}")
    except FileNotFoundError:
        print("No data file found. Please provide a CSV file first.")

