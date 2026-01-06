"""
Data loading and cleaning module for Music Recommendation System.

This module handles loading CSV data, cleaning, and basic validation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load music recommendation data from CSV file.
    
    Args:
        file_path: Path to the CSV file containing user-track interactions
        
    Returns:
        DataFrame with columns: user_id, track_id, rating/play_count, 
        and optionally song_name, artist_name
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If required columns are missing
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data from {file_path}: {len(df)} rows")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate that the DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_columns = ['user_id', 'track_id']
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for rating or play_count
    has_rating = 'rating' in df.columns
    has_play_count = 'play_count' in df.columns
    
    if not (has_rating or has_play_count):
        raise ValueError("DataFrame must contain either 'rating' or 'play_count' column")
    
    return True


def clean_data(df: pd.DataFrame, 
               min_interactions: int = 1,
               rating_column: Optional[str] = None) -> pd.DataFrame:
    """
    Clean and preprocess the data.
    
    Args:
        df: Raw DataFrame
        min_interactions: Minimum number of interactions per user/track to keep
        rating_column: Name of the rating/play_count column. If None, auto-detect.
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting data cleaning...")
    original_size = len(df)
    
    # Auto-detect rating column
    if rating_column is None:
        if 'rating' in df.columns:
            rating_column = 'rating'
        elif 'play_count' in df.columns:
            rating_column = 'play_count'
        else:
            raise ValueError("Could not find rating or play_count column")
    
    # Remove rows with missing user_id or track_id
    df = df.dropna(subset=['user_id', 'track_id'])
    
    # Remove rows with invalid ratings (negative or zero if rating, zero if play_count)
    if rating_column == 'rating':
        df = df[df[rating_column] > 0]
    else:
        df = df[df[rating_column] >= 0]
    
    # Convert user_id and track_id to string for consistency
    df['user_id'] = df['user_id'].astype(str)
    df['track_id'] = df['track_id'].astype(str)
    
    # Filter users with minimum interactions
    if min_interactions > 1:
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        df = df[df['user_id'].isin(valid_users)]
        
        # Filter tracks with minimum interactions
        track_counts = df['track_id'].value_counts()
        valid_tracks = track_counts[track_counts >= min_interactions].index
        df = df[df['track_id'].isin(valid_tracks)]
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['user_id', 'track_id'], keep='first')
    
    logger.info(f"Data cleaning complete: {original_size} -> {len(df)} rows")
    return df


def load_and_clean_data(file_path: str, 
                       min_interactions: int = 1,
                       rating_column: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to load and clean data in one step.
    
    Args:
        file_path: Path to the CSV file
        min_interactions: Minimum interactions per user/track
        rating_column: Name of rating column (auto-detected if None)
        
    Returns:
        Cleaned DataFrame ready for preprocessing
    """
    df = load_data(file_path)
    validate_data(df)
    df = clean_data(df, min_interactions=min_interactions, rating_column=rating_column)
    return df


if __name__ == "__main__":
    # Example usage
    try:
        # This will fail if no data file exists, which is expected
        df = load_and_clean_data("data/music_data.csv")
        print(f"Loaded and cleaned data: {len(df)} rows")
        print(df.head())
    except FileNotFoundError:
        print("No data file found. Please provide a CSV file with columns: user_id, track_id, rating/play_count")

