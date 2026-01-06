"""
Model training module for k-NN collaborative filtering.

This module implements k-nearest neighbors recommendation using cosine similarity.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict, List, Optional
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MusicRecommender:
    """
    k-NN based music recommender using collaborative filtering.
    
    Supports both user-based and item-based recommendations.
    """
    
    def __init__(self, n_neighbors: int = 10, metric: str = 'cosine', algorithm: str = 'brute'):
        """
        Initialize the recommender.
        
        Args:
            n_neighbors: Number of neighbors to consider
            metric: Distance metric ('cosine', 'euclidean', etc.)
            algorithm: Algorithm for nearest neighbors ('brute', 'ball_tree', 'kd_tree')
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        self.user_model: Optional[NearestNeighbors] = None
        self.item_model: Optional[NearestNeighbors] = None
        self.user_item_matrix: Optional[csr_matrix] = None
        self.user_to_idx: Optional[Dict[str, int]] = None
        self.track_to_idx: Optional[Dict[str, int]] = None
        self.idx_to_user: Optional[Dict[int, str]] = None
        self.idx_to_track: Optional[Dict[int, str]] = None
        self.metadata_df = None
        
    def fit(self, 
            user_item_matrix: csr_matrix,
            user_to_idx: Dict[str, int],
            track_to_idx: Dict[str, int],
            metadata_df: Optional[pd.DataFrame] = None):
        """
        Train the recommendation models.
        
        Args:
            user_item_matrix: Sparse user-item interaction matrix
            user_to_idx: Mapping from user_id to matrix index
            track_to_idx: Mapping from track_id to matrix index
            metadata_df: Optional DataFrame with track metadata
        """
        logger.info("Training recommendation models...")
        
        self.user_item_matrix = user_item_matrix
        self.user_to_idx = user_to_idx
        self.track_to_idx = track_to_idx
        self.idx_to_user = {idx: user for user, idx in user_to_idx.items()}
        self.idx_to_track = {idx: track for track, idx in track_to_idx.items()}
        self.metadata_df = metadata_df
        
        # Train user-based model
        logger.info("Training user-based model...")
        self.user_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, user_item_matrix.shape[0]),
            metric=self.metric,
            algorithm=self.algorithm
        )
        self.user_model.fit(user_item_matrix)
        
        # Train item-based model (transpose matrix)
        logger.info("Training item-based model...")
        item_user_matrix = user_item_matrix.T
        self.item_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, item_user_matrix.shape[0]),
            metric=self.metric,
            algorithm=self.algorithm
        )
        self.item_model.fit(item_user_matrix)
        
        logger.info("Model training complete!")
    
    def recommend_for_user(self, 
                          user_id: str, 
                          n_recommendations: int = 10,
                          exclude_interacted: bool = True) -> List[Tuple[str, float]]:
        """
        Recommend tracks for a given user.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            exclude_interacted: Whether to exclude tracks the user has already interacted with
            
        Returns:
            List of tuples (track_id, similarity_score) sorted by score descending
        """
        if self.user_model is None or self.user_item_matrix is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        user_idx = self.user_to_idx[user_id]
        user_vector = self.user_item_matrix[user_idx]
        
        # Find similar users
        distances, indices = self.user_model.kneighbors(user_vector, n_neighbors=self.n_neighbors + 1)
        
        # Get tracks from similar users (excluding the user themselves)
        similar_user_indices = indices[0][1:]  # Skip first (self)
        similar_user_distances = distances[0][1:]
        
        # Aggregate recommendations from similar users
        track_scores = {}
        user_interactions = set(user_vector.indices) if exclude_interacted else set()
        
        for similar_idx, distance in zip(similar_user_indices, similar_user_distances):
            similarity = 1 - distance  # Convert distance to similarity
            similar_user_vector = self.user_item_matrix[similar_idx]
            
            # Add scores for tracks this similar user interacted with
            for track_idx in similar_user_vector.indices:
                if track_idx not in user_interactions:
                    if track_idx not in track_scores:
                        track_scores[track_idx] = 0
                    track_scores[track_idx] += similarity * similar_user_vector[0, track_idx]
        
        # Sort by score and return top N
        sorted_tracks = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [
            (self.idx_to_track[track_idx], score) 
            for track_idx, score in sorted_tracks[:n_recommendations]
        ]
        
        return recommendations
    
    def recommend_similar_tracks(self, 
                                track_id: str, 
                                n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """
        Recommend tracks similar to a given track (item-based).
        
        Args:
            track_id: Track identifier
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of tuples (track_id, similarity_score) sorted by score descending
        """
        if self.item_model is None or self.user_item_matrix is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if track_id not in self.track_to_idx:
            logger.warning(f"Track {track_id} not found in training data")
            return []
        
        track_idx = self.track_to_idx[track_id]
        item_user_matrix = self.user_item_matrix.T
        track_vector = item_user_matrix[track_idx]
        
        # Find similar tracks
        distances, indices = self.item_model.kneighbors(track_vector, n_neighbors=self.n_neighbors + 1)
        
        # Convert to recommendations (excluding the track itself)
        similar_track_indices = indices[0][1:]  # Skip first (self)
        similar_track_distances = distances[0][1:]
        
        recommendations = [
            (self.idx_to_track[idx], 1 - dist)  # Convert distance to similarity
            for idx, dist in zip(similar_track_indices, similar_track_distances)
        ]
        
        # Sort by similarity and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def save(self, filepath: str):
        """Save the trained model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            self.__dict__.update(pickle.load(f))
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    try:
        import pandas as pd
        from data_loader import load_and_clean_data
        from preprocess import build_user_item_matrix
        
        # Load and preprocess data
        df = load_and_clean_data("data/music_data.csv")
        matrix, user_map, track_map, metadata = build_user_item_matrix(df)
        
        # Train model
        recommender = MusicRecommender(n_neighbors=10)
        recommender.fit(matrix, user_map, track_map, metadata)
        
        # Test recommendations
        if len(user_map) > 0:
            test_user = list(user_map.keys())[0]
            print(f"\nRecommendations for user {test_user}:")
            recs = recommender.recommend_for_user(test_user, n_recommendations=5)
            for track_id, score in recs:
                print(f"  {track_id}: {score:.4f}")
        
        if len(track_map) > 0:
            test_track = list(track_map.keys())[0]
            print(f"\nSimilar tracks to {test_track}:")
            recs = recommender.recommend_similar_tracks(test_track, n_recommendations=5)
            for track_id, score in recs:
                print(f"  {track_id}: {score:.4f}")
                
    except FileNotFoundError:
        print("No data file found. Please provide a CSV file first.")

