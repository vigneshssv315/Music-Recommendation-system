"""
High-level recommendation interface.

This module provides easy-to-use functions for getting recommendations.
"""

from typing import List, Tuple, Optional, Dict
import pandas as pd
from model import MusicRecommender
from data_loader import load_and_clean_data
from preprocess import build_user_item_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    High-level interface for the music recommendation system.
    """
    
    def __init__(self, data_path: str, n_neighbors: int = 10, min_interactions: int = 1):
        """
        Initialize the recommendation engine.
        
        Args:
            data_path: Path to CSV file with user-track interactions
            n_neighbors: Number of neighbors for k-NN
            min_interactions: Minimum interactions per user/track
        """
        logger.info(f"Initializing recommendation engine with data from {data_path}")
        
        # Load and preprocess data
        self.df = load_and_clean_data(data_path, min_interactions=min_interactions)
        self.matrix, self.user_map, self.track_map, self.metadata = build_user_item_matrix(self.df)
        
        # Train model
        self.recommender = MusicRecommender(n_neighbors=n_neighbors)
        self.recommender.fit(self.matrix, self.user_map, self.track_map, self.metadata)
        
        logger.info("Recommendation engine ready!")
    
    def recommend_for_user(self, 
                          user_id: str, 
                          n_recommendations: int = 10,
                          include_metadata: bool = True) -> pd.DataFrame:
        """
        Get recommendations for a user.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations
            include_metadata: Whether to include song/artist names if available
            
        Returns:
            DataFrame with recommendations (track_id, score, song_name, artist_name)
        """
        recommendations = self.recommender.recommend_for_user(
            str(user_id), 
            n_recommendations=n_recommendations
        )
        
        if not recommendations:
            return pd.DataFrame(columns=['track_id', 'score', 'song_name', 'artist_name'])
        
        # Create results DataFrame
        results = pd.DataFrame(recommendations, columns=['track_id', 'score'])
        
        # Add metadata if available
        if include_metadata and self.metadata is not None and not self.metadata.empty:
            if 'song_name' in self.metadata.columns:
                results = results.merge(
                    self.metadata[['song_name']], 
                    left_on='track_id', 
                    right_index=True, 
                    how='left'
                )
            if 'artist_name' in self.metadata.columns:
                results = results.merge(
                    self.metadata[['artist_name']], 
                    left_on='track_id', 
                    right_index=True, 
                    how='left'
                )
        
        return results
    
    def recommend_similar_tracks(self, 
                                track_id: str, 
                                n_recommendations: int = 10,
                                include_metadata: bool = True) -> pd.DataFrame:
        """
        Get tracks similar to a given track.
        
        Args:
            track_id: Track identifier
            n_recommendations: Number of recommendations
            include_metadata: Whether to include song/artist names if available
            
        Returns:
            DataFrame with recommendations (track_id, score, song_name, artist_name)
        """
        recommendations = self.recommender.recommend_similar_tracks(
            str(track_id), 
            n_recommendations=n_recommendations
        )
        
        if not recommendations:
            return pd.DataFrame(columns=['track_id', 'score', 'song_name', 'artist_name'])
        
        # Create results DataFrame
        results = pd.DataFrame(recommendations, columns=['track_id', 'score'])
        
        # Add metadata if available
        if include_metadata and self.metadata is not None and not self.metadata.empty:
            if 'song_name' in self.metadata.columns:
                results = results.merge(
                    self.metadata[['song_name']], 
                    left_on='track_id', 
                    right_index=True, 
                    how='left'
                )
            if 'artist_name' in self.metadata.columns:
                results = results.merge(
                    self.metadata[['artist_name']], 
                    left_on='track_id', 
                    right_index=True, 
                    how='left'
                )
        
        return results
    
    def get_user_stats(self) -> Dict:
        """Get statistics about users in the system."""
        return {
            'total_users': len(self.user_map),
            'total_tracks': len(self.track_map),
            'total_interactions': self.matrix.nnz,
            'avg_interactions_per_user': self.matrix.nnz / len(self.user_map) if len(self.user_map) > 0 else 0
        }
    
    def get_track_info(self, track_id: str) -> Optional[Dict]:
        """Get information about a specific track."""
        if track_id not in self.track_map:
            return None
        
        info = {'track_id': track_id}
        
        if self.metadata is not None and not self.metadata.empty and track_id in self.metadata.index:
            if 'song_name' in self.metadata.columns:
                info['song_name'] = self.metadata.loc[track_id, 'song_name']
            if 'artist_name' in self.metadata.columns:
                info['artist_name'] = self.metadata.loc[track_id, 'artist_name']
        
        return info


def quick_recommend(data_path: str, 
                   user_id: Optional[str] = None,
                   track_id: Optional[str] = None,
                   n_recommendations: int = 10) -> pd.DataFrame:
    """
    Quick function to get recommendations without creating an engine instance.
    
    Args:
        data_path: Path to CSV file
        user_id: User ID for user-based recommendations
        track_id: Track ID for item-based recommendations
        n_recommendations: Number of recommendations
        
    Returns:
        DataFrame with recommendations
    """
    engine = RecommendationEngine(data_path)
    
    if user_id:
        return engine.recommend_for_user(user_id, n_recommendations)
    elif track_id:
        return engine.recommend_similar_tracks(track_id, n_recommendations)
    else:
        raise ValueError("Must provide either user_id or track_id")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python recommend.py <data_path> [user_id|track_id]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    try:
        engine = RecommendationEngine(data_path)
        stats = engine.get_user_stats()
        print(f"\nSystem Statistics:")
        print(f"  Total Users: {stats['total_users']}")
        print(f"  Total Tracks: {stats['total_tracks']}")
        print(f"  Total Interactions: {stats['total_interactions']}")
        print(f"  Avg Interactions per User: {stats['avg_interactions_per_user']:.2f}")
        
        if len(sys.argv) >= 3:
            identifier = sys.argv[2]
            
            # Try as user_id first
            if identifier in engine.user_map:
                print(f"\nRecommendations for user {identifier}:")
                recs = engine.recommend_for_user(identifier)
                print(recs.to_string(index=False))
            # Try as track_id
            elif identifier in engine.track_map:
                print(f"\nSimilar tracks to {identifier}:")
                recs = engine.recommend_similar_tracks(identifier)
                print(recs.to_string(index=False))
            else:
                print(f"Error: {identifier} not found as user_id or track_id")
        else:
            # Show example recommendations
            if len(engine.user_map) > 0:
                example_user = list(engine.user_map.keys())[0]
                print(f"\nExample: Recommendations for user {example_user}:")
                recs = engine.recommend_for_user(example_user, n_recommendations=5)
                print(recs.to_string(index=False))
                
    except FileNotFoundError:
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

