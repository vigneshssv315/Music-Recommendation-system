"""
Streamlit web application for Music Recommendation System.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import sys
import os
from recommend import RecommendationEngine
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1DB954;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1DB954;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 2rem;
    }
    .stButton>button:hover {
        background-color: #1ed760;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False


def load_engine(data_path: str):
    """Load the recommendation engine."""
    try:
        with st.spinner("Loading data and training model... This may take a moment."):
            engine = RecommendationEngine(data_path, n_neighbors=10)
            st.session_state.engine = engine
            st.session_state.data_loaded = True
            return True
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return False


def main():
    """Main application function."""
    st.markdown('<p class="main-header">🎵 Music Recommendation System</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for data loading
    with st.sidebar:
        st.header("📁 Data Configuration")
        
        data_source = st.radio(
            "Data Source",
            ["Upload CSV", "Use File Path"],
            help="Choose to upload a file or provide a path to existing data"
        )
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload your music data CSV",
                type=['csv'],
                help="CSV should have columns: user_id, track_id, rating/play_count"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                temp_path = "temp_data.csv"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                if st.button("Load Data", type="primary"):
                    if load_engine(temp_path):
                        st.success("Data loaded successfully!")
        else:
            data_path = st.text_input(
                "Enter data file path",
                value="data/music_data.csv",
                help="Path to your CSV file"
            )
            
            if st.button("Load Data", type="primary"):
                if os.path.exists(data_path):
                    if load_engine(data_path):
                        st.success("Data loaded successfully!")
                else:
                    st.error(f"File not found: {data_path}")
        
        st.markdown("---")
        
        if st.session_state.data_loaded:
            stats = st.session_state.engine.get_user_stats()
            st.header("📊 Statistics")
            st.metric("Total Users", stats['total_users'])
            st.metric("Total Tracks", stats['total_tracks'])
            st.metric("Total Interactions", stats['total_interactions'])
            st.metric("Avg Interactions/User", f"{stats['avg_interactions_per_user']:.1f}")
    
    # Main content area
    if not st.session_state.data_loaded:
        st.info("👈 Please load your data file using the sidebar to get started.")
        st.markdown("""
        ### 📋 Expected CSV Format
        
        Your CSV file should contain the following columns:
        - **user_id**: Unique identifier for each user
        - **track_id**: Unique identifier for each track/song
        - **rating** or **play_count**: User's interaction with the track (rating 1-5 or play count)
        - **song_name** (optional): Name of the song
        - **artist_name** (optional): Name of the artist
        
        ### Example Data
        
        ```csv
        user_id,track_id,rating,song_name,artist_name
        user1,track1,5,Bohemian Rhapsody,Queen
        user1,track2,4,Stairway to Heaven,Led Zeppelin
        user2,track1,5,Bohemian Rhapsody,Queen
        ```
        """)
    else:
        st.markdown('<p class="sub-header">Get Recommendations</p>', unsafe_allow_html=True)
        
        # Recommendation type selection
        rec_type = st.radio(
            "Recommendation Type",
            ["Recommend for User", "Find Similar Tracks"],
            horizontal=True
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if rec_type == "Recommend for User":
                user_input = st.text_input(
                    "Enter User ID",
                    placeholder="e.g., user1",
                    help="Enter a user ID to get personalized recommendations"
                )
                
                if st.button("Get Recommendations", type="primary"):
                    if user_input:
                        if user_input in st.session_state.engine.user_map:
                            with st.spinner("Generating recommendations..."):
                                recommendations = st.session_state.engine.recommend_for_user(
                                    user_input, 
                                    n_recommendations=10
                                )
                            
                            if not recommendations.empty:
                                st.success(f"Found {len(recommendations)} recommendations!")
                                
                                # Display recommendations
                                st.dataframe(
                                    recommendations,
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                # Download button
                                csv = recommendations.to_csv(index=False)
                                st.download_button(
                                    label="Download Recommendations as CSV",
                                    data=csv,
                                    file_name=f"recommendations_user_{user_input}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.warning("No recommendations found for this user.")
                        else:
                            st.error(f"User '{user_input}' not found in the dataset.")
                    else:
                        st.warning("Please enter a user ID.")
            
            else:  # Find Similar Tracks
                track_input = st.text_input(
                    "Enter Track ID",
                    placeholder="e.g., track1",
                    help="Enter a track ID to find similar tracks"
                )
                
                if st.button("Find Similar Tracks", type="primary"):
                    if track_input:
                        if track_input in st.session_state.engine.track_map:
                            # Show track info if available
                            track_info = st.session_state.engine.get_track_info(track_input)
                            if track_info:
                                info_text = f"**Track:** {track_info.get('track_id', track_input)}"
                                if 'song_name' in track_info:
                                    info_text += f" | **Song:** {track_info['song_name']}"
                                if 'artist_name' in track_info:
                                    info_text += f" | **Artist:** {track_info['artist_name']}"
                                st.info(info_text)
                            
                            with st.spinner("Finding similar tracks..."):
                                recommendations = st.session_state.engine.recommend_similar_tracks(
                                    track_input, 
                                    n_recommendations=10
                                )
                            
                            if not recommendations.empty:
                                st.success(f"Found {len(recommendations)} similar tracks!")
                                
                                # Display recommendations
                                st.dataframe(
                                    recommendations,
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                # Download button
                                csv = recommendations.to_csv(index=False)
                                st.download_button(
                                    label="Download Similar Tracks as CSV",
                                    data=csv,
                                    file_name=f"similar_tracks_{track_input}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.warning("No similar tracks found.")
                        else:
                            st.error(f"Track '{track_input}' not found in the dataset.")
                    else:
                        st.warning("Please enter a track ID.")
        
        with col2:
            st.markdown("### 💡 Tips")
            st.markdown("""
            - **User Recommendations**: Enter a user ID to get personalized song recommendations based on similar users' preferences.
            
            - **Similar Tracks**: Enter a track ID to find songs that are similar based on user listening patterns.
            
            - **Explore**: Try different user IDs or track IDs to see various recommendations!
            """)
            
            if st.session_state.data_loaded:
                st.markdown("### 🔍 Quick Lookup")
                
                # Show sample users
                sample_users = list(st.session_state.engine.user_map.keys())[:5]
                st.markdown("**Sample User IDs:**")
                for user in sample_users:
                    if st.button(f"Use {user}", key=f"user_{user}"):
                        st.session_state.user_input = user
                        st.rerun()
                
                # Show sample tracks
                sample_tracks = list(st.session_state.engine.track_map.keys())[:5]
                st.markdown("**Sample Track IDs:**")
                for track in sample_tracks:
                    if st.button(f"Use {track}", key=f"track_{track}"):
                        st.session_state.track_input = track
                        st.rerun()


if __name__ == "__main__":
    main()

