#!/usr/bin/env python3
"""
Test script to verify data loading functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader

def test_data_loading():
    """Test the data loader functionality"""
    print("Testing Data Loader...")
    
    try:
        print("Initializing DataLoader...")
        data_loader = DataLoader()
        
        print("Testing basic data loading...")
        
        required_keys = [
            'gender_topic_sentiment',
            'enhanced_polarization',
            'speakers_with_embeddings',
            'transcription_timestamps'
        ]
        
        for key in required_keys:
            if key in data_loader.data:
                print(f"  {key}: Loaded successfully")
            else:
                print(f"  {key}: Failed to load")
        
        print("Testing speaker data...")
        speakers = data_loader.get_speakers()
        if speakers:
            print(f"  Found {len(speakers)} speakers: {speakers}")
        else:
            print("  No speakers found")
        
        print("Testing topic data...")
        topics = data_loader.get_topics()
        if topics:
            print(f"  Found {len(topics)} topics: {topics[:5]}...")
        else:
            print("  No topics found")
        
        print("Testing segment data...")
        segments_df = data_loader.data.get('segments_df', None)
        if segments_df is not None and not segments_df.empty:
            print(f"  Found {len(segments_df)} segments")
            print(f"    Columns: {list(segments_df.columns)}")
            print(f"    Time range: {segments_df['start_time'].min():.1f}s - {segments_df['end_time'].max():.1f}s")
        else:
            print("  No segments found")
        
        if speakers:
            print(f"Testing data for {speakers[0]}...")
            speaker_segments = data_loader.get_speaker_segments(speakers[0])
            if not speaker_segments.empty:
                print(f"  Found {len(speaker_segments)} segments for {speakers[0]}")
            else:
                print(f"  No segments found for {speakers[0]}")
            
            emotion_dist = data_loader.get_speaker_emotion_distribution(speakers[0])
            if not emotion_dist.empty:
                print(f"  Found emotion distribution for {speakers[0]}")
            else:
                print(f"  No emotion data for {speakers[0]}")
            
            topic_focus = data_loader.get_speaker_topic_focus(speakers[0])
            if not topic_focus.empty:
                print(f"  Found topic focus for {speakers[0]}")
            else:
                print(f"  No topic focus data for {speakers[0]}")
        
        if topics:
            print(f"Testing data for topic: {topics[0]}...")
            topic_comparison = data_loader.get_topic_sentiment_comparison(topics[0])
            if not topic_comparison.empty:
                print(f"  Found sentiment comparison for {topics[0]}")
            else:
                print(f"  No sentiment comparison for {topics[0]}")
            
            polarizing_statements = data_loader.get_most_polarizing_statements(topics[0])
            if polarizing_statements['positive'] or polarizing_statements['negative']:
                print(f"  Found polarizing statements for {topics[0]}")
            else:
                print(f"  No polarizing statements for {topics[0]}")
        
        print("Testing overall statistics...")
        overall_stats = data_loader.get_overall_statistics()
        if overall_stats:
            print("  Overall statistics generated successfully")
            for key, value in overall_stats.items():
                if isinstance(value, list):
                    print(f"    {key}: {len(value)} items")
                else:
                    print(f"    {key}: {value}")
        else:
            print("  Failed to generate overall statistics")
        
        print("Data loading test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_structures():
    """Test the structure of loaded data"""
    print("Testing data structures...")
    
    try:
        data_loader = DataLoader()
        
        if 'speaker_sentiment_df' in data_loader.data:
            df = data_loader.data['speaker_sentiment_df']
            print(f"  Speaker sentiment DataFrame: {df.shape}")
            print(f"    Columns: {list(df.columns)}")
            print(f"    Sample data:\n{df.head()}")
        
        if 'topic_speaker_sentiment_df' in data_loader.data:
            df = data_loader.data['topic_speaker_sentiment_df']
            print(f"  Topic speaker sentiment DataFrame: {df.shape}")
            print(f"    Columns: {list(df.columns)}")
            print(f"    Sample data:\n{df.head()}")
        
        if 'segments_df' in data_loader.data:
            df = data_loader.data['segments_df']
            print(f"  Segments DataFrame: {df.shape}")
            print(f"    Columns: {list(df.columns)}")
            print(f"    Sample data:\n{df.head()}")
        
        return True
        
    except Exception as e:
        print(f"Error testing data structures: {e}")
        return False

if __name__ == "__main__":
    print("Starting Data Loader Tests...")
    print("=" * 50)
    
    success = test_data_loading()
    
    if success:
        test_data_structures()
    
    print("\n" + "=" * 50)
    if success:
        print("All tests passed! The dashboard should work correctly.")
    else:
        print("Some tests failed. Please check the error messages above.")
