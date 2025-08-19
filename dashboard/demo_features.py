#!/usr/bin/env python3
"""
Demo script to showcase dashboard capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
import pandas as pd

def demo_dashboard_features():
    """Demonstrate the dashboard's analysis capabilities"""
    print("Interactive Multi-Modal Analysis Dashboard - Feature Demo")
    print("=" * 70)
    
    try:
        print("Loading analysis data...")
        data_loader = DataLoader()
        
        print("Data loaded successfully!")
        print()
        
        demo_speaker_analysis(data_loader)
        demo_topic_analysis(data_loader)
        demo_overall_statistics(data_loader)
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

def demo_speaker_analysis(data_loader):
    """Demonstrate speaker analysis features"""
    print("SPEAKER ANALYSIS FEATURES")
    print("-" * 40)
    
    speakers = data_loader.get_speakers()
    if not speakers:
        print("No speakers found in the data.")
        return
    
    print(f"Found {len(speakers)} speakers: {', '.join(speakers)}")
    print()
    
    for speaker in speakers[:2]:
        print(f"Analysis for {speaker}:")
        
        segments_df = data_loader.get_speaker_segments(speaker)
        if not segments_df.empty:
            print(f"  Total segments: {len(segments_df)}")
            print(f"  Speaking time: {segments_df['end_time'].max() - segments_df['start_time'].min():.1f} seconds")
            print(f"  Average sentiment: {segments_df['sentiment_compound'].mean():.3f}")
            print(f"  Most frequent emotion: {segments_df['emotion'].mode().iloc[0] if not segments_df['emotion'].mode().empty else 'Unknown'}")
        
        emotion_dist = data_loader.get_speaker_emotion_distribution(speaker)
        if not emotion_dist.empty:
            print(f"  Emotion distribution:")
            for _, row in emotion_dist.head(3).iterrows():
                print(f"    {row['emotion']}: {row['count']} ({row['percentage']:.1f}%)")
        
        topic_focus = data_loader.get_speaker_topic_focus(speaker)
        if not topic_focus.empty:
            print(f"  Top topics:")
            for _, row in topic_focus.head(3).iterrows():
                print(f"    {row['topic']}: {row['segment_count']} segments")
        
        print()

def demo_topic_analysis(data_loader):
    """Demonstrate topic analysis features"""
    print("TOPIC ANALYSIS FEATURES")
    print("-" * 40)
    
    topics = data_loader.get_topics()
    if not topics:
        print("No topics found in the data.")
        return
    
    print(f"Found {len(topics)} topics: {', '.join(topics[:5])}...")
    print()
    
    for topic in topics[:2]:
        print(f"Analysis for topic: {topic}")
        
        topic_comparison = data_loader.get_topic_sentiment_comparison(topic)
        if not topic_comparison.empty:
            print(f"  Speaker sentiment comparison:")
            for _, row in topic_comparison.iterrows():
                sentiment_label = "Positive" if row['mean_sentiment'] > 0.1 else "Negative" if row['mean_sentiment'] < -0.1 else "Neutral"
                print(f"    {row['speaker_id']}: {row['mean_sentiment']:.3f} ({sentiment_label})")
        
        polarizing_statements = data_loader.get_most_polarizing_statements(topic, top_n=2)
        if polarizing_statements['positive']:
            print(f"  Most positive statement:")
            statement = polarizing_statements['positive'][0]
            print(f"    {statement['speaker_id']}: \"{statement['text'][:80]}...\" (Sentiment: {statement['sentiment_compound']:.3f})")
        
        if polarizing_statements['negative']:
            print(f"  Most negative statement:")
            statement = polarizing_statements['negative'][0]
            print(f"    {statement['speaker_id']}: \"{statement['text'][:80]}...\" (Sentiment: {statement['sentiment_compound']:.3f})")
        
        print()

def demo_overall_statistics(data_loader):
    """Demonstrate overall analysis features"""
    print("OVERALL ANALYSIS FEATURES")
    print("-" * 40)
    
    overall_stats = data_loader.get_overall_statistics()
    
    if overall_stats:
        print("Overall Debate Statistics:")
        
        if 'speaking_time' in overall_stats:
            print("  Speaking time per speaker:")
            for item in overall_stats['speaking_time']:
                print(f"    {item['speaker_id']}: {item['total_time']:.1f} seconds")
        
        if 'avg_sentiment' in overall_stats:
            print("  Average sentiment per speaker:")
            for item in overall_stats['avg_sentiment']:
                sentiment_label = "Positive" if item['avg_sentiment'] > 0.1 else "Negative" if item['avg_sentiment'] < -0.1 else "Neutral"
                print(f"    {item['speaker_id']}: {item['avg_sentiment']:.3f} ({sentiment_label})")
        
        if 'most_frequent_emotion' in overall_stats:
            print("  Most frequent emotion per speaker:")
            for item in overall_stats['most_frequent_emotion']:
                print(f"    {item['speaker_id']}: {item['emotion']}")
    
    segments_df = data_loader.data.get('segments_df', pd.DataFrame())
    if not segments_df.empty:
        print()
        print("Summary Statistics:")
        print(f"  Total segments analyzed: {len(segments_df)}")
        print(f"  Total speakers: {segments_df['speaker_id'].nunique()}")
        print(f"  Total topics: {segments_df['topic'].nunique()}")
        print(f"  Total duration: {segments_df['end_time'].max() - segments_df['start_time'].min():.1f} seconds")
        
        emotion_counts = segments_df['emotion'].value_counts()
        print(f"  Most common emotion: {emotion_counts.index[0] if not emotion_counts.empty else 'Unknown'}")
        
        avg_sentiment = segments_df['sentiment_compound'].mean()
        sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
        print(f"  Overall average sentiment: {avg_sentiment:.3f} ({sentiment_label})")

if __name__ == "__main__":
    demo_dashboard_features()
