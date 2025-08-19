import json
import pandas as pd
from typing import Dict, List, Any, Optional
import os

class DataLoader:
    """
    Loads and parses all analysis data from the output directory
    """
    
    def __init__(self, output_dir: str = "../output"):
        self.output_dir = output_dir
        self.data = {}
        self.load_all_data()
    
    def load_all_data(self):
        """Load all JSON files and parse them into structured data"""
        try:
            self.data['gender_topic_sentiment'] = self._load_json('gender_topic_sentiment_report.json')
            self.data['enhanced_polarization'] = self._load_json('enhanced_polarization_report.json')
            self.data['speakers_with_embeddings'] = self._load_json('speakers_with_embeddings.json')
            self.data['transcription_timestamps'] = self._load_json('transcription_with_timestamps.json')
            
            self._parse_speaker_data()
            self._parse_topic_data()
            self._parse_segment_data()
            
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def _load_json(self, filename: str) -> Dict[str, Any]:
        """Load a JSON file from the output directory"""
        filepath = os.path.join(self.output_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return {}
    
    def _parse_speaker_data(self):
        """Parse speaker-level analysis data"""
        if 'gender_topic_sentiment' in self.data:
            analysis = self.data['gender_topic_sentiment']
            
            if 'speaker_level_analysis' in analysis:
                speaker_analysis = analysis['speaker_level_analysis']
                
                if 'speaker_sentiment_analysis' in speaker_analysis:
                    sentiment_data = speaker_analysis['speaker_sentiment_analysis']
                    self.data['speaker_sentiment_df'] = pd.DataFrame({
                        'speaker_id': list(sentiment_data['mean'].keys()),
                        'mean_sentiment': list(sentiment_data['mean'].values()),
                        'segment_count': list(sentiment_data['count'].values()),
                        'sentiment_std': list(sentiment_data['std'].values())
                    })
                
                if 'speaker_emotion_analysis' in speaker_analysis:
                    emotion_data = speaker_analysis['speaker_emotion_analysis']
                    emotion_df_data = []
                    for emotion, speaker_counts in emotion_data.items():
                        for speaker_id, count in speaker_counts.items():
                            emotion_df_data.append({
                                'speaker_id': speaker_id,
                                'emotion': emotion,
                                'count': count
                            })
                    self.data['speaker_emotion_df'] = pd.DataFrame(emotion_df_data)
    
    def _parse_topic_data(self):
        """Parse topic-level analysis data"""
        if 'gender_topic_sentiment' in self.data:
            analysis = self.data['gender_topic_sentiment']
            
            if 'speaker_level_analysis' in analysis:
                speaker_analysis = analysis['speaker_level_analysis']
                
                if 'topic_speaker_sentiment_analysis' in speaker_analysis:
                    topic_data = speaker_analysis['topic_speaker_sentiment_analysis']
                    topic_df_data = []
                    
                    for key, data in topic_data.items():
                        topic_df_data.append({
                            'topic': data.get('topic', ''),
                            'speaker_id': data.get('speaker_id', ''),
                            'gender': data.get('gender', ''),
                            'mean_sentiment': data.get('mean_sentiment', 0),
                            'segment_count': data.get('segment_count', 0),
                            'sentiment_std': data.get('sentiment_std', 0)
                        })
                    
                    self.data['topic_speaker_sentiment_df'] = pd.DataFrame(topic_df_data)
    
    def _parse_segment_data(self):
        """Parse segment-level analysis data for timeline and detailed analysis"""
        if 'enhanced_polarization' in self.data:
            polarization_data = self.data['enhanced_polarization']
            
            if 'speaker_results' in polarization_data:
                all_segments = []
                
                for speaker_id, speaker_data in polarization_data['speaker_results'].items():
                    if 'segments' in speaker_data:
                        for segment in speaker_data['segments']:
                            segment_data = {
                                'speaker_id': speaker_id,
                                'segment_id': segment.get('segment_id', 0),
                                'start_time': segment.get('start_time', 0),
                                'end_time': segment.get('end_time', 0),
                                'text': segment.get('text', ''),
                                'sentiment_compound': segment.get('sentiment', {}).get('compound', 0),
                                'sentiment_positive': segment.get('sentiment', {}).get('pos', 0),
                                'sentiment_negative': segment.get('sentiment', {}).get('neg', 0),
                                'sentiment_neutral': segment.get('sentiment', {}).get('neu', 0),
                                'emotion': segment.get('emotion', {}).get('emotion', ''),
                                'emotion_confidence': segment.get('emotion', {}).get('confidence', 0),
                                'topic': segment.get('topic', ''),
                                'gender': segment.get('gender', '')
                            }
                            all_segments.append(segment_data)
                
                self.data['segments_df'] = pd.DataFrame(all_segments)
                self.data['segments_df'] = self.data['segments_df'].sort_values('start_time')
    
    def get_speakers(self) -> List[str]:
        """Get list of all speaker IDs"""
        if 'speaker_sentiment_df' in self.data:
            return self.data['speaker_sentiment_df']['speaker_id'].tolist()
        return []
    
    def get_topics(self) -> List[str]:
        """Get list of all unique topics"""
        if 'topic_speaker_sentiment_df' in self.data:
            return sorted(self.data['topic_speaker_sentiment_df']['topic'].unique().tolist())
        return []
    
    def get_speaker_segments(self, speaker_id: str) -> pd.DataFrame:
        """Get all segments for a specific speaker"""
        if 'segments_df' in self.data:
            return self.data['segments_df'][
                self.data['segments_df']['speaker_id'] == speaker_id
            ].copy()
        return pd.DataFrame()
    
    def get_topic_sentiment_comparison(self, topic: str) -> pd.DataFrame:
        """Get sentiment comparison for all speakers on a specific topic"""
        if 'topic_speaker_sentiment_df' in self.data:
            return self.data['topic_speaker_sentiment_df'][
                self.data['topic_speaker_sentiment_df']['topic'] == topic
            ].copy()
        return pd.DataFrame()
    
    def get_most_polarizing_statements(self, topic: str, top_n: int = 3) -> Dict[str, List[Dict]]:
        """Get the most positive and negative statements for a topic"""
        if 'segments_df' not in self.data:
            return {'positive': [], 'negative': []}
        
        topic_segments = self.data['segments_df'][
            self.data['segments_df']['topic'] == topic
        ].copy()
        
        if topic_segments.empty:
            return {'positive': [], 'negative': []}
        
        positive_statements = topic_segments.nlargest(top_n, 'sentiment_compound')[
            ['speaker_id', 'text', 'sentiment_compound', 'emotion']
        ].to_dict('records')
        
        negative_statements = topic_segments.nsmallest(top_n, 'sentiment_compound')[
            ['speaker_id', 'text', 'sentiment_compound', 'emotion']
        ].to_dict('records')
        
        return {
            'positive': positive_statements,
            'negative': negative_statements
        }
    
    def get_speaker_emotion_distribution(self, speaker_id: str) -> pd.DataFrame:
        """Get emotion distribution for a specific speaker"""
        if 'speaker_emotion_df' in self.data:
            speaker_emotions = self.data['speaker_emotion_df'][
                self.data['speaker_emotion_df']['speaker_id'] == speaker_id
            ].copy()
            
            if not speaker_emotions.empty:
                total_segments = speaker_emotions['count'].sum()
                speaker_emotions['percentage'] = (speaker_emotions['count'] / total_segments) * 100
            
            return speaker_emotions
        return pd.DataFrame()
    
    def get_speaker_topic_focus(self, speaker_id: str) -> pd.DataFrame:
        """Get topic focus distribution for a specific speaker"""
        if 'segments_df' in self.data:
            speaker_segments = self.data['segments_df'][
                self.data['segments_df']['speaker_id'] == speaker_id
            ].copy()
            
            if not speaker_segments.empty:
                topic_counts = speaker_segments['topic'].value_counts().reset_index()
                topic_counts.columns = ['topic', 'segment_count']
                topic_counts['percentage'] = (topic_counts['segment_count'] / len(speaker_segments)) * 100
                return topic_counts
            
        return pd.DataFrame()
    
    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get overall debate statistics"""
        if 'segments_df' not in self.data:
            return {}
        
        segments_df = self.data['segments_df']
        
        speaking_time = segments_df.groupby('speaker_id').agg({
            'start_time': 'min',
            'end_time': 'max'
        }).reset_index()
        speaking_time['total_time'] = speaking_time['end_time'] - speaking_time['start_time']
        
        avg_sentiment = segments_df.groupby('speaker_id')['sentiment_compound'].mean().reset_index()
        
        emotion_counts = segments_df.groupby(['speaker_id', 'emotion']).size().reset_index(name='count')
        most_frequent_emotion = emotion_counts.loc[
            emotion_counts.groupby('speaker_id')['count'].idxmax()
        ][['speaker_id', 'emotion']]
        
        return {
            'speaking_time': speaking_time.to_dict('records'),
            'avg_sentiment': avg_sentiment.to_dict('records'),
            'most_frequent_emotion': most_frequent_emotion.to_dict('records')
        }
