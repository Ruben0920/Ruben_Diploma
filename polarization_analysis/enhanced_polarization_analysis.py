import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import os

GENDER_MAP = {
    'SPEAKER_00': 'male',
    'SPEAKER_01': 'female',
    'SPEAKER_02': 'male'
}

class EnhancedPolarizationAnalyzer:
    """
    Advanced analyzer for political discourse polarization using multimodal sentiment analysis,
    emotion detection, and topic classification.
    """
    
    def __init__(self):
        """
        Initialize the analyzer with sentiment, emotion, and topic classification models.
        """
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        try:
            self.emotion_classifier = pipeline("text-classification", 
                                            model="j-hartmann/emotion-english-distilroberta-base")
        except:
            self.emotion_classifier = None
        self.topic_descriptions = {}
        self.zero_shot_classifier = None
        try:
            self.zero_shot_classifier = pipeline("zero-shot-classification", 
                                              model="facebook/bart-large-mnli")
        except:
            print("Warning: Zero-shot classifier not available. Will use fallback topic classification.")
    
    def detect_emotion(self, text):
        """
        Detect emotion in text using the emotion classifier.
        
        Args:
            text: Input text to analyze
            
        Returns:
            dict: Emotion label and confidence score
        """
        if self.emotion_classifier and text:
            try:
                result = self.emotion_classifier(text[:512])[0]  # Limit text length
                return {
                    'emotion': result['label'],
                    'confidence': result['score']
                }
            except:
                pass
        return {'emotion': 'neutral', 'confidence': 0.0}
    
    def classify_topics_in_batch(self, text_segments: list) -> list:
        """
        Classifies a list of text segments into predefined topics using a
        zero-shot model in a single, efficient batch.
        """
        if not self.zero_shot_classifier:
            print("Warning: Zero-shot classifier not available. Using fallback topic classification.")
            return ["Uncategorized"] * len(text_segments)

        print(f"Starting batch topic classification for {len(text_segments)} segments...")

        # Define the 15 categories from the image
        topic_labels = [
            'Economy & Finance', 'Health & Social Services', 'Education & Research',
            'Immigration & Border Policy', 'Law, Order & Justice', 'Environment & Energy',
            'Foreign Policy & Defense', 'Identity & Culture', 'Technology & Data Governance',
            'Infrastructure & Urban Policy', 'Governance & Political Reform',
            'Labor & Workforce', 'Trade & Commerce', 'Media & Information',
            'Public Opinion & Polarization Dynamics'
        ]

        # Classify all segments in one go for maximum efficiency
        results = self.zero_shot_classifier(text_segments, candidate_labels=topic_labels, multi_label=False)

        # Extract just the top-scoring label for each segment
        top_topics = [result['labels'][0] for result in results]

        print("Batch topic classification complete.")
        return top_topics
    
    def load_speaker_data(self, speakers_file):
        with open(speakers_file, 'r') as f:
            return json.load(f)
    
    def get_topic_descriptions(self):
        """
        Get descriptions for the 15 predefined topic categories.
        
        Returns:
            Dictionary with topic descriptions
        """
        topic_descriptions = {
            'Economy & Finance': {
                'name': 'Economy & Finance',
                'clean_name': 'Economy & Finance',
                'description': 'Economic policies, financial regulations, taxation, and fiscal matters',
                'keywords': ['economy', 'finance', 'tax', 'money', 'financial']
            },
            'Health & Social Services': {
                'name': 'Health & Social Services',
                'clean_name': 'Health & Social Services',
                'description': 'Healthcare policies, social welfare, and public health initiatives',
                'keywords': ['health', 'healthcare', 'social', 'welfare', 'medical']
            },
            'Education & Research': {
                'name': 'Education & Research',
                'clean_name': 'Education & Research',
                'description': 'Educational policies, research funding, and academic matters',
                'keywords': ['education', 'research', 'school', 'university', 'academic']
            },
            'Immigration & Border Policy': {
                'name': 'Immigration & Border Policy',
                'clean_name': 'Immigration & Border Policy',
                'description': 'Immigration laws, border security, and migration policies',
                'keywords': ['immigration', 'border', 'migration', 'security', 'policy']
            },
            'Law, Order & Justice': {
                'name': 'Law, Order & Justice',
                'clean_name': 'Law, Order & Justice',
                'description': 'Criminal justice, law enforcement, and legal system reforms',
                'keywords': ['law', 'justice', 'crime', 'police', 'legal']
            },
            'Environment & Energy': {
                'name': 'Environment & Energy',
                'clean_name': 'Environment & Energy',
                'description': 'Environmental protection, energy policies, and climate change',
                'keywords': ['environment', 'energy', 'climate', 'green', 'sustainability']
            },
            'Foreign Policy & Defense': {
                'name': 'Foreign Policy & Defense',
                'clean_name': 'Foreign Policy & Defense',
                'description': 'International relations, military affairs, and national security',
                'keywords': ['foreign', 'defense', 'military', 'security', 'international']
            },
            'Identity & Culture': {
                'name': 'Identity & Culture',
                'clean_name': 'Identity & Culture',
                'description': 'Cultural issues, identity politics, and social values',
                'keywords': ['identity', 'culture', 'values', 'social', 'cultural']
            },
            'Technology & Data Governance': {
                'name': 'Technology & Data Governance',
                'clean_name': 'Technology & Data Governance',
                'description': 'Technology regulation, data privacy, and digital governance',
                'keywords': ['technology', 'data', 'privacy', 'digital', 'tech']
            },
            'Infrastructure & Urban Policy': {
                'name': 'Infrastructure & Urban Policy',
                'clean_name': 'Infrastructure & Urban Policy',
                'description': 'Infrastructure development, urban planning, and public works',
                'keywords': ['infrastructure', 'urban', 'development', 'planning', 'construction']
            },
            'Governance & Political Reform': {
                'name': 'Governance & Political Reform',
                'clean_name': 'Governance & Political Reform',
                'description': 'Political system reforms, governance structures, and democratic processes',
                'keywords': ['governance', 'political', 'reform', 'democracy', 'government']
            },
            'Labor & Workforce': {
                'name': 'Labor & Workforce',
                'clean_name': 'Labor & Workforce',
                'description': 'Employment policies, labor rights, and workforce development',
                'keywords': ['labor', 'workforce', 'employment', 'workers', 'jobs']
            },
            'Trade & Commerce': {
                'name': 'Trade & Commerce',
                'clean_name': 'Trade & Commerce',
                'description': 'Trade agreements, commerce policies, and business regulations',
                'keywords': ['trade', 'commerce', 'business', 'agreements', 'commerce']
            },
            'Media & Information': {
                'name': 'Media & Information',
                'clean_name': 'Media & Information',
                'description': 'Media policies, information dissemination, and communication',
                'keywords': ['media', 'information', 'communication', 'news', 'press']
            },
            'Public Opinion & Polarization Dynamics': {
                'name': 'Public Opinion & Polarization Dynamics',
                'clean_name': 'Public Opinion & Polarization Dynamics',
                'description': 'Public sentiment, political polarization, and opinion dynamics',
                'keywords': ['opinion', 'polarization', 'sentiment', 'public', 'dynamics']
            }
        }
        return topic_descriptions
    
    def load_sentiment_data(self, sentiment_report_file):
        """
        Load sentiment data from the enhanced polarization report.
        
        Args:
            sentiment_report_file: Path to sentiment report
            
        Returns:
            DataFrame with sentiment data
        """
        with open(sentiment_report_file, 'r', encoding='utf-8') as f:
            sentiment_data = json.load(f)
        
        # Extract all segments with sentiment data
        all_segments = []
        for speaker_id, speaker_data in sentiment_data['speaker_results'].items():
            for segment in speaker_data['segments']:
                all_segments.append({
                    'unique_segment_id': segment.get('unique_segment_id', f"{speaker_id}_{segment.get('segment_id', 0)}"),
                    'text': segment['text'],
                    'speaker_id': speaker_id,
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'sentiment_compound': segment['sentiment']['compound'],
                    'sentiment_positive': segment['sentiment']['pos'],
                    'sentiment_negative': segment['sentiment']['neg'],
                    'sentiment_neutral': segment['sentiment']['neu'],
                    'emotion': segment['emotion']['emotion'],
                    'emotion_confidence': segment['emotion']['confidence'],
                    'gender': GENDER_MAP.get(speaker_id, 'unknown')
                })
        
        sentiment_df = pd.DataFrame(all_segments)
        print(f"Loaded {len(sentiment_df)} sentiment segments")
        
        return sentiment_df
    
    def add_topic_classifications(self, sentiment_df, topic_labels):
        """
        Add topic classifications to the sentiment DataFrame.
        
        Args:
            sentiment_df: DataFrame with sentiment data
            topic_labels: List of topic labels from zero-shot classification
            
        Returns:
            DataFrame with topic column added
        """
        print(f"Adding topic classifications to {len(sentiment_df)} segments...")
        
        # Add topic column
        sentiment_df['topic'] = topic_labels
        
        # Print topic distribution
        topic_distribution = sentiment_df['topic'].value_counts()
        print(f"Topic distribution: {topic_distribution.to_dict()}")
        
        return sentiment_df
    
    def analyze_segment(self, text):
        # Sentiment analysis
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        
        # Emotion analysis
        emotion = {'emotion': 'neutral', 'confidence': 0.0}
        if self.emotion_classifier:
            try:
                result = self.emotion_classifier(text[:500])[0]
                emotion = {'emotion': result['label'], 'confidence': result['score']}
            except:
                pass
        
        return {
            'sentiment': sentiment,
            'emotion': emotion
        }
    
    def analyze_speakers(self, speakers_data):
        all_segments = []
        speaker_results = {}
        
        for speaker in speakers_data:
            speaker_id = speaker['speaker_id']
            segments = speaker.get('segments', [])
            
            speaker_segments = []
            sentiments = []
            emotions = []
            
            for i, segment in enumerate(segments):
                if not segment.get('transcription'):
                    continue
                
                text = segment['transcription']
                analysis = self.analyze_segment(text)
                
                segment_result = {
                    'unique_segment_id': f"{speaker_id}_{i}",
                    'segment_id': i,
                    'speaker_id': speaker_id,
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'text': text,
                    'sentiment': analysis['sentiment'],
                    'emotion': analysis['emotion'],
                    'gender': GENDER_MAP.get(speaker_id, 'unknown')
                }
                
                speaker_segments.append(segment_result)
                sentiments.append(analysis['sentiment']['compound'])
                emotions.append(analysis['emotion']['emotion'])
                all_segments.append(segment_result)
            
            if speaker_segments:
                speaker_results[speaker_id] = {
                    'speaker_id': speaker_id,
                    'segments': speaker_segments,
                    'sentiment_summary': {
                        'mean_polarity': np.mean(sentiments),
                        'polarity_variance': np.var(sentiments) if len(sentiments) > 1 else 0,
                        'extreme_ratio': sum(1 for s in sentiments if abs(s) > 0.5) / len(sentiments)
                    },
                    'emotion_summary': {
                        'emotion_distribution': Counter(emotions),
                        'high_arousal_ratio': sum(1 for e in emotions if e in ['anger', 'disgust', 'fear']) / len(emotions)
                    }
                }
        
        return {
            'speaker_results': speaker_results,
            'all_segments': all_segments
        }
    
    def perform_comprehensive_analysis(self, sentiment_df):
        """
        Perform comprehensive speaker-first analysis using sentiment data with topic classifications.
        
        Args:
            sentiment_df: DataFrame with sentiment and topic data
            
        Returns:
            Dictionary containing both speaker-level and gender-level analysis results
        """
        if sentiment_df.empty:
            print("Warning: No sentiment data available for comprehensive analysis")
            return None
        
        print("Performing comprehensive speaker-first analysis with zero-shot topic classifications...")
        
        # Analysis A: Speaker-Level Analysis (Primary)
        speaker_sentiment = sentiment_df.groupby('speaker_id')['sentiment_compound'].agg(['mean', 'count', 'std']).round(3)
        speaker_emotion = sentiment_df.groupby('speaker_id')['emotion'].value_counts().unstack(fill_value=0)
        
        # Analysis B: Topic and Speaker Interaction
        topic_speaker_sentiment = sentiment_df.groupby(['topic', 'speaker_id'])['sentiment_compound'].agg(['mean', 'count', 'std']).round(3)
        
        # Analysis C: Gender-Level Analysis (Aggregated from Speaker Data)
        gender_sentiment = sentiment_df.groupby('gender')['sentiment_compound'].agg(['mean', 'count', 'std']).round(3)
        gender_emotion = sentiment_df.groupby('gender')['emotion'].value_counts().unstack(fill_value=0)
        topic_gender_sentiment = sentiment_df.groupby(['topic', 'gender'])['sentiment_compound'].agg(['mean', 'count', 'std']).round(3)
        
        return {
            'speaker_level_analysis': {
                'speaker_sentiment': speaker_sentiment,
                'speaker_emotion': speaker_emotion,
                'topic_speaker_sentiment': topic_speaker_sentiment
            },
            'gender_level_analysis': {
                'gender_sentiment': gender_sentiment,
                'gender_emotion': gender_emotion,
                'topic_gender_sentiment': topic_gender_sentiment
            },
            'full_dataframe': sentiment_df
        }
    
    def create_comprehensive_visualizations(self, comprehensive_analysis, topic_descriptions, output_dir):
        """
        Create comprehensive visualizations showing both speaker-level and gender-level analysis.
        
        Args:
            comprehensive_analysis: Dictionary containing comprehensive analysis results
            topic_descriptions: Dictionary mapping topic IDs to descriptions
            output_dir: Directory to save visualizations
        """
        if not comprehensive_analysis:
            return
        
        df = comprehensive_analysis['full_dataframe']
        speaker_level = comprehensive_analysis['speaker_level_analysis']
        gender_level = comprehensive_analysis['gender_level_analysis']
        
        speaker_sentiment = speaker_level['speaker_sentiment']
        speaker_emotion = speaker_level['speaker_emotion']
        topic_speaker_sentiment = speaker_level['topic_speaker_sentiment']
        
        gender_sentiment = gender_level['gender_sentiment']
        gender_emotion = gender_level['gender_emotion']
        topic_gender_sentiment = gender_level['topic_gender_sentiment']
        
        # 1. Speaker Sentiment Comparison
        plt.figure(figsize=(12, 8))
        speaker_sentiment['mean'].plot(kind='bar', color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        plt.title('Average Sentiment Score by Speaker', fontsize=14, fontweight='bold')
        plt.xlabel('Speaker ID')
        plt.ylabel('Average Sentiment Score')
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(speaker_sentiment['mean']):
            plt.text(i, v + (0.01 if v >= 0 else -0.01), f'{v:.3f}', 
                    ha='center', va='bottom' if v >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/speaker_sentiment_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Gender Sentiment Comparison (for comparison)
        plt.figure(figsize=(10, 6))
        gender_sentiment['mean'].plot(kind='bar', color=['#ff9999', '#66b3ff'])
        plt.title('Average Sentiment Score by Gender', fontsize=14, fontweight='bold')
        plt.xlabel('Gender')
        plt.ylabel('Average Sentiment Score')
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(gender_sentiment['mean']):
            plt.text(i, v + (0.01 if v >= 0 else -0.01), f'{v:.3f}', 
                    ha='center', va='bottom' if v >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gender_sentiment_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Speaker Emotion Distribution
        if not speaker_emotion.empty:
            plt.figure(figsize=(14, 8))
            speaker_emotion.plot(kind='bar', width=0.8)
            plt.title('Emotion Distribution by Speaker', fontsize=14, fontweight='bold')
            plt.xlabel('Speaker ID')
            plt.ylabel('Count')
            plt.xticks(rotation=0)
            plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/speaker_emotion_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Gender Emotion Distribution (for comparison)
        if not gender_emotion.empty:
            plt.figure(figsize=(12, 8))
            gender_emotion.plot(kind='bar', width=0.8)
            plt.title('Emotion Distribution by Gender', fontsize=14, fontweight='bold')
            plt.xlabel('Gender')
            plt.ylabel('Count')
            plt.xticks(rotation=0)
            plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/gender_emotion_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Topic-Speaker Sentiment Heatmap (Primary visualization)
        if not topic_speaker_sentiment.empty:
            # Create pivot table for heatmap
            pivot_data = topic_speaker_sentiment['mean'].unstack(fill_value=0)
            
            plt.figure(figsize=(16, 12))
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                       cbar_kws={'label': 'Average Sentiment Score'}, linewidths=0.5)
            plt.title('Average Sentiment by Topic and Speaker (Zero-Shot Classifications)', fontsize=14, fontweight='bold')
            plt.xlabel('Speaker ID')
            plt.ylabel('Topics')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/topic_speaker_sentiment_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Add visual layer for gender on the heatmap
            plt.figure(figsize=(16, 12))
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                       cbar_kws={'label': 'Average Sentiment Score'}, linewidths=0.5)
            plt.title('Average Sentiment by Topic and Speaker (with Gender Color Coding)', fontsize=14, fontweight='bold')
            plt.xlabel('Speaker ID')
            plt.ylabel('Topics')
            
            # Color code speaker labels based on gender
            ax = plt.gca()
            speaker_labels = ax.get_xticklabels()
            for i, label in enumerate(speaker_labels):
                speaker_id = label.get_text()
                gender = GENDER_MAP.get(speaker_id, 'unknown')
                color = 'red' if gender == 'female' else 'blue' if gender == 'male' else 'gray'
                label.set_color(color)
                label.set_fontweight('bold')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/topic_speaker_sentiment_heatmap_gender_coded.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. Topic-Gender Sentiment Heatmap (for comparison)
        if not topic_gender_sentiment.empty:
            # Create pivot table for heatmap
            pivot_data = topic_gender_sentiment['mean'].unstack(fill_value=0)
            
            plt.figure(figsize=(16, 10))
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                       cbar_kws={'label': 'Average Sentiment Score'}, linewidths=0.5)
            plt.title('Average Sentiment by Topic and Gender (Zero-Shot Classifications)', fontsize=14, fontweight='bold')
            plt.xlabel('Gender')
            plt.ylabel('Topics')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/gender_topic_sentiment_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def convert_nan_to_none(self, obj):
        """
        Convert NaN values to None for JSON serialization.
        
        Args:
            obj: Object that may contain NaN values
            
        Returns:
            Object with NaN values converted to None
        """
        if isinstance(obj, dict):
            return {k: self.convert_nan_to_none(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_nan_to_none(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def save_comprehensive_analysis_report(self, comprehensive_analysis, topic_descriptions, output_path):
        """
        Save comprehensive analysis results to JSON with clean topic labels and fixed NaN values.
        
        Args:
            comprehensive_analysis: Dictionary containing comprehensive analysis results
            topic_descriptions: Dictionary mapping topic IDs to descriptions
            output_path: Path to save the JSON file
        """
        if not comprehensive_analysis:
            return
        
        # Convert topic-speaker data with zero-shot topic classifications
        topic_speaker_dict = {}
        for (topic, speaker), row in comprehensive_analysis['speaker_level_analysis']['topic_speaker_sentiment'].iterrows():
            key = f"{topic}_{speaker}"
            topic_info = topic_descriptions.get(topic, {})
            
            topic_speaker_dict[key] = {
                'topic': topic,
                'topic_description': topic_info.get('description', ''),
                'topic_keywords': topic_info.get('keywords', []),
                'speaker_id': speaker,
                'gender': GENDER_MAP.get(speaker, 'unknown'),
                'mean_sentiment': float(row['mean']),
                'segment_count': int(row['count']),
                'sentiment_std': float(row['std']) if not pd.isna(row['std']) else None
            }
        
        # Convert topic-gender data with zero-shot topic classifications
        topic_gender_dict = {}
        for (topic, gender), row in comprehensive_analysis['gender_level_analysis']['topic_gender_sentiment'].iterrows():
            key = f"{topic}_{gender}"
            topic_info = topic_descriptions.get(topic, {})
            
            topic_gender_dict[key] = {
                'topic': topic,
                'topic_description': topic_info.get('description', ''),
                'topic_keywords': topic_info.get('keywords', []),
                'gender': gender,
                'mean_sentiment': float(row['mean']),
                'segment_count': int(row['count']),
                'sentiment_std': float(row['std']) if not pd.isna(row['std']) else None
            }
        
        report_data = {
            'summary': {
                'total_segments': len(comprehensive_analysis['full_dataframe']),
                'speakers_analyzed': comprehensive_analysis['speaker_level_analysis']['speaker_sentiment'].index.tolist(),
                'genders_analyzed': comprehensive_analysis['gender_level_analysis']['gender_sentiment'].index.tolist(),
                'topics_analyzed': list(comprehensive_analysis['speaker_level_analysis']['topic_speaker_sentiment'].index.get_level_values('topic').unique())
            },
            'speaker_level_analysis': {
                'speaker_sentiment_analysis': self.convert_nan_to_none(comprehensive_analysis['speaker_level_analysis']['speaker_sentiment'].to_dict()),
                'speaker_emotion_analysis': self.convert_nan_to_none(comprehensive_analysis['speaker_level_analysis']['speaker_emotion'].to_dict()) if not comprehensive_analysis['speaker_level_analysis']['speaker_emotion'].empty else {},
                'topic_speaker_sentiment_analysis': topic_speaker_dict
            },
            'gender_level_analysis': {
                'gender_sentiment_analysis': self.convert_nan_to_none(comprehensive_analysis['gender_level_analysis']['gender_sentiment'].to_dict()),
                'gender_emotion_analysis': self.convert_nan_to_none(comprehensive_analysis['gender_level_analysis']['gender_emotion'].to_dict()) if not comprehensive_analysis['gender_level_analysis']['gender_emotion'].empty else {},
                'topic_gender_sentiment_analysis': topic_gender_dict
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"Comprehensive analysis report saved to {output_path}")
    
    def print_comprehensive_analysis_summary(self, comprehensive_analysis, topic_descriptions):
        """
        Print a summary of the comprehensive analysis with clean topic labels.
        
        Args:
            comprehensive_analysis: Dictionary containing comprehensive analysis results
            topic_descriptions: Dictionary mapping topic IDs to descriptions
        """
        if not comprehensive_analysis:
            return
        
        print("\n" + "="*60)
        print("COMPREHENSIVE SPEAKER-FIRST ANALYSIS SUMMARY")
        print("="*60)
        
        # Speaker sentiment summary
        speaker_sentiment = comprehensive_analysis['speaker_level_analysis']['speaker_sentiment']
        print("\nAverage Sentiment by Speaker:")
        for speaker, row in speaker_sentiment.iterrows():
            gender = GENDER_MAP.get(speaker, 'unknown')
            print(f"  {speaker} ({gender}): {row['mean']:.3f} (n={row['count']})")
        
        # Speaker emotion summary
        speaker_emotion = comprehensive_analysis['speaker_level_analysis']['speaker_emotion']
        if not speaker_emotion.empty:
            print("\nEmotion Distribution by Speaker:")
            for speaker in speaker_emotion.index:
                emotions = speaker_emotion.loc[speaker]
                dominant_emotion = emotions.idxmax()
                gender = GENDER_MAP.get(speaker, 'unknown')
                print(f"  {speaker} ({gender}): Most common emotion = {dominant_emotion} ({emotions[dominant_emotion]} segments)")
        
        # Topic-speaker summary with zero-shot topic classifications
        topic_speaker = comprehensive_analysis['speaker_level_analysis']['topic_speaker_sentiment']
        if not topic_speaker.empty:
            print("\nTopic-Speaker Sentiment Highlights:")
            for (topic, speaker), row in topic_speaker.iterrows():
                if row['count'] > 0:  # Only show topics with data
                    gender = GENDER_MAP.get(speaker, 'unknown')
                    print(f"  {speaker} ({gender}) on {topic}: {row['mean']:.3f} (n={row['count']})")
        
        # Gender-level summary (for comparison)
        print("\n" + "-"*40)
        print("GENDER-LEVEL SUMMARY (FOR COMPARISON)")
        print("-"*40)
        
        gender_sentiment = comprehensive_analysis['gender_level_analysis']['gender_sentiment']
        print("\nAverage Sentiment by Gender:")
        for gender, row in gender_sentiment.iterrows():
            print(f"  {gender.capitalize()}: {row['mean']:.3f} (n={row['count']})")
    
    def create_visualizations(self, results, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        # Overall sentiment timeline
        self._create_sentiment_timeline(results, output_dir)
        
        # Create speaker-specific visualizations
        for speaker_id, speaker_data in results['speaker_results'].items():
            speaker_dir = f"{output_dir}/speaker_{speaker_id}"
            os.makedirs(speaker_dir, exist_ok=True)
            
            self._create_speaker_sentiment_timeline(speaker_data, speaker_dir)
            self._create_speaker_emotion_distribution(speaker_data, speaker_dir)
            self._create_speaker_analysis_dashboard(speaker_data, speaker_dir)
    
    def _create_sentiment_timeline(self, results, output_dir):
        all_segments = results['all_segments']
        if not all_segments:
            return
        
        # Sort segments by start time
        sorted_segments = sorted(all_segments, key=lambda x: x['start_time'])
        
        times = [seg['start_time'] for seg in sorted_segments]
        sentiments = [seg['sentiment']['compound'] for seg in sorted_segments]
        speakers = [seg['speaker_id'] for seg in sorted_segments]
        
        plt.figure(figsize=(15, 8))
        
        # Color code by speaker
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        unique_speakers = list(set(speakers))
        
        for i, speaker in enumerate(unique_speakers):
            speaker_times = [t for j, t in enumerate(times) if speakers[j] == speaker]
            speaker_sentiments = [s for j, s in enumerate(sentiments) if speakers[j] == speaker]
            plt.plot(speaker_times, speaker_sentiments, 'o-', 
                    label=speaker, color=colors[i % len(colors)], linewidth=2, markersize=6)
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.fill_between(times, sentiments, 0, 
                        where=[s > 0 for s in sentiments], alpha=0.3, color='green')
        plt.fill_between(times, sentiments, 0, 
                        where=[s < 0 for s in sentiments], alpha=0.3, color='red')
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Sentiment Polarity')
        plt.title('Overall Sentiment Timeline')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-1, 1)
        plt.savefig(f"{output_dir}/overall_sentiment_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_speaker_sentiment_timeline(self, speaker_data, output_dir):
        segments = speaker_data['segments']
        if not segments:
            return
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x['start_time'])
        
        times = [seg['start_time'] for seg in sorted_segments]
        sentiments = [seg['sentiment']['compound'] for seg in sorted_segments]
        
        plt.figure(figsize=(12, 6))
        plt.plot(times, sentiments, 'o-', linewidth=2, markersize=6, color='blue')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.fill_between(times, sentiments, 0, 
                        where=[s > 0 for s in sentiments], alpha=0.3, color='green')
        plt.fill_between(times, sentiments, 0, 
                        where=[s < 0 for s in sentiments], alpha=0.3, color='red')
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Sentiment Polarity')
        plt.title(f'Sentiment Timeline - {speaker_data["speaker_id"]}')
        plt.grid(True, alpha=0.3)
        plt.ylim(-1, 1)
        plt.savefig(f"{output_dir}/sentiment_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_speaker_emotion_distribution(self, speaker_data, output_dir):
        segments = speaker_data['segments']
        if not segments:
            return
        
        emotions = [seg['emotion']['emotion'] for seg in segments]
        emotion_counts = Counter(emotions)
        
        plt.figure(figsize=(10, 8))
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']
        plt.pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%', 
                colors=colors[:len(emotion_counts)], startangle=90)
        plt.title(f'Emotion Distribution - {speaker_data["speaker_id"]}')
        plt.axis('equal')
        plt.savefig(f"{output_dir}/emotion_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_speaker_analysis_dashboard(self, speaker_data, output_dir):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Sentiment metrics
        sentiment = speaker_data['sentiment_summary']
        metrics = ['Mean Polarity', 'Polarity Variance', 'Extreme Ratio']
        values = [sentiment['mean_polarity'], sentiment['polarity_variance'], sentiment['extreme_ratio']]
        
        bars = ax1.bar(metrics, values, color=['blue', 'green', 'orange'])
        ax1.set_title('Sentiment Metrics')
        ax1.set_ylabel('Value')
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Sentiment distribution
        sentiments = [seg['sentiment']['compound'] for seg in speaker_data['segments']]
        ax2.hist(sentiments, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax2.axvline(x=np.mean(sentiments), color='green', linestyle='-', alpha=0.7)
        ax2.set_title('Sentiment Distribution')
        ax2.set_xlabel('Sentiment Polarity')
        ax2.set_ylabel('Frequency')
        
        # Emotion distribution
        emotion_dist = speaker_data['emotion_summary']['emotion_distribution']
        if emotion_dist:
            ax3.pie(emotion_dist.values(), labels=emotion_dist.keys(), autopct='%1.1f%%')
            ax3.set_title('Emotion Distribution')
        
        # High arousal ratio
        arousal_ratio = speaker_data['emotion_summary']['high_arousal_ratio']
        ax4.bar(['High Arousal'], [arousal_ratio], color='red', alpha=0.7)
        ax4.set_title('High Arousal Emotion Ratio')
        ax4.set_ylabel('Ratio')
        ax4.set_ylim(0, 1)
        ax4.text(0, arousal_ratio + 0.01, f'{arousal_ratio:.3f}', ha='center', va='bottom')
        
        plt.suptitle(f'Analysis Dashboard - {speaker_data["speaker_id"]}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/analysis_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    analyzer = EnhancedPolarizationAnalyzer()
    
    # Get video filename and output directory from environment variables
    video_filename = os.getenv('VIDEO_FILENAME', 'test_video_1.mp4')
    output_base_dir = os.getenv('OUTPUT_BASE_DIR', '/app/output')
    
    # Extract video name without extension for directory structure
    video_name = os.path.splitext(video_filename)[0]
    
    # Set paths based on video-specific directory structure
    speaker_dir = f"/app/output/{video_name}/speaker_diarization"
    polarization_dir = f"/app/output/{video_name}/polarization_analysis"
    
    # Create polarization analysis output directory
    os.makedirs(polarization_dir, exist_ok=True)
    os.makedirs(f"{polarization_dir}/visualizations", exist_ok=True)
    
    # First, we need to create the enhanced_polarization_report.json from speakers data
    speakers_file = f"{speaker_dir}/speakers_with_embeddings.json"
    
    # Create sentiment report if it doesn't exist
    sentiment_report_file = f"{polarization_dir}/enhanced_polarization_report.json"
    
    if os.path.exists(speakers_file):
        # Process speakers data to create sentiment report
        with open(speakers_file, 'r') as f:
            speakers_data = json.load(f)
        
        # Create sentiment segments from speakers data
        sentiment_segments = []
        for speaker in speakers_data:
            speaker_id = speaker['speaker_id']
            for segment in speaker['segments']:
                if segment.get('transcription'):
                    sentiment_scores = analyzer.sentiment_analyzer.polarity_scores(segment['transcription'])
                    emotion = analyzer.detect_emotion(segment['transcription'])
                    
                    sentiment_segments.append({
                        'speaker_id': speaker_id,
                        'text': segment['transcription'],
                        'start_time': segment['start_time'],
                        'end_time': segment['end_time'],
                        'sentiment': sentiment_scores,
                        'emotion': emotion
                    })
        
        # Save the sentiment report
        with open(sentiment_report_file, 'w') as f:
            json.dump({'segments': sentiment_segments}, f, indent=2)
    
    try:
        print("Loading data for comprehensive speaker-first analysis...")
        
        # Load sentiment data
        sentiment_df = analyzer.load_sentiment_data(sentiment_report_file)
        
        # Extract all text segments for batch classification
        all_texts = sentiment_df['text'].tolist()
        print(f"Extracted {len(all_texts)} text segments for topic classification")
        
        # Perform zero-shot topic classification
        topic_labels = analyzer.classify_topics_in_batch(all_texts)
        
        # Add topic classifications to sentiment DataFrame
        sentiment_df = analyzer.add_topic_classifications(sentiment_df, topic_labels)
        
        # Get topic descriptions for the 15 predefined categories
        topic_descriptions = analyzer.get_topic_descriptions()
        
        # Perform comprehensive speaker-first analysis with zero-shot topic classifications
        print("Performing comprehensive speaker-first analysis with zero-shot topic classifications...")
        comprehensive_analysis = analyzer.perform_comprehensive_analysis(sentiment_df)
        
        # Create comprehensive visualizations with zero-shot topic classifications
        print("Creating comprehensive visualizations with zero-shot topic classifications...")
        analyzer.create_comprehensive_visualizations(comprehensive_analysis, topic_descriptions, f'{polarization_dir}/visualizations')
        
        # Save results with zero-shot topic classifications
        analyzer.save_comprehensive_analysis_report(comprehensive_analysis, topic_descriptions, f'{polarization_dir}/gender_topic_sentiment_report.json')
        analyzer.print_comprehensive_analysis_summary(comprehensive_analysis, topic_descriptions)
        
        print("Comprehensive speaker-first analysis complete!")
        print("- gender_topic_sentiment_report.json (with speaker-level and gender-level analysis)")
        print("- enhanced_visualizations/ (with comprehensive speaker-first visualizations)")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 