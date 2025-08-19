import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

# Gender mapping for the specific dataset
GENDER_MAP = {
    'SPEAKER_00': 'male',   # Donald Trump
    'SPEAKER_01': 'female', # Hillary Clinton
    'SPEAKER_02': 'male'    # Lester Holt
}

class NarrativeReportGenerator:
    def __init__(self):
        self.analysis_data = None
        self.precomputed_data = None
    
    def load_analysis_data(self, json_file_path: str) -> Dict[str, Any]:
        """
        Load the comprehensive analysis JSON data.
        
        Args:
            json_file_path: Path to the JSON file
            
        Returns:
            Dictionary containing the analysis data
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded analysis data with {data['summary']['total_segments']} segments")
        return data
    
    def precompute_key_actors_and_metrics(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pre-compute key actors and metrics from the speaker-level analysis data.
        
        Args:
            analysis_data: Dictionary containing comprehensive analysis results
            
        Returns:
            Dictionary containing pre-computed key actors and metrics
        """
        speaker_sentiment = analysis_data['speaker_level_analysis']['speaker_sentiment_analysis']['mean']
        topic_speaker_data = analysis_data['speaker_level_analysis']['topic_speaker_sentiment_analysis']
        
        # Find the moderator (speaker with sentiment closest to 0.0)
        moderator_id = min(speaker_sentiment.keys(), key=lambda x: abs(speaker_sentiment[x]))
        moderator_sentiment = speaker_sentiment[moderator_id]
        
        # Find primary debaters (all speakers except moderator)
        primary_debaters = [speaker for speaker in speaker_sentiment.keys() if speaker != moderator_id]
        
        # Find the most negative debater
        most_negative_debater = min(primary_debaters, key=lambda x: speaker_sentiment[x])
        most_negative_sentiment = speaker_sentiment[most_negative_debater]
        
        # Find the most polarizing topic
        topic_sentiment_differences = {}
        for key, data in topic_speaker_data.items():
            topic = data['topic']
            if topic not in topic_sentiment_differences:
                topic_sentiment_differences[topic] = []
            topic_sentiment_differences[topic].append(data['mean_sentiment'])
        
        # Calculate sentiment differences for each topic
        topic_polarization = {}
        for topic, sentiments in topic_sentiment_differences.items():
            if len(sentiments) > 1:
                topic_polarization[topic] = max(sentiments) - min(sentiments)
        
        # Handle empty topic_polarization case
        if topic_polarization:
            most_polarizing_topic = max(topic_polarization.keys(), key=lambda x: topic_polarization[x])
            most_polarizing_difference = topic_polarization[most_polarizing_topic]
            most_agreeable_topic = min(topic_polarization.keys(), key=lambda x: topic_polarization[x])
            most_agreeable_difference = topic_polarization[most_agreeable_topic]
        else:
            most_polarizing_topic = None
            most_polarizing_difference = 0
            most_agreeable_topic = None
            most_agreeable_difference = 0
        
        # Get emotion data for speaker profiles
        speaker_emotion_data = analysis_data['speaker_level_analysis']['speaker_emotion_analysis']
        
        return {
            'moderator_id': moderator_id,
            'moderator_sentiment': moderator_sentiment,
            'primary_debaters': primary_debaters,
            'most_negative_debater': most_negative_debater,
            'most_negative_sentiment': most_negative_sentiment,
            'most_polarizing_topic': most_polarizing_topic,
            'most_polarizing_difference': most_polarizing_difference,
            'most_agreeable_topic': most_agreeable_topic,
            'most_agreeable_difference': most_agreeable_difference,
            'speaker_sentiment': speaker_sentiment,
            'speaker_emotion_data': speaker_emotion_data,
            'topic_speaker_data': topic_speaker_data,
            'topic_polarization': topic_polarization
        }
    
    def generate_executive_summary(self, precomputed_data: Dict[str, Any]) -> str:
        """
        Generate the executive summary paragraph.
        
        Args:
            precomputed_data: Dictionary containing pre-computed key actors and metrics
            
        Returns:
            String containing the executive summary
        """
        moderator_id = precomputed_data['moderator_id']
        primary_debaters = precomputed_data['primary_debaters']
        most_negative_debater = precomputed_data['most_negative_debater']
        most_negative_sentiment = precomputed_data['most_negative_sentiment']
        most_polarizing_topic = precomputed_data['most_polarizing_topic']
        most_polarizing_difference = precomputed_data['most_polarizing_difference']
        most_agreeable_topic = precomputed_data['most_agreeable_topic']
        most_agreeable_difference = precomputed_data['most_agreeable_difference']
        
        # Sentence 1: Overall finding
        debater_a, debater_b = primary_debaters[0], primary_debaters[1] if len(primary_debaters) > 1 else primary_debaters[0]
        sentence1 = f"The analysis reveals a debate driven by significant polarization between the primary debaters, {debater_a} and {debater_b}, while {moderator_id} remained neutral."
        
        # Sentence 2: The "Who" - main driver of negativity
        gender = GENDER_MAP.get(most_negative_debater, 'unknown')
        sentence2 = f"The emotional tone was primarily driven by {most_negative_debater} ({gender}), who registered an average sentiment of {most_negative_sentiment:.3f}, in stark contrast to the other debater."
        
        # Sentence 3: The "Why" - source of conflict
        if most_polarizing_topic:
            sentence3 = f"This conflict was most acute during discussions of '{most_polarizing_topic}', where sentiment diverged by {most_polarizing_difference:.3f} points."
        else:
            sentence3 = "The debate showed consistent polarization across all topics discussed."
        
        # Sentence 4: Nuance - point of agreement
        if most_agreeable_topic and most_agreeable_difference < most_polarizing_difference:
            sentence4 = f"Conversely, the debaters found the most common ground on the topic of '{most_agreeable_topic}', indicating that substantive policy could be discussed with less animosity."
        else:
            sentence4 = "The debate showed consistent polarization across all topics, with limited areas of agreement."
        
        return f"{sentence1} {sentence2} {sentence3} {sentence4}"
    
    def generate_speaker_profiles(self, precomputed_data: Dict[str, Any]) -> str:
        """
        Generate individual speaker profiles.
        
        Args:
            precomputed_data: Dictionary containing pre-computed key actors and metrics
            
        Returns:
            String containing speaker profiles
        """
        speaker_sentiment = precomputed_data['speaker_sentiment']
        speaker_emotion_data = precomputed_data['speaker_emotion_data']
        topic_speaker_data = precomputed_data['topic_speaker_data']
        moderator_id = precomputed_data['moderator_id']
        
        profiles = []
        
        for speaker_id in speaker_sentiment.keys():
            gender = GENDER_MAP.get(speaker_id, 'unknown')
            sentiment_score = speaker_sentiment[speaker_id]
            
            # Find dominant emotion
            speaker_emotions = {}
            for emotion, speaker_counts in speaker_emotion_data.items():
                if speaker_id in speaker_counts:
                    speaker_emotions[emotion] = speaker_counts[speaker_id]
            
            if speaker_emotions:
                dominant_emotion = max(speaker_emotions.keys(), key=lambda x: speaker_emotions[x])
            else:
                dominant_emotion = "neutral"
            
            # Find most negative topic for this speaker
            speaker_topics = {key: data for key, data in topic_speaker_data.items() if data['speaker_id'] == speaker_id}
            if speaker_topics:
                most_negative_topic = min(speaker_topics.keys(), key=lambda x: speaker_topics[x]['mean_sentiment'])
                topic_name = speaker_topics[most_negative_topic]['topic']
            else:
                topic_name = "general discussion"
            
            # Create profile
            if speaker_id == moderator_id:
                profile = f"{speaker_id} ({gender.capitalize()}) - MODERATOR: This speaker's average sentiment was {sentiment_score:.3f}. Their most frequent emotion was '{dominant_emotion}'. They spoke most negatively on the topic of '{topic_name}'."
            else:
                profile = f"{speaker_id} ({gender.capitalize()}) - DEBATER: This speaker's average sentiment was {sentiment_score:.3f}. Their most frequent emotion was '{dominant_emotion}'. They spoke most negatively on the topic of '{topic_name}'."
            
            profiles.append(profile)
        
        return "\n".join(profiles)
    
    def generate_topic_deep_dive(self, precomputed_data: Dict[str, Any]) -> str:
        """
        Generate topic analysis with sentiment scores for each speaker.
        
        Args:
            precomputed_data: Dictionary containing pre-computed key actors and metrics
            
        Returns:
            String containing topic analysis
        """
        topic_polarization = precomputed_data['topic_polarization']
        topic_speaker_data = precomputed_data['topic_speaker_data']
        
        # Sort topics by polarization (most polarizing first)
        if topic_polarization:
            sorted_topics = sorted(topic_polarization.items(), key=lambda x: x[1], reverse=True)
            
            # Top 3 most polarizing topics
            polarizing_section = "TOP 3 MOST POLARIZING TOPICS:\n"
            for i, (topic, polarization) in enumerate(sorted_topics[:3], 1):
                polarizing_section += f"\n{i}. {topic} (Polarization: {polarization:.3f})\n"
                
                # Get sentiment scores for each speaker on this topic
                topic_speakers = {key: data for key, data in topic_speaker_data.items() if data['topic'] == topic}
                for key, data in topic_speakers.items():
                    speaker_id = data['speaker_id']
                    gender = GENDER_MAP.get(speaker_id, 'unknown')
                    sentiment = data['mean_sentiment']
                    polarizing_section += f"   - {speaker_id} ({gender}): {sentiment:.3f}\n"
            
            # Top 3 most agreeable topics
            agreeable_section = "\nTOP 3 MOST AGREEABLE TOPICS:\n"
            sorted_agreeable = sorted(topic_polarization.items(), key=lambda x: x[1])  # Smallest first
            for i, (topic, polarization) in enumerate(sorted_agreeable[:3], 1):
                agreeable_section += f"\n{i}. {topic} (Polarization: {polarization:.3f})\n"
                
                # Get sentiment scores for each speaker on this topic
                topic_speakers = {key: data for key, data in topic_speaker_data.items() if data['topic'] == topic}
                for key, data in topic_speakers.items():
                    speaker_id = data['speaker_id']
                    gender = GENDER_MAP.get(speaker_id, 'unknown')
                    sentiment = data['mean_sentiment']
                    agreeable_section += f"   - {speaker_id} ({gender}): {sentiment:.3f}\n"
        else:
            polarizing_section = "TOP 3 MOST POLARIZING TOPICS:\n\nNo topics with sufficient data for polarization analysis.\n"
            agreeable_section = "\nTOP 3 MOST AGREEABLE TOPICS:\n\nNo topics with sufficient data for agreement analysis.\n"
        
        return polarizing_section + agreeable_section
    
    def generate_full_report(self, analysis_data: Dict[str, Any]) -> str:
        """
        Generate the complete narrative report.
        
        Args:
            analysis_data: Dictionary containing comprehensive analysis results
            
        Returns:
            String containing the complete narrative report
        """
        # Pre-compute key actors and metrics
        precomputed_data = self.precompute_key_actors_and_metrics(analysis_data)
        
        # Generate sections
        executive_summary = self.generate_executive_summary(precomputed_data)
        speaker_profiles = self.generate_speaker_profiles(precomputed_data)
        topic_deep_dive = self.generate_topic_deep_dive(precomputed_data)
        
        # Combine into full report
        report = f"""
FINAL ANALYSIS SUMMARY
{'='*50}

EXECUTIVE SUMMARY
{'-'*20}
{executive_summary}

SPEAKER PROFILES
{'-'*20}
{speaker_profiles}

TOPIC ANALYSIS
{'-'*20}
{topic_deep_dive}

{'='*50}
Report generated automatically from comprehensive speaker-first analysis.
"""
        
        return report
    
    def save_report(self, report_text: str, output_path: str) -> None:
        """
        Save the narrative report to a text file.
        
        Args:
            report_text: The complete narrative report text
            output_path: Path to save the report
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"Narrative report saved to: {output_path}")

def main():
    """
    Main function to generate the narrative report.
    """
    generator = NarrativeReportGenerator()
    
    try:
        # Load the analysis data
        print("Loading comprehensive analysis data...")
        analysis_data = generator.load_analysis_data('/app/output/gender_topic_sentiment_report.json')
        

        
        # Generate the full narrative report
        print("Generating narrative report...")
        report_text = generator.generate_full_report(analysis_data)
        
        # Save the report
        generator.save_report(report_text, '/app/output/final_analysis_summary.txt')
        
        print("Narrative report generation complete!")
        print("- final_analysis_summary.txt (human-readable narrative report)")
        
    except Exception as e:
        print(f"Error generating narrative report: {e}")

if __name__ == "__main__":
    main() 