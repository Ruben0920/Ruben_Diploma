import json

def test_json_loading():
    try:
        with open('/app/output/gender_topic_sentiment_report.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("JSON loaded successfully")
        print(f"Keys: {list(data.keys())}")
        print(f"Speaker level keys: {list(data['speaker_level_analysis'].keys())}")
        
        speaker_sentiment = data['speaker_level_analysis']['speaker_sentiment_analysis']['mean']
        print(f"Speaker sentiment: {speaker_sentiment}")
        
        # Test the problematic operation
        moderator_id = min(speaker_sentiment.keys(), key=lambda x: abs(speaker_sentiment[x]))
        print(f"Moderator ID: {moderator_id}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_json_loading() 