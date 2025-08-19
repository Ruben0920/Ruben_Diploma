#!/usr/bin/env python3
"""
Minimal test with short audio to isolate the issue.
"""

import os
import sys
import traceback

def test_short_audio():
    """Test with just the first 30 seconds of audio."""
    video_path = "/app/input/1968 Canadian Federal Election Debate [N8ssqU9qOEo].mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return False
    
    from pydub import AudioSegment
    print("Loading video...")
    audio = AudioSegment.from_file(video_path)
    print(f"Full video: {len(audio)}ms")
    
    # Extract just first 30 seconds
    short_audio = audio[:30000]  # 30 seconds
    print(f"Short audio: {len(short_audio)}ms")
    
    output_dir = "/app/output/1968 Canadian Federal Election Debate [N8ssqU9qOEo]/speaker_diarization"
    os.makedirs(output_dir, exist_ok=True)
    
    short_audio_path = f"{output_dir}/short_audio.mp3"
    print(f"Exporting short audio...")
    short_audio.export(short_audio_path, format="mp3")
    
    if os.path.exists(short_audio_path):
        size = os.path.getsize(short_audio_path)
        print(f"Short audio exported: {size} bytes")
        
        # Test Whisper on short audio
        try:
            import whisper
            print("Loading Whisper tiny model...")
            model = whisper.load_model("tiny")
            print("Model loaded, transcribing short audio...")
            
            result = model.transcribe(short_audio_path, word_timestamps=True)
            print(f"Transcription successful! Found {len(result.get('segments', []))} segments")
            
            # Save result
            transcription_path = f"{output_dir}/short_transcription.json"
            import json
            with open(transcription_path, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"Short transcription saved: {transcription_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Whisper failed: {e}")
            traceback.print_exc()
            return False
    else:
        print("❌ Short audio file not created")
        return False

if __name__ == "__main__":
    print("🔍 Mini Test - Short Audio Only")
    print("=" * 40)
    
    success = test_short_audio()
    
    if success:
        print("\n✅ Mini test completed successfully!")
    else:
        print("\n❌ Mini test failed!")

