#!/usr/bin/env python3
"""
Simplified test script to debug audio processing step by step.
"""

import os
import sys
import traceback

def test_step(step_name, func):
    """Test a single step and show detailed error if it fails."""
    print(f"\n{'='*50}")
    print(f"Testing: {step_name}")
    print(f"{'='*50}")
    
    try:
        result = func()
        print(f"✅ {step_name} SUCCESS")
        return result
    except Exception as e:
        print(f"❌ {step_name} FAILED")
        print(f"Error: {str(e)}")
        print(f"Type: {type(e).__name__}")
        print("\nFull traceback:")
        traceback.print_exc()
        return None

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import pydub
        print("✅ pydub imported")
    except Exception as e:
        print(f"❌ pydub import failed: {e}")
    
    try:
        import whisper
        print("✅ whisper imported")
    except Exception as e:
        print(f"❌ whisper import failed: {e}")
    
    try:
        from pyannote.audio import Pipeline, Model, Inference
        print("✅ pyannote.audio imported")
    except Exception as e:
        print(f"❌ pyannote.audio import failed: {e}")
    
    try:
        import torchaudio
        print("✅ torchaudio imported")
    except Exception as e:
        print(f"❌ torchaudio import failed: {e}")
    
    try:
        import torch
        print("✅ torch imported")
    except Exception as e:
        print(f"❌ torch import failed: {e}")

def test_audio_extraction():
    """Test audio extraction step."""
    video_path = "/app/input/1968 Canadian Federal Election Debate [N8ssqU9qOEo].mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return None
    
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(video_path)
        audio = audio.set_channels(1)
        
        output_dir = "/app/output/1968 Canadian Federal Election Debate [N8ssqU9qOEo]/speaker_diarization"
        os.makedirs(output_dir, exist_ok=True)
        
        audio_output_path = f"{output_dir}/audio.mp3"
        audio.export(audio_output_path, format="mp3")
        
        print(f"✅ Audio extracted to: {audio_output_path}")
        print(f"✅ Audio duration: {len(audio)/1000:.2f} seconds")
        return audio_output_path
        
    except Exception as e:
        print(f"❌ Audio extraction failed: {e}")
        return None

def test_whisper_transcription():
    """Test Whisper transcription step."""
    audio_path = "/app/output/1968 Canadian Federal Election Debate [N8ssqU9qOEo]/speaker_diarization/audio.mp3"
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return None
    
    try:
        import whisper
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print("✅ Whisper model loaded")
        
        print("Transcribing audio...")
        result = model.transcribe(audio_path, word_timestamps=True)
        print("✅ Transcription completed")
        
        # Save transcription
        output_dir = "/app/output/1968 Canadian Federal Election Debate [N8ssqU9qOEo]/speaker_diarization"
        transcription_path = f"{output_dir}/transcription_with_timestamps.json"
        
        import json
        with open(transcription_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        print(f"✅ Transcription saved to: {transcription_path}")
        return result
        
    except Exception as e:
        print(f"❌ Whisper transcription failed: {e}")
        return None

def test_pyannote_diarization():
    """Test Pyannote speaker diarization step."""
    audio_path = "/app/output/1968 Canadian Federal Election Debate [N8ssqU9qOEo]/speaker_diarization/audio.mp3"
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return None
    
    try:
        from pyannote.audio import Pipeline, Model, Inference
        
        print("Loading Pyannote models...")
        # Get Hugging Face token from environment variable
        hf_token = os.getenv('HUGGING_FACE_TOKEN')
        if not hf_token:
            raise ValueError("HUGGING_FACE_TOKEN environment variable not set")
        
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=hf_token
        )
        print("✅ Speaker diarization pipeline loaded")
        
        embedding_base_model = Model.from_pretrained(
            "pyannote/embedding",
            use_auth_token=hf_token
        )
        print("✅ Embedding model loaded")
        
        embedding_inference = Inference(
            embedding_base_model,
            window="sliding",
            duration=5.0,
            step=1.0
        )
        print("✅ Embedding inference setup completed")
        
        print("Running speaker diarization...")
        diarization = pipeline(audio_path)
        print("✅ Speaker diarization completed")
        
        # Save results
        output_dir = "/app/output/1968 Canadian Federal Election Debate [N8ssqU9qOEo]/speaker_diarization"
        speakers_path = f"{output_dir}/speakers_with_embeddings.json"
        
        # Simple output for now
        import json
        speakers_data = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            speakers_data.append({
                "speaker_id": speaker,
                "start_time": segment.start,
                "end_time": segment.end
            })
        
        with open(speakers_path, 'w') as f:
            json.dump(speakers_data, f, indent=4)
        
        print(f"✅ Speakers data saved to: {speakers_path}")
        return speakers_data
        
    except Exception as e:
        print(f"❌ Pyannote diarization failed: {e}")
        return None

def main():
    """Main test function."""
    print("🔍 Audio Processing Debug Test")
    print("=" * 50)
    
    # Test imports first
    test_imports()
    
    # Test each step
    audio_path = test_step("Audio Extraction", test_audio_extraction)
    if audio_path:
        transcription = test_step("Whisper Transcription", test_whisper_transcription)
        if transcription:
            speakers = test_step("Pyannote Diarization", test_pyannote_diarization)
    
    print("\n" + "=" * 50)
    print("Debug test completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()
