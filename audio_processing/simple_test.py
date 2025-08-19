#!/usr/bin/env python3
"""
Simple step-by-step test to identify where audio processing fails.
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

def step1_extract_audio():
    """Step 1: Extract audio from video."""
    video_path = "/app/input/1968 Canadian Federal Election Debate [N8ssqU9qOEo].mp4"
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    from pydub import AudioSegment
    print("Loading video with pydub...")
    audio = AudioSegment.from_file(video_path)
    print(f"Video loaded: {len(audio)}ms duration")
    
    audio = audio.set_channels(1)
    print("Converted to mono")
    
    output_dir = "/app/output/1968 Canadian Federal Election Debate [N8ssqU9qOEo]/speaker_diarization"
    os.makedirs(output_dir, exist_ok=True)
    
    audio_output_path = f"{output_dir}/audio.mp3"
    print(f"Exporting to: {audio_output_path}")
    audio.export(audio_output_path, format="mp3")
    
    if os.path.exists(audio_output_path):
        size = os.path.getsize(audio_output_path)
        print(f"Audio exported: {size} bytes")
        return audio_output_path
    else:
        raise FileNotFoundError("Audio file was not created")

def step2_transcribe():
    """Step 2: Transcribe audio with Whisper."""
    audio_path = "/app/output/1968 Canadian Federal Election Debate [N8ssqU9qOEo]/speaker_diarization/audio.mp3"
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    import whisper
    print("Loading Whisper model...")
    model = whisper.load_model("tiny")  # Use smallest model for testing
    print("Whisper model loaded")
    
    print("Transcribing...")
    result = model.transcribe(audio_path, word_timestamps=True)
    print("Transcription completed")
    
    # Save transcription
    output_dir = "/app/output/1968 Canadian Federal Election Debate [N8ssqU9qOEo]/speaker_diarization"
    transcription_path = f"{output_dir}/transcription_with_timestamps.json"
    
    import json
    with open(transcription_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"Transcription saved: {transcription_path}")
    return result

def step3_diarize():
    """Step 3: Run speaker diarization."""
    audio_path = "/app/output/1968 Canadian Federal Election Debate [N8ssqU9qOEo]/speaker_diarization/audio.mp3"
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    from pyannote.audio import Pipeline
    print("Loading Pyannote pipeline...")
    # Get Hugging Face token from environment variable
    hf_token = os.getenv('HUGGING_FACE_TOKEN')
    if not hf_token:
        raise ValueError("HUGGING_FACE_TOKEN environment variable not set")
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=hf_token
    )
    print("Pipeline loaded")
    
    print("Running diarization...")
    diarization = pipeline(audio_path)
    print("Diarization completed")
    
    # Count segments
    segments = list(diarization.itertracks(yield_label=True))
    print(f"Found {len(segments)} segments")
    
    # Save simple output
    output_dir = "/app/output/1968 Canadian Federal Election Debate [N8ssqU9qOEo]/speaker_diarization"
    speakers_path = f"{output_dir}/speakers_with_embeddings.json"
    
    import json
    speakers_data = []
    for segment, _, speaker in segments:
        speakers_data.append({
            "speaker_id": speaker,
            "start_time": segment.start,
            "end_time": segment.end
        })
    
    with open(speakers_path, 'w') as f:
        json.dump(speakers_data, f, indent=4)
    
    print(f"Speakers data saved: {speakers_path}")
    return speakers_data

def main():
    """Run all steps sequentially."""
    print("🔍 Audio Processing Step-by-Step Test")
    print("=" * 50)
    
    # Step 1: Extract audio
    audio_path = test_step("Audio Extraction", step1_extract_audio)
    if not audio_path:
        print("❌ Stopping due to audio extraction failure")
        return
    
    # Step 2: Transcribe
    transcription = test_step("Whisper Transcription", step2_transcribe)
    if not transcription:
        print("❌ Stopping due to transcription failure")
        return
    
    # Step 3: Diarize
    speakers = test_step("Speaker Diarization", step3_diarize)
    if not speakers:
        print("❌ Stopping due to diarization failure")
        return
    
    print("\n" + "=" * 50)
    print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    main()
