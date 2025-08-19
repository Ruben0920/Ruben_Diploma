import os
from pydub import AudioSegment
from whisper import load_model
from pyannote.audio import Pipeline, Model, Inference
import json
import torchaudio
import torch
import numpy as np
import math

MIN_SEGMENT_DURATION = 5.0 


def save_speakers_to_file(diarization, audio_path, embedding_inference, sample_rate, output_file):
    """
    Save speaker segments and embeddings plus one average embedding per speaker to a JSON file.
    Loads the full audio once and converts to mono 16 kHz, extracts exactly MIN_SEGMENT_DURATION
    seconds per turn with zero-padding for shorter segments, and computes average embeddings.
    
    Args:
        diarization: Speaker diarization results from pyannote
        audio_path: Path to the audio file
        embedding_inference: Pyannote embedding inference object
        sample_rate: Audio sample rate (unused in current implementation)
        output_file: Output JSON file path
    """
    waveform, orig_sr = torchaudio.load(audio_path, normalize=True)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    TARGET_SR = 16000
    if orig_sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    total_frames = waveform.shape[1]
    frames_per_seg = int(MIN_SEGMENT_DURATION * TARGET_SR)

    speakers = {}

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_frame = int(segment.start * TARGET_SR)
        end_frame = int(segment.end * TARGET_SR)
        if segment.end - segment.start >= MIN_SEGMENT_DURATION:
            embedding = None
            if start_frame < total_frames:
                end_frame = min(end_frame, total_frames)
                chunk = waveform[:, start_frame:end_frame]

                if chunk.shape[1] < frames_per_seg:
                    pad_amt = frames_per_seg - chunk.shape[1]
                    chunk = torch.cat([chunk, torch.zeros((1, pad_amt))], dim=1)

                try:
                    out = embedding_inference({
                        "waveform": chunk,
                        "sample_rate": TARGET_SR
                    })
                    emb_np = out.data         
                    embedding = emb_np.squeeze().tolist()
                    print(f"[DEBUG] speaker={speaker} dur={segment.end - segment.start:.2f}s emb_shape={emb_np.shape}")
                except Exception as e:
                    print(f"[ERROR] embedding failed for {segment}: {e}")

            seg_info = {
                "start_time": segment.start,
                "end_time": segment.end,
                "voice_embedding": embedding
            }

            if speaker not in speakers:
                speakers[speaker] = {"segments": [], "embeddings": []}
            speakers[speaker]["segments"].append(seg_info)
            if embedding is not None:
                speakers[speaker]["embeddings"].append(embedding)

    output = []
    for spk, info in speakers.items():
        embeddings = info["embeddings"]
        avg = np.mean(np.vstack(embeddings), axis=0).tolist() if embeddings else None
        output.append({
            "speaker_id": spk,
            "segments": info["segments"],
            "average_embedding": avg
        })

    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)


def diarize_audio(audio_file, output_file):
    """
    Perform speaker diarization on an audio file and extract voice embeddings.
    
    Args:
        audio_file: Path to input audio file
        output_file: Path to output JSON file for speaker data
    """
    hf_token = os.getenv('HUGGING_FACE_TOKEN')
    if not hf_token:
        raise ValueError("HUGGING_FACE_TOKEN environment variable not set")
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=hf_token
    )
    embedding_base_model = Model.from_pretrained(
        "pyannote/embedding",
        use_auth_token=hf_token
    )
    embedding_inference = Inference(
        embedding_base_model,
        window="sliding",
        duration=MIN_SEGMENT_DURATION,
        step=1.0
    )

    diarization = pipeline(audio_file)
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        print(f"Speaker {speaker} from {segment.start:.2f}s to {segment.end:.2f}s")

    save_speakers_to_file(diarization, audio_file, embedding_inference, None, output_file)


def extract_audio(video_path, audio_output_path):
    """
    Extract mono audio from a video file and save as MP3.
    
    Args:
        video_path: Path to input video file
        audio_output_path: Path to output audio file
        
    Returns:
        str: Path to the extracted audio file
    """
    os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)
    audio = AudioSegment.from_file(video_path)
    audio = audio.set_channels(1)
    audio.export(audio_output_path, format="mp3")
    return audio_output_path


def transcribe_audio(audio_path, model_name="tiny"):
    """
    Transcribe audio using OpenAI Whisper with word-level timestamps.
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model size (default: 'tiny')
        
    Returns:
        dict: Transcription result with segments and word timestamps
    """
    model = load_model(model_name)
    result = model.transcribe(audio_path, word_timestamps=True)
    return result


def save_transcription_to_file(transcription_result, output_path):
    """
    Save transcription results to a JSON file.
    
    Args:
        transcription_result: Whisper transcription output
        output_path: Path to output JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(transcription_result, file, indent=4)


def main():
    """
    Main function to process a video file through the audio analysis pipeline.
    Extracts audio, performs transcription, speaker diarization, and voice embedding extraction.
    """
    video_filename = os.getenv('VIDEO_FILENAME')
    if not video_filename:
        print("Error: VIDEO_FILENAME environment variable not set!")
        print("Using default video file...")
        video_filename = "test_video_1.mp4"
    
    output_base_dir = os.getenv('OUTPUT_BASE_DIR', '/app/output')
    video_name = os.path.splitext(video_filename)[0]
    video_path = f"/app/input/{video_filename}"
    
    output_dir = f"/app/output/{video_name}/speaker_diarization"
    os.makedirs(output_dir, exist_ok=True)
    
    audio_output_path = f"{output_dir}/audio.mp3"
    transcription_output_path = f"{output_dir}/transcription_with_timestamps.json"
    speakers_output_path = f"{output_dir}/speakers_with_embeddings.json"

    print("Extracting audio...")
    extracted_audio_path = extract_audio(video_path, audio_output_path)
    print(f"Audio saved as: {extracted_audio_path}")

    print("Transcribing audio...")
    transcription_result = transcribe_audio(extracted_audio_path)
    print("Audio transcription completed.")

    print("Saving transcription with timestamps...")
    save_transcription_to_file(transcription_result, transcription_output_path)
    print(f"Transcription saved as: {transcription_output_path}")

    print("Performing speaker diarization and embedding extraction...")
    diarize_audio(extracted_audio_path, speakers_output_path)
    print(f"Speaker diarization with embeddings saved as: {speakers_output_path}")

    with open(speakers_output_path, "r") as f:
        speakers = json.load(f)

    trans_segments = transcription_result.get("segments", [])

    def get_segment_text(start, end):
        pieces = []
        for seg in trans_segments:
            if seg["end"] > start and seg["start"] < end:
                pieces.append(seg["text"].strip())
        return "\n".join(pieces).strip()

    for speaker in speakers:
        for seg in speaker["segments"]:
            seg["transcription"] = get_segment_text(seg["start_time"], seg["end_time"])

    with open(speakers_output_path, "w") as f:
        json.dump(speakers, f, indent=2)
    print(f"Added transcriptions into {speakers_output_path}")


if __name__ == "__main__":
    main()
