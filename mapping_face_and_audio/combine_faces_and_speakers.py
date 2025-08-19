"""
Mapping module to combine face recognition and speaker diarization results.
Links detected faces to speaker segments based on temporal overlap and similarity.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

def load_data(faces_file, speakers_file):
    """
    Load face and speaker data from JSON files.
    
    Args:
        faces_file: Path to faces JSON file
        speakers_file: Path to speakers JSON file
        
    Returns:
        tuple: (faces data, speakers data)
    """
    with open(faces_file, "r") as f:
        faces = json.load(f)
    with open(speakers_file, "r") as f:
        speakers = json.load(f)
    return faces, speakers

def map_faces_to_speakers(faces, speakers, similarity_threshold=0.4):
    """
    Map detected faces to speakers based on temporal overlap and embedding similarity.
    
    Args:
        faces: List of face detection results with timestamps and embeddings
        speakers: List of speaker segments with voice embeddings
        similarity_threshold: Minimum similarity score for face-speaker matching
        
    Returns:
        list: Results mapping speakers to their associated faces
    """
    results = []
    for speaker in speakers:
        speaker_faces = []
        for face in faces:
            if speaker["start_time"] <= face["timestamp"] <= speaker["end_time"]:
                similarity = cosine_similarity(
                    np.array(face["embedding"]).reshape(1, -1),
                    np.array(speaker["voice_embedding"]).reshape(1, -1)
                )[0][0]
                if similarity > similarity_threshold:
                    speaker_faces.append({"face": face["face_image"], "similarity": similarity})
        results.append({"speaker": speaker["speaker_id"], "faces": speaker_faces})
    return results

def main():
    """
    Main function to combine face detection and speaker diarization results.
    Maps detected faces to speakers based on temporal and embedding similarity.
    """
    faces_file = "/app/output/faces.json"
    speakers_file = "/app/output/speakers.json"

    faces, speakers = load_data(faces_file, speakers_file)
    results = map_faces_to_speakers(faces, speakers)

    for result in results:
        print(f"Speaker {result['speaker']}:")
        for face in result["faces"]:
            print(f"  - Similarity: {face['similarity']}")

if __name__ == "__main__":
    main()
