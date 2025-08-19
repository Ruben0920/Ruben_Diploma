import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

def load_data(faces_file, speakers_file):
    with open(faces_file, "r") as f:
        faces = json.load(f)
    with open(speakers_file, "r") as f:
        speakers = json.load(f)
    return faces, speakers

def map_faces_to_speakers(faces, speakers, similarity_threshold=0.4):
    results = []
    for speaker in speakers:
        speaker_faces = []
        for face in faces:
            # Check if face timestamp overlaps with speaker time segment
            if speaker["start_time"] <= face["timestamp"] <= speaker["end_time"]:
                # Compute similarity between face and speaker embeddings
                similarity = cosine_similarity(
                    np.array(face["embedding"]).reshape(1, -1),
                    np.array(speaker["voice_embedding"]).reshape(1, -1)
                )[0][0]
                if similarity > similarity_threshold:
                    speaker_faces.append({"face": face["face_image"], "similarity": similarity})
        results.append({"speaker": speaker["speaker_id"], "faces": speaker_faces})
    return results

def main():
    faces_file = "/app/output/faces.json"
    speakers_file = "/app/output/speakers.json"

    faces, speakers = load_data(faces_file, speakers_file)

    # Map faces to speakers 
    results = map_faces_to_speakers(faces, speakers)

    # Output results
    for result in results:
        print(f"Speaker {result['speaker']}:")
        for face in result["faces"]:
            print(f"  - Similarity: {face['similarity']}")

if __name__ == "__main__":
    main()
