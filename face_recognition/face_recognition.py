import os
import cv2
import json
import numpy as np
from deepface import DeepFace
from numpy.linalg import norm as l2_norm

# Get video filename and output directory from environment variables
VIDEO_FILENAME = os.getenv('VIDEO_FILENAME', 'test_video_1.mp4')
OUTPUT_BASE_DIR = os.getenv('OUTPUT_BASE_DIR', '/app/output')

# Extract video name without extension for directory structure
VIDEO_NAME = os.path.splitext(VIDEO_FILENAME)[0]

# Set dynamic paths with video-specific structure
VIDEO_FILE = f"/app/input/{VIDEO_FILENAME}"
SPEAKER_DIR = f"/app/output/{VIDEO_NAME}/speaker_diarization"
FACE_DIR = f"/app/output/{VIDEO_NAME}/face_recognition"

# Create face recognition output directory
os.makedirs(FACE_DIR, exist_ok=True)

# Input and output JSON paths
SPEAKERS_JSON = os.path.join(SPEAKER_DIR, "speakers_with_embeddings.json") 
OUT_JSON = os.path.join(FACE_DIR, "speaker_average_face_mapping.json")

FRAMES_PER_SECOND = 1
MIN_CONF = 0.90
BACKEND  = "retinaface"
MODEL    = "Facenet"
CLUSTER_SIM = 0.6 

def cosine_similarity(a,b):
    a=np.asarray(a); b=np.asarray(b)
    na,nb = l2_norm(a), l2_norm(b)
    if na==0 or nb==0: return 0.0
    return float(np.dot(a/na,b/nb).clip(-1,1))

def sample_frames(cap, start, end, fps_video):
    frames=[]; t=start
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    current_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0


    while t <= end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frm = cap.read()
        if not ret:
            break
        frames.append(frm)
        t += 1.0 / FRAMES_PER_SECOND 
        if t > end + (1.0 / FRAMES_PER_SECOND):
             break
    return frames

def extract_embs(frm):
    embs=[]
    try:
        regs = DeepFace.extract_faces(img_path=frm,
              detector_backend=BACKEND,
              enforce_detection=False,
              align=True)
        
        for r in regs:
            if r["confidence"] < MIN_CONF:
                continue
            try:
                rep = DeepFace.represent(
                    img_path=r["face"],
                    model_name=MODEL,
                    enforce_detection=False, 
                    detector_backend="skip"
                )
                if rep and len(rep) > 0: 
                    embs.append(np.array(rep[0]["embedding"], dtype=np.float32))
            except Exception as e_rep:
                pass 
    except Exception as e_ext:
        pass
    return embs

def cluster(embs):
    clusters=[]; cents=[]
    if not embs: # Handle case where embs is empty
        return []
    for e in embs:
        if not clusters:
            clusters.append([e]); cents.append(e); continue
        sims=[cosine_similarity(e,c) for c in cents]
        if not sims: # Handle case where cents is empty (shouldn't happen if clusters isn't)
            clusters.append([e]); cents.append(e); continue
        best=int(np.argmax(sims))
        if sims[best]>=CLUSTER_SIM:
            clusters[best].append(e)
            cents[best]=np.mean(np.stack(clusters[best]),axis=0)
        else:
            clusters.append([e]); cents.append(e)
    return clusters

def main():
    with open(SPEAKERS_JSON) as f:
        speakers_data=json.load(f) 

    cap=cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {VIDEO_FILE}")
        return
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0: 
        print("Warning: Video FPS could not be determined, defaulting to 25.0")
        video_fps = 25.0

    output_speakers = [] 
    for spk_info in speakers_data:
        speaker_id = spk_info["speaker_id"]
        print(f"Processing speaker: {speaker_id}")
        all_speaker_face_embeddings = []

        for seg_idx, seg in enumerate(spk_info["segments"]):
            st, en = seg["start_time"], seg["end_time"]
            if st >= en:
                continue

            # Sample frames for the current segment
            frs = sample_frames(cap, st, en, video_fps)
            
            segment_face_embeddings = []
            if not frs:
                pass

            for i, fr in enumerate(frs):
                embs_from_frame = extract_embs(fr)
                if embs_from_frame:
                    segment_face_embeddings.extend(embs_from_frame)
            
            if segment_face_embeddings:
                # cl = cluster(segment_face_embeddings)
                # if cl:
                #     dom = max(cl, key=len)
                #     avg_segment_embedding = np.mean(np.stack(dom), axis=0)
                #     # seg["face_embedding"] = avg_segment_embedding.tolist() # Store if you need per-segment
                all_speaker_face_embeddings.extend(segment_face_embeddings) # Collect all embeddings
            # else:
                # seg["face_embedding"] = None # No faces found in this segment
                # print(f"    No face embeddings found in segment {seg_idx + 1}")


        # After processing all segments for the current speaker
        current_speaker_output = {
            "speaker_id": speaker_id,
            # "segments": spk_info["segments"], # Keep original segments if needed, but remove face_embedding from them
            "voice_embedding_avg": spk_info.get("average_embedding") # Assuming the input JSON has this
        }

        if all_speaker_face_embeddings:
            # Calculate the average of all face embeddings for this speaker
            average_face_embedding_for_speaker = np.mean(np.stack(all_speaker_face_embeddings), axis=0)
            current_speaker_output["average_face_embedding"] = average_face_embedding_for_speaker.tolist()
            print(f"  Average face embedding calculated for {speaker_id} from {len(all_speaker_face_embeddings)} detected faces.")
        else:
            current_speaker_output["average_face_embedding"] = None
            print(f"  No face embeddings found for speaker {speaker_id} across all segments.")
        
        output_speakers.append(current_speaker_output)

    cap.release()
    
    # Save the new structure
    with open(OUT_JSON, "w") as o:
        json.dump(output_speakers, o, indent=2)
    print(f"'{OUT_JSON}' written with average face embeddings per speaker.")

if __name__=="__main__":
    main()