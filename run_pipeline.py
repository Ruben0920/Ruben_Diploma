#!/usr/bin/env python3
"""
Master control script to orchestrate the video processing pipeline.
Processes videos in parallel using ThreadPoolExecutor for improved performance.
Generates metadata for each video before processing.
"""

import os
import subprocess
import json
import time
from pathlib import Path
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import threading
import cv2

MAX_CONCURRENT_WORKERS = 4
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}

print_lock = threading.Lock()

def thread_safe_print(message, video_name=None):
    """
    Thread-safe printing function that prefixes messages with video name.
    """
    with print_lock:
        if video_name:
            print(f"[{video_name}] - {message}")
        else:
            print(message)

def extract_video_metadata(video_path, video_name):
    """
    Extract metadata from a video file using OpenCV.
    """
    thread_safe_print("Extracting video metadata...", video_name)
    
    metadata = {
        "filename": video_name,
        "processing_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "file_size_bytes": 0,
        "duration_seconds": 0.0,
        "resolution": {"width": 0, "height": 0},
        "fps": 0.0
    }
    
    try:
        metadata["file_size_bytes"] = os.path.getsize(video_path)
        
        cap = cv2.VideoCapture(str(video_path))
        
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if fps > 0:
                duration = frame_count / fps
            else:
                duration = 0.0
            
            metadata["fps"] = round(fps, 2)
            metadata["duration_seconds"] = round(duration, 2)
            metadata["resolution"]["width"] = width
            metadata["resolution"]["height"] = height
            
            cap.release()
            
            thread_safe_print(
                f"Metadata extracted: {width}x{height}, {fps:.1f}fps, {duration:.1f}s", 
                video_name
            )
        else:
            thread_safe_print("Warning: Could not open video with OpenCV", video_name)
            
    except Exception as e:
        thread_safe_print(f"Error extracting metadata: {str(e)}", video_name)
    
    return metadata

def save_metadata(metadata, output_dir, video_name):
    """
    Save metadata to a JSON file in the video's output directory.
    """
    metadata_file = output_dir / "metadata.json"
    
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        thread_safe_print(f"Metadata saved to {metadata_file.name}", video_name)
    except Exception as e:
        thread_safe_print(f"Error saving metadata: {str(e)}", video_name)

class VideoProcessingPipeline:
    """
    Main pipeline class for processing videos through all analysis stages.
    """
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.input_dir = self.base_dir / "input" / "videos"
        self.output_dir = self.base_dir / "output"
        
        self.services = [
            "audio_processing",
            "face_recognition",
            "polarization_analysis"
        ]
        
        self.custom_network_available = False
        
    def get_video_files(self):
        """Get all video files from the input directory."""
        video_files = []
        
        if not self.input_dir.exists():
            thread_safe_print(f"Error: Input directory {self.input_dir} does not exist!")
            return []
            
        for file in self.input_dir.iterdir():
            if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS:
                video_files.append(file.name)
                
        return sorted(video_files)
    
    def create_output_structure(self, video_name):
        """Create output directory structure for a specific video."""
        video_dir_name = Path(video_name).stem
        video_output_dir = self.output_dir / video_dir_name
        
        subdirs = [
            "speaker_diarization",
            "face_recognition",
            "polarization_analysis",
            "polarization_analysis/visualizations"
        ]
        
        for subdir in subdirs:
            dir_path = video_output_dir / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return video_output_dir
    
    def ensure_docker_network(self):
        """
        Ensure Docker network exists before parallel processing.
        This prevents network creation conflicts during parallel execution.
        """
        try:
            # Check if network exists
            result = subprocess.run(
                ["docker", "network", "ls", "--filter", "name=ruben_diploma_default", "--format", "{{.Name}}"],
                capture_output=True,
                text=True,
                cwd=self.base_dir
            )
            
            if "ruben_diploma_default" not in result.stdout:
                thread_safe_print("Creating Docker network for parallel processing...")
                create_result = subprocess.run(
                    ["docker", "network", "create", "ruben_diploma_default"],
                    capture_output=True,
                    text=True,
                    cwd=self.base_dir
                )
                if create_result.returncode == 0:
                    thread_safe_print("✓ Docker network created successfully")
                else:
                    thread_safe_print(f"Warning: Could not create network: {create_result.stderr}")
                    # Try to use existing default network instead
                    thread_safe_print("Trying to use existing default network...")
                    return False
            else:
                thread_safe_print("✓ Docker network already exists")
                return True
                
        except Exception as e:
            thread_safe_print(f"Warning: Could not manage Docker network: {e}")
            return False
        
        return True

    def get_service_image_name(self, service_name):
        """Get the Docker image name for a service."""
        # Use the existing naming convention from docker-compose
        return f"ruben_diploma-{service_name}"

    def run_service(self, service_name, video_filename, video_output_dir):
        """
        Run a specific Docker service for a video using direct docker run.
        This avoids network conflicts in parallel execution.
        """
        thread_safe_print(f"Starting {service_name}...", video_filename)
        
        # Build the service image if it doesn't exist
        self.build_service_image(service_name, video_filename)
        
        env_vars = {
            "VIDEO_FILENAME": video_filename,
            "OUTPUT_BASE_DIR": str(video_output_dir.relative_to(self.base_dir)),
            "PYTHONUNBUFFERED": "1",
            "MAX_WORKERS": "2"
        }
        
        # Get absolute paths for volume mounting
        input_path = str(self.input_dir.absolute())
        output_path = str(self.output_dir.absolute())
        
        image_name = self.get_service_image_name(service_name)
        
        cmd = [
            "docker", "run", "--rm",
            "--memory=16g", "--cpus=4",
            "-v", f"{input_path}:/app/input:ro",
            "-v", f"{output_path}:/app/output"
        ]
        
        # Only add custom network if it exists
        if self.custom_network_available:
            cmd.extend(["--network", "ruben_diploma_default"])
        
        # Add environment variables
        for key, value in env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])
        
        cmd.append(image_name)
        
        try:
            thread_safe_print(f"Running command: {' '.join(cmd[:6])}...", video_filename)
            
            # Use Popen for long-running processes without timeout
            process = subprocess.Popen(
                cmd,
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for completion without timeout
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                thread_safe_print(f"Error in {service_name}:", video_filename)
                if stderr:
                    thread_safe_print(f"STDERR: {stderr[:1000] if len(stderr) > 1000 else stderr}", video_filename)
                else:
                    thread_safe_print("STDERR: No error output", video_filename)
                if stdout:
                    thread_safe_print(f"STDOUT: {stdout[:500]}", video_filename)
                
                # Check if output files were created
                self.check_service_outputs(service_name, video_output_dir, video_filename)
                return False
            
            thread_safe_print(f"{service_name} completed successfully", video_filename)
            return True
            
        except Exception as e:
            thread_safe_print(f"Exception in {service_name}: {str(e)}", video_filename)
            return False

    def build_service_image(self, service_name, video_filename):
        """Build Docker image for a service if it doesn't exist."""
        image_name = self.get_service_image_name(service_name)
        
        # Check if image exists
        check_cmd = ["docker", "images", "-q", image_name]
        result = subprocess.run(check_cmd, capture_output=True, text=True, cwd=self.base_dir)
        
        if not result.stdout.strip():
            thread_safe_print(f"Building {service_name} image...", video_filename)
            # Use docker-compose to build the image to ensure consistency
            build_cmd = ["docker-compose", "build", service_name]
            
            build_result = subprocess.run(
                build_cmd, 
                capture_output=True, 
                text=True, 
                cwd=self.base_dir
            )
            
            if build_result.returncode != 0:
                thread_safe_print(f"Failed to build {service_name}: {build_result.stderr[:200]}", video_filename)
            else:
                thread_safe_print(f"✓ {service_name} image built successfully", video_filename)
    
    def check_service_outputs(self, service_name, video_output_dir, video_filename):
        """Check what output files were created by a service to help diagnose failures."""
        thread_safe_print(f"Checking outputs for {service_name}...", video_filename)
        
        if service_name == "audio_processing":
            expected_files = [
                "speaker_diarization/speakers_with_embeddings.json",
                "speaker_diarization/transcription_with_timestamps.json"
            ]
        elif service_name == "face_recognition":
            expected_files = [
                "face_recognition/faces.json"
            ]
        elif service_name == "polarization_analysis":
            expected_files = [
                "polarization_analysis/enhanced_polarization_report.json",
                "polarization_analysis/gender_topic_sentiment_report.json"
            ]
        else:
            return
        
        for expected_file in expected_files:
            file_path = video_output_dir / expected_file
            if file_path.exists():
                file_size = file_path.stat().st_size
                thread_safe_print(f"✓ Found: {expected_file} ({file_size} bytes)", video_filename)
            else:
                thread_safe_print(f"✗ Missing: {expected_file}", video_filename)
    
    def check_service_dependencies(self, service_name, video_output_dir, video_filename):
        """Check if required input files exist before running a service."""
        thread_safe_print(f"Checking dependencies for {service_name}...", video_filename)
        
        if service_name == "audio_processing":
            # Audio processing only needs the input video file
            return True
        elif service_name == "face_recognition":
            # Face recognition needs speakers_with_embeddings.json from audio processing
            required_file = video_output_dir / "speaker_diarization/speakers_with_embeddings.json"
            if not required_file.exists():
                thread_safe_print(f"✗ Missing required input: speakers_with_embeddings.json", video_filename)
                return False
            thread_safe_print(f"✓ Found required input: speakers_with_embeddings.json", video_filename)
            return True
        elif service_name == "polarization_analysis":
            # Polarization analysis needs speakers_with_embeddings.json from audio processing
            required_file = video_output_dir / "speaker_diarization/speakers_with_embeddings.json"
            if not required_file.exists():
                thread_safe_print(f"✗ Missing required input: speakers_with_embeddings.json", video_filename)
                return False
            thread_safe_print(f"✓ Found required input: speakers_with_embeddings.json", video_filename)
            return True
        else:
            return True

    def process_single_video(self, video_filename):
        """
        Process a single video through all services sequentially.
        This is the main function that will be executed in parallel.
        """
        start_time = time.time()
        
        thread_safe_print("=" * 60, video_filename)
        thread_safe_print("Starting processing", video_filename)
        
        video_path = self.input_dir / video_filename
        
        video_output_dir = self.create_output_structure(video_filename)
        thread_safe_print(f"Output directory: {video_output_dir}", video_filename)
        
        metadata = extract_video_metadata(video_path, video_filename)
        save_metadata(metadata, video_output_dir, video_filename)
        
        successful_services = []
        failed_services = []
        
        # Process services sequentially for each video (audio -> face -> polarization)
        for i, service in enumerate(self.services):
            thread_safe_print(f"Starting {service} (step {i+1}/{len(self.services)})...", video_filename)
            
            # Check dependencies before running the service
            if not self.check_service_dependencies(service, video_output_dir, video_filename):
                thread_safe_print(f"✗ {service} dependencies not met. Skipping.", video_filename)
                failed_services.append(service)
                continue
            
            success = self.run_service(service, video_filename, video_output_dir)
            
            if success:
                successful_services.append(service)
                thread_safe_print(f"✓ {service} completed successfully", video_filename)
                
                # Wait a moment for file system to sync
                time.sleep(1)
            else:
                failed_services.append(service)
                thread_safe_print(f"✗ {service} failed", video_filename)
                
                # If a critical service fails, stop processing this video
                if service == "audio_processing":
                    thread_safe_print("Critical failure: audio_processing failed. Stopping video processing.", video_filename)
                    break
                elif service == "face_recognition":
                    thread_safe_print("Warning: face_recognition failed. Polarization analysis may not work properly.", video_filename)
        
        elapsed_time = time.time() - start_time
        
        thread_safe_print("=" * 60, video_filename)
        thread_safe_print(f"Processing complete in {elapsed_time:.2f} seconds", video_filename)
        thread_safe_print(f"Successful: {', '.join(successful_services) if successful_services else 'None'}", video_filename)
        thread_safe_print(f"Failed: {', '.join(failed_services) if failed_services else 'None'}", video_filename)
        
        success = len(successful_services) == len(self.services)
        return (video_filename, success, elapsed_time, successful_services, failed_services)
    
    def run_serial(self):
        """
        Main execution method with serial processing.
        Processes videos one at a time for better reliability and resource management.
        """
        print("\n" + "=" * 70)
        print("Multi-Modal Video Analysis Pipeline - Serial Processing")
        print("=" * 70)
        
        video_files = self.get_video_files()
        
        if not video_files:
            print("No video files found in input/videos directory!")
            return
        
        # DEBUG MODE: Process only the first video
        video_files = video_files[:1]
        
        print(f"\nFound {len(video_files)} video(s) to process (DEBUG MODE - ONE VIDEO ONLY):")
        for i, video in enumerate(video_files, 1):
            print(f"  {i}. {video}")
        
        print(f"\nStarting serial processing (DEBUG MODE - ONE VIDEO ONLY)...")
        print("-" * 70)
        
        results = []
        total_start_time = time.time()
        
        for i, video_filename in enumerate(video_files, 1):
            print(f"\n{'='*20} Processing Video {i}/{len(video_files)} {'='*20}")
            print(f"Current: {video_filename}")
            
            try:
                result = self.process_single_video(video_filename)
                results.append(result)
                
                if result[1]:  # If successful
                    print(f"✅ Video {i} completed successfully")
                else:
                    print(f"❌ Video {i} failed")
                    
            except Exception as e:
                print(f"💥 Video {i} crashed: {str(e)}")
                results.append((video_filename, False, 0, [], self.services))
        
        total_elapsed = time.time() - total_start_time
        
        self.print_final_summary(results, total_elapsed, len(video_files))
    
    def print_final_summary(self, results, total_time, total_videos):
        """
        Print a comprehensive final summary of all processing results.
        """
        print("\n" + "=" * 70)
        print("FINAL SUMMARY - PARALLEL PROCESSING COMPLETE")
        print("=" * 70)
        
        successful_videos = [r for r in results if r[1]]
        failed_videos = [r for r in results if not r[1]]
        
        print(f"\nProcessing Statistics:")
        print(f"  Total videos processed: {total_videos}")
        print(f"  Successful: {len(successful_videos)}")
        print(f"  Failed/Partial: {len(failed_videos)}")
        print(f"  Total processing time: {total_time:.2f} seconds")
        print(f"  Average time per video: {total_time/total_videos:.2f} seconds")
        
        if successful_videos:
            print("\nSuccessfully Processed Videos:")
            for video, _, proc_time, services, _ in successful_videos:
                print(f"  • {video} ({proc_time:.1f}s)")
        
        if failed_videos:
            print("\nFailed or Partially Processed Videos:")
            for video, _, proc_time, successful, failed in failed_videos:
                print(f"  • {video} ({proc_time:.1f}s)")
                if successful:
                    print(f"    Completed: {', '.join(successful)}")
                if failed:
                    print(f"    Failed: {', '.join(failed)}")
        
        print(f"\nAll outputs saved to: {self.output_dir}")
        print("\n" + "=" * 70)
        
        if total_videos > 1:
            sequential_estimate = sum([r[2] for r in results])
            speedup = sequential_estimate / total_time
            print(f"Performance: ~{speedup:.1f}x speedup compared to sequential processing")
            print("=" * 70)


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import cv2
        return True
    except ImportError:
        print("\nWarning: opencv-python is not installed.")
        print("Install it with: pip install opencv-python")
        print("Metadata extraction will be limited without it.\n")
        return False


def main():
    """Main entry point."""
    check_dependencies()
    
    pipeline = VideoProcessingPipeline()
    
    try:
        pipeline.run_serial()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()