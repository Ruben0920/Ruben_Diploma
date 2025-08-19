#!/usr/bin/env python3
"""
Parallel Pipeline Orchestrator
Manages Docker containers for efficient parallel video processing
"""

import os
import time
import json
import signal
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import docker
from docker.errors import DockerException
import psutil
from colorama import init, Fore, Back, Style
from tqdm import tqdm

init(autoreset=True)

class ParallelPipelineOrchestrator:
    """
    Orchestrates parallel video processing using Docker containers
    with resource management and health monitoring
    """
    
    def __init__(self):
        self.client = docker.from_env()
        self.shutdown_event = threading.Event()
        self.active_containers = {}
        self.container_locks = {}
        
        self.max_concurrent_videos = int(os.getenv('MAX_CONCURRENT_VIDEOS', '4'))
        self.max_concurrent_services = int(os.getenv('MAX_CONCURRENT_SERVICES', '2'))
        
        self.services = [
            'audio_processing',
            'face_recognition', 
            'mapping_face_and_audio',
            'polarization_analysis'
        ]
        
        self.resource_limits = {
            'audio_processing': {'memory': '2G', 'cpus': '1.0'},
            'face_recognition': {'memory': '3G', 'cpus': '1.5'},
            'mapping_face_and_audio': {'memory': '2G', 'cpus': '1.0'},
            'polarization_analysis': {'memory': '2G', 'cpus': '1.0'}
        }
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.base_dir = Path('/app')
        self.input_dir = self.base_dir / 'input'
        self.output_dir = self.base_dir / 'output'
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"Received signal {signum}. Shutting down gracefully...")
        self.shutdown_event.set()
        self._cleanup_all_containers()
        sys.exit(0)
    
    def _cleanup_all_containers(self):
        """Clean up all active containers"""
        print("Cleaning up containers...")
        
        for container_id, container_info in self.active_containers.items():
            try:
                container = self.client.containers.get(container_id)
                if container.status == 'running':
                    print(f"  Stopping {container_info['service']} for {container_info['video']}")
                    container.stop(timeout=30)
                container.remove()
            except Exception as e:
                print(f"  Warning: Could not clean up container {container_id}: {e}")
        
        self.active_containers.clear()
    
    def _check_system_resources(self) -> bool:
        """Check if system has enough resources to start processing"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            print("System Resources:")
            print(f"  Memory: {memory.available / (1024**3):.1f}GB available / {memory.total / (1024**3):.1f}GB total")
            print(f"  CPU Usage: {cpu_percent:.1f}%")
            
            if memory.available < 8 * 1024**3:
                print(f"Warning: Low memory available. Consider reducing concurrent videos.")
                return False
            
            return True
            
        except Exception as e:
            print(f"Could not check system resources: {e}")
            return True
    
    def _get_video_files(self) -> List[str]:
        """Get list of video files to process"""
        video_files = []
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
        
        if not self.input_dir.exists():
            print(f"Input directory {self.input_dir} does not exist!")
            return []
        
        for file in self.input_dir.iterdir():
            if file.is_file() and file.suffix.lower() in video_extensions:
                video_files.append(file.name)
        
        return sorted(video_files)
    
    def _create_output_structure(self, video_name: str) -> Path:
        """Create output directory structure for a video"""
        video_dir_name = Path(video_name).stem
        video_output_dir = self.output_dir / video_dir_name
        
        subdirs = [
            "speaker_diarization",
            "face_recognition", 
            "mapping_face_and_audio",
            "polarization_analysis",
            "polarization_analysis/visualizations"
        ]
        
        for subdir in subdirs:
            dir_path = video_output_dir / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return video_output_dir
    
    def _run_service_container(self, service_name: str, video_name: str, 
                              video_output_dir: Path) -> Tuple[bool, str]:
        """Run a single service container for a video"""
        if self.shutdown_event.is_set():
            return False, "Shutdown requested"
        
        try:
            env_vars = {
                'VIDEO_FILENAME': video_name,
                'OUTPUT_BASE_DIR': str(video_output_dir.relative_to(self.base_dir)),
                'PYTHONUNBUFFERED': '1'
            }
            
            volumes = {
                str(self.input_dir): {'bind': '/app/input', 'mode': 'ro'},
                str(video_output_dir): {'bind': '/app/output', 'mode': 'rw'}
            }
            
            container = self.client.containers.run(
                f"ruben_diploma_{service_name}:latest",
                environment=env_vars,
                volumes=volumes,
                detach=True,
                remove=True,
                mem_limit=self.resource_limits[service_name]['memory'],
                cpu_quota=int(float(self.resource_limits[service_name]['cpus']) * 100000),
                cpu_period=100000
            )
            
            container_id = container.id
            self.active_containers[container_id] = {
                'service': service_name,
                'video': video_name,
                'start_time': time.time()
            }
            
            result = container.wait()
            
            if container_id in self.active_containers:
                del self.active_containers[container_id]
            
            if result['StatusCode'] == 0:
                return True, "Success"
            else:
                logs = container.logs().decode('utf-8')
                return False, f"Container failed with exit code {result['StatusCode']}: {logs[-500:]}"
                
        except DockerException as e:
            return False, f"Docker error: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    def _process_video_parallel(self, video_name: str) -> Tuple[str, bool, float, List[str], List[str]]:
        """Process a single video with parallel services"""
        start_time = time.time()
        
        print(f"[{video_name}] Starting parallel processing")
        
        video_output_dir = self._create_output_structure(video_name)
        
        successful_services = []
        failed_services = []
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_services) as executor:
            future_to_service = {
                executor.submit(self._run_service_container, service, video_name, video_output_dir): service
                for service in self.services
            }
            
            for future in as_completed(future_to_service):
                if self.shutdown_event.is_set():
                    break
                
                service = future_to_service[future]
                try:
                    success, message = future.result()
                    if success:
                        successful_services.append(service)
                        print(f"[{video_name}] {service} completed")
                    else:
                        failed_services.append(service)
                        print(f"[{video_name}] {service} failed: {message}")
                except Exception as e:
                    failed_services.append(service)
                    print(f"[{video_name}] {service} exception: {str(e)}")
        
        elapsed_time = time.time() - start_time
        
        success = len(successful_services) == len(self.services)
        print(f"[{video_name}] Processing complete in {elapsed_time:.1f}s - "
              f"Success: {len(successful_services)}/{len(self.services)}")
        
        return video_name, success, elapsed_time, successful_services, failed_services
    
    def run_parallel_pipeline(self):
        """Main execution method with parallel video and service processing"""
        print("="*70)
        print("Parallel Pipeline Orchestrator")
        print("="*70)
        
        if not self._check_system_resources():
            print("Proceeding with resource warnings...")
        
        video_files = self._get_video_files()
        if not video_files:
            print("No video files found in input directory!")
            return
        
        print(f"Found {len(video_files)} video(s) to process:")
        for i, video in enumerate(video_files, 1):
            print(f"  {i}. {video}")
        
        print(f"Configuration:")
        print(f"  Max concurrent videos: {self.max_concurrent_videos}")
        print(f"  Max concurrent services per video: {self.max_concurrent_services}")
        print(f"  Total services: {len(self.services)}")
        
        print("Starting parallel processing...")
        print("-"*70)
        
        results = []
        total_start_time = time.time()
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_concurrent_videos) as executor:
                future_to_video = {
                    executor.submit(self._process_video_parallel, video): video
                    for video in video_files
                }
                
                with tqdm(total=len(video_files), desc="Processing videos", 
                         unit="video", colour="green") as pbar:
                    for future in as_completed(future_to_video):
                        if self.shutdown_event.is_set():
                            break
                        
                        video = future_to_video[future]
                        try:
                            result = future.result()
                            results.append(result)
                            pbar.update(1)
                        except Exception as e:
                            print(f"[{video}] Unexpected error: {str(e)}")
                            results.append((video, False, 0, [], self.services))
                            pbar.update(1)
            
        except KeyboardInterrupt:
            print("Pipeline interrupted by user")
        finally:
            total_elapsed = time.time() - total_start_time
            
            self._print_final_summary(results, total_elapsed, len(video_files))
    
    def _print_final_summary(self, results: List[Tuple], total_time: float, total_videos: int):
        """Print comprehensive final summary"""
        print("\n" + "="*70)
        print("FINAL SUMMARY - PARALLEL PROCESSING COMPLETE")
        print("="*70)
        
        successful_videos = [r for r in results if r[1]]
        failed_videos = [r for r in results if not r[1]]
        
        print(f"Processing Statistics:")
        print(f"  Total videos processed: {total_videos}")
        print(f"  Successful: {len(successful_videos)}")
        print(f"  Failed/Partial: {len(failed_videos)}")
        print(f"  Total processing time: {total_time:.1f} seconds")
        print(f"  Average time per video: {total_time/total_videos:.1f} seconds")
        
        if successful_videos:
            print("Successfully Processed Videos:")
            for video, _, proc_time, services, _ in successful_videos:
                print(f"  • {video} ({proc_time:.1f}s)")
        
        if failed_videos:
            print("Failed or Partially Processed Videos:")
            for video, _, proc_time, successful, failed in failed_videos:
                print(f"  • {video} ({proc_time:.1f}s)")
                if successful:
                    print(f"    Completed: {', '.join(successful)}")
                if failed:
                    print(f"    Failed: {', '.join(failed)}")
        
        print(f"All outputs saved to: {self.output_dir}")
        
        if total_videos > 1:
            sequential_estimate = sum([r[2] for r in results])
            speedup = sequential_estimate / total_time
            print(f"Performance: ~{speedup:.1f}x speedup compared to sequential processing")
        
        print("="*70)


def main():
    """Main entry point"""
    try:
        orchestrator = ParallelPipelineOrchestrator()
        orchestrator.run_parallel_pipeline()
    except KeyboardInterrupt:
        print("Orchestrator stopped by user")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
