import os
import yt_dlp
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event

# Configuration constants
MAX_CONCURRENT_DOWNLOADS = 4

# List of YouTube video URLs to download for analysis
VIDEO_URLS = [
    "https://www.youtube.com/watch?v=VgsC_aBquUE", "https://www.youtube.com/watch?v=DQzKw30LeTA",
    "https://www.youtube.com/watch?v=wolqWDmbp4A", "https://www.youtube.com/watch?v=OXJhvhNNHWA",
    "https://www.youtube.com/watch?v=BwB6x8CUkZA", "https://www.youtube.com/watch?v=_VfkVB2eX7U",
    "https://www.youtube.com/watch?v=185BHJ9avog", "https://www.youtube.com/watch?v=4dOgWZsDB6Q",
    "https://www.youtube.com/watch?v=xT0t6zdAknk", "https://www.youtube.com/watch?v=hx1mjT73xYE",
    "https://www.youtube.com/watch?v=KfaBRyCKRhk", "https://www.youtube.com/watch?v=QXrdp2rQk8Q",
    "https://www.youtube.com/watch?v=B2wgkp2CYYc", "https://www.youtube.com/watch?v=rk5HvJmy_yg",
    "https://www.youtube.com/watch?v=H6bJhKvFVw4", "https://www.youtube.com/watch?v=LC2dutG6KxM",
    "https://www.youtube.com/watch?v=EAl6qlAAWdc", "https://www.youtube.com/watch?v=dapucEtbO9s",
    "https://www.youtube.com/watch?v=GzqgzO14R9o", "https://www.youtube.com/watch?v=1VRliFlrvfA",
    "https://www.youtube.com/watch?v=jGYE2d4LJ5M", "https://www.youtube.com/watch?v=K1VTt_THL4A",
    "https://www.youtube.com/watch?v=DmDPDgOwCKM", "https://www.youtube.com/watch?v=N8ssqU9qOEo",
    "https://www.youtube.com/watch?v=nXQgqbXqggE", "https://www.youtube.com/watch?v=Ydv3f4Qfd0E",
    "https://www.youtube.com/watch?v=T7THOmIYB2w", "https://www.youtube.com/watch?v=cRTvtuj6r2c",
    "https://www.youtube.com/watch?v=EI0_5v6Ljdk", "https://www.youtube.com/watch?v=ZH5qaZmwX00",
    "https://www.youtube.com/watch?v=fofytJnb0Sk", "https://www.youtube.com/watch?v=y6zVqSWj2_0",
    "https://www.youtube.com/watch?v=eNBkJa2swuY", "https://www.youtube.com/watch?v=yX4qzKMaIqA",
    "https://www.youtube.com/watch?v=D49iaqNMWEs", "https://www.youtube.com/watch?v=p7_3Ri5bW4U",
    "https://www.youtube.com/watch?v=nXQoQO3iJOo", "https://www.youtube.com/watch?v=o-MDyvTirzc",
    "https://www.youtube.com/watch?v=wt6_Y8g2IYo", "https://www.youtube.com/watch?v=4GUOEVW0CD4",
    "https://www.youtube.com/watch?v=ESuIRp-x0_g", "https://www.youtube.com/watch?v=hd4YDHZcz0s",
    "https://www.youtube.com/watch?v=o3ZinMENzVA", "https://www.youtube.com/watch?v=GxHBMR3HN-4",
    "https://www.youtube.com/watch?v=z-Ld4FJTra0", "https://www.youtube.com/watch?v=7AVucpSPXR0",
    "https://www.youtube.com/watch?v=xXF08O2z8BU", "https://www.youtube.com/watch?v=qHV9opwgmLU",
    "https://www.youtube.com/watch?v=F95ifWpk_7I", "https://www.youtube.com/watch?v=myug1U3uklQ"
]

# Directory where videos will be saved
OUTPUT_DIR = "input/videos"

# Global shutdown event
shutdown_event = Event()

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"Received signal {signum}. Shutting down gracefully...")
    shutdown_event.set()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def download_video(url, output_path):
    """
    Downloads a single video using yt-dlp.
    Returns the URL and a success/failure status.
    """
    if shutdown_event.is_set():
        return url, False, "Shutdown requested"
    
    try:
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': os.path.join(output_path, '%(title)s [%(id)s].%(ext)s'),
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            if shutdown_event.is_set():
                return url, False, "Shutdown requested"
            
            ydl.download([url])
        
        print(f"Finished downloading: {url}")
        return url, True, "Success"
    except Exception as e:
        print(f"ERROR downloading {url}: {e}")
        return url, False, str(e)

def download_video_list_parallel(urls, output_path):
    """
    Downloads a list of videos in parallel using ThreadPoolExecutor.
    """
    print(f"Starting parallel video download for {len(urls)} videos...")
    print(f"Max concurrent downloads set to: {MAX_CONCURRENT_DOWNLOADS}")
    print(f"Output directory: '{os.path.abspath(output_path)}'")
    print("Press Ctrl+C to stop all downloads gracefully\n")
    
    os.makedirs(output_path, exist_ok=True)

    successful_downloads = 0
    failed_downloads = []
    cancelled_downloads = []

    try:
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS, thread_name_prefix="Downloader") as executor:
            future_to_url = {executor.submit(download_video, url, output_path): url for url in urls}
            
            for future in as_completed(future_to_url):
                if shutdown_event.is_set():
                    print("Shutdown requested. Cancelling remaining downloads...")
                    for f in future_to_url:
                        if not f.done():
                            f.cancel()
                            cancelled_downloads.append(future_to_url[f])
                    break
                
                url, success, message = future.result()
                if success:
                    successful_downloads += 1
                    print(f"Progress: {successful_downloads}/{len(urls)} completed")
                else:
                    if message == "Shutdown requested":
                        cancelled_downloads.append(url)
                    else:
                        failed_downloads.append((url, message))
                
                if shutdown_event.is_set():
                    break
                    
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down...")
        shutdown_event.set()
    except Exception as e:
        print(f"Unexpected error: {e}")
        shutdown_event.set()

    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Successfully downloaded: {successful_downloads} video(s)")
    print(f"Failed to download: {len(failed_downloads)} video(s)")
    print(f"Cancelled downloads: {len(cancelled_downloads)} video(s)")
    
    if failed_downloads:
        print(f"Failed downloads:")
        for url, error in failed_downloads[:5]:
            print(f"  - {url}: {error}")
        if len(failed_downloads) > 5:
            print(f"  ... and {len(failed_downloads) - 5} more")
    
    if cancelled_downloads:
        print(f"Cancelled downloads:")
        for url in cancelled_downloads[:5]:
            print(f"  - {url}")
        if len(cancelled_downloads) > 5:
            print(f"  ... and {len(cancelled_downloads) - 5} more")
    
    print("="*50)
    
    return successful_downloads, len(failed_downloads), len(cancelled_downloads)

if __name__ == "__main__":
    if not VIDEO_URLS:
        print("No video URLs found. Please add URLs to the VIDEO_URLS list in the script.")
    else:
        try:
            download_video_list_parallel(VIDEO_URLS, OUTPUT_DIR)
        except KeyboardInterrupt:
            print("Script stopped by user")
        except Exception as e:
            print(f"Script error: {e}")
        finally:
            print("Download script finished")