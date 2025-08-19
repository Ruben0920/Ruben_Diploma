#!/usr/bin/env python3
"""
Parallel Download Utility
Downloads multiple files concurrently using ThreadPoolExecutor
"""

import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import threading

class ParallelDownloader:
    """
    Downloads multiple files in parallel with progress tracking
    """
    
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.download_lock = threading.Lock()
        self.session = requests.Session()
        
    def download_file(self, url, output_path):
        """
        Download a single file with progress tracking
        """
        try:
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
            return True, url, output_path, total_size
            
        except Exception as e:
            return False, url, str(e), 0
    
    def download_multiple(self, download_list):
        """
        Download multiple files in parallel
        
        Args:
            download_list: List of tuples (url, output_path)
        """
        print(f"Starting parallel download of {len(download_list)} files...")
        print(f"Using {self.max_workers} concurrent workers")
        print("-" * 50)
        
        successful_downloads = []
        failed_downloads = []
        total_downloaded = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_url = {
                executor.submit(self.download_file, url, output_path): (url, output_path)
                for url, output_path in download_list
            }
            
            # Process completed downloads
            with tqdm(total=len(download_list), desc="Downloading", unit="file") as pbar:
                for future in as_completed(future_to_url):
                    success, url, result, size = future.result()
                    
                    if success:
                        successful_downloads.append((url, result, size))
                        total_downloaded += size
                        pbar.set_postfix({
                            'Success': len(successful_downloads),
                            'Failed': len(failed_downloads),
                            'Total MB': f"{total_downloaded / (1024*1024):.1f}"
                        })
                    else:
                        failed_downloads.append((url, result))
                        pbar.set_postfix({
                            'Success': len(successful_downloads),
                            'Failed': len(failed_downloads)
                        })
                    
                    pbar.update(1)
        
        # Print summary
        print("\n" + "=" * 50)
        print("DOWNLOAD SUMMARY")
        print("=" * 50)
        print(f"Successful: {len(successful_downloads)}")
        print(f"Failed: {len(failed_downloads)}")
        print(f"Total downloaded: {total_downloaded / (1024*1024):.1f} MB")
        
        if successful_downloads:
            print("\nSuccessfully downloaded:")
            for url, path, size in successful_downloads:
                print(f"  ✓ {path.name} ({size / (1024*1024):.1f} MB)")
        
        if failed_downloads:
            print("\nFailed downloads:")
            for url, error in failed_downloads:
                print(f"  ✗ {url}: {error}")
        
        return successful_downloads, failed_downloads

def main():
    """
    Example usage of the parallel downloader
    """
    # Example download list - replace with your actual URLs and paths
    download_list = [
        ("https://example.com/file1.mp4", "downloads/file1.mp4"),
        ("https://example.com/file2.mp4", "downloads/file2.mp4"),
        ("https://example.com/file3.mp4", "downloads/file3.mp4"),
    ]
    
    downloader = ParallelDownloader(max_workers=4)
    successful, failed = downloader.download_multiple(download_list)
    
    if failed:
        print(f"\n⚠️  {len(failed)} downloads failed. Check the errors above.")
    else:
        print("\n🎉 All downloads completed successfully!")

if __name__ == "__main__":
    main()
