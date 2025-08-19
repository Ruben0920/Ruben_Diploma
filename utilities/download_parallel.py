#!/usr/bin/env python3
"""
Parallel YouTube Video Download Utility
Downloads multiple YouTube videos concurrently using yt-dlp and ThreadPoolExecutor
"""

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import threading
import json

class ParallelYouTubeDownloader:
    """
    Downloads multiple YouTube videos in parallel with progress tracking
    """
    
    def __init__(self, max_workers=4, output_dir="input/videos"):
        self.max_workers = max_workers
        self.download_lock = threading.Lock()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if yt-dlp is installed
        try:
            subprocess.run(["yt-dlp", "--version"], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ yt-dlp is not installed!")
            print("Install it with: pip install yt-dlp")
            raise SystemExit(1)
        
    def download_video(self, url):
        """
        Download a single YouTube video
        """
        try:
            # Get video info first
            info_cmd = [
                "yt-dlp",
                "--print", "%(title)s",
                "--print", "%(duration)s", 
                "--print", "%(id)s",
                url
            ]
            
            result = subprocess.run(info_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return False, url, f"Failed to get video info: {result.stderr}", 0
                
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 3:
                title = lines[0]
                duration = lines[1] if lines[1] != 'NA' else '0'
                video_id = lines[2]
                
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_title = safe_title[:100]
                
                output_template = str(self.output_dir / f"{safe_title} [{video_id}].%(ext)s")
                
                download_cmd = [
                    "yt-dlp",
                    "--format", "best[height<=720]/best",
                    "--output", output_template,
                    "--no-playlist",
                    "--write-info-json",
                    "--write-sub", "--write-auto-sub",
                    "--sub-lang", "en",
                    url
                ]
                
                with self.download_lock:
                    print(f"Downloading: {safe_title}")
                
                result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=1800)
                
                if result.returncode == 0:
                    video_files = list(self.output_dir.glob(f"{safe_title} [{video_id}].*"))
                    video_files = [f for f in video_files if f.suffix in ['.mp4', '.mkv', '.webm']]
                    
                    if video_files:
                        file_size = video_files[0].stat().st_size
                        return True, url, video_files[0], file_size
                    else:
                        return False, url, "Video file not found after download", 0
                else:
                    return False, url, f"Download failed: {result.stderr}", 0
            else:
                return False, url, "Could not parse video info", 0
                
        except subprocess.TimeoutExpired:
            return False, url, "Download timeout (30 minutes)", 0
        except Exception as e:
            return False, url, f"Unexpected error: {str(e)}", 0
    
    def download_multiple(self, url_list):
        """
        Download multiple YouTube videos in parallel
        
        Args:
            url_list: List of YouTube URLs
        """
        print(f"Starting parallel download of {len(url_list)} YouTube videos...")
        print(f"Using {self.max_workers} concurrent workers")
        print(f"Output directory: {self.output_dir}")
        print("-" * 70)
        
        successful_downloads = []
        failed_downloads = []
        total_downloaded = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_url = {
                executor.submit(self.download_video, url): url
                for url in url_list
            }
            
            # Process completed downloads
            with tqdm(total=len(url_list), desc="Downloading videos", unit="video", colour="green") as pbar:
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        success, original_url, result, size = future.result()
                        
                        if success:
                            successful_downloads.append((original_url, result, size))
                            total_downloaded += size
                            pbar.set_postfix({
                                'Success': len(successful_downloads),
                                'Failed': len(failed_downloads),
                                'Total GB': f"{total_downloaded / (1024*1024*1024):.2f}"
                            })
                        else:
                            failed_downloads.append((original_url, result))
                            pbar.set_postfix({
                                'Success': len(successful_downloads),
                                'Failed': len(failed_downloads)
                            })
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        failed_downloads.append((url, f"Exception: {str(e)}"))
                        pbar.update(1)
        
        # Print summary
        print("\n" + "=" * 70)
        print("DOWNLOAD SUMMARY")
        print("=" * 70)
        print(f"Successful: {len(successful_downloads)}")
        print(f"Failed: {len(failed_downloads)}")
        print(f"Total downloaded: {total_downloaded / (1024*1024*1024):.2f} GB")
        
        if successful_downloads:
            print("\nSuccessfully downloaded videos:")
            for url, path, size in successful_downloads:
                print(f"  {path.name} ({size / (1024*1024):.1f} MB)")
        
        if failed_downloads:
            print("\nFailed downloads:")
            for url, error in failed_downloads:
                print(f"  {url}: {error}")
        
        print(f"\nAll videos saved to: {self.output_dir.absolute()}")
        
        return successful_downloads, failed_downloads

def main():
    """
    Download political debate videos for pipeline processing
    """
    # Political debate YouTube URLs
    youtube_urls = [
        "https://www.youtube.com/watch?v=VgsC_aBquUE",  # Full Debate: Harris vs. Trump
        "https://www.youtube.com/watch?v=DQzKw30LeTA",  # Were Israel's Actions in the Gaza War Justified?
        "https://www.youtube.com/watch?v=wolqWDmbp4A",  # Trump, Epstein and immigration raids
        "https://www.youtube.com/watch?v=OXJhvhNNHWA",  # Who is to blame for Sri Lanka's crises?
        "https://www.youtube.com/watch?v=BwB6x8CUkZA",  # Immigration, populism and the far right
        "https://www.youtube.com/watch?v=_VfkVB2eX7U",  # Anthony Albanese and Peter Dutton clash
        "https://www.youtube.com/watch?v=185BHJ9avog",  # Trump Voter Challenges Destiny To A Debate
        "https://www.youtube.com/watch?v=4dOgWZsDB6Q",  # DEBATE REPLAY: VP Harris and former President Trump
        "https://www.youtube.com/watch?v=xT0t6zdAknk",  # The Final Showdown: Albanese and Dutton's last debate
        "https://www.youtube.com/watch?v=hx1mjT73xYE",  # Final Presidential Debate 2012 Complete
        "https://www.youtube.com/watch?v=KfaBRyCKRhk",  # Obama vs. Romney: The first 2012 presidential debate
        "https://www.youtube.com/watch?v=QXrdp2rQk8Q",  # Confrontational Debate vs InfoWars Host
        "https://www.youtube.com/watch?v=B2wgkp2CYYc",  # Socialism vs. Capitalism: A Debate
        "https://www.youtube.com/watch?v=rk5HvJmy_yg",  # The First Election Debate | ITV
        "https://www.youtube.com/watch?v=H6bJhKvFVw4",  # The Channel 4 News #ClimateDebate
        "https://www.youtube.com/watch?v=LC2dutG6KxM",  # 2010 British General Election - BBC Debate
        "https://www.youtube.com/watch?v=EAl6qlAAWdc",  # UK Election Debate - Immigration, Law and Order
        "https://www.youtube.com/watch?v=dapucEtbO9s",  # From Westminster to your world - UK election debate
        "https://www.youtube.com/watch?v=GzqgzO14R9o",  # Corbyn v Johnson - General Election Leader's Debate
        "https://www.youtube.com/watch?v=1VRliFlrvfA",  # Federal Leaders' Debate 2019
        "https://www.youtube.com/watch?v=jGYE2d4LJ5M",  # 2011 Canadian Federal Election Debate
        "https://www.youtube.com/watch?v=K1VTt_THL4A",  # Debate: Anti-Zionism is Anti-Semitism
        "https://www.youtube.com/watch?v=DmDPDgOwCKM",  # David Speers hosts post-election Insiders
        "https://www.youtube.com/watch?v=N8ssqU9qOEo",  # 1968 Canadian Federal Election Debate
        "https://www.youtube.com/watch?v=nXQgqbXqggE",  # 2008 Canadian Federal Election Debate
        "https://www.youtube.com/watch?v=Ydv3f4Qfd0E",  # 1984 Canadian Federal Election Debate
        "https://www.youtube.com/watch?v=T7THOmIYB2w",  # CBC News: Federal leaders debate Canada's future
        "https://www.youtube.com/watch?v=cRTvtuj6r2c",  # Georgia gubernatorial candidates face off
        "https://www.youtube.com/watch?v=EI0_5v6Ljdk",  # Hochul, Zeldin spar over crime, Trump
        "https://www.youtube.com/watch?v=ZH5qaZmwX00",  # Democratic candidates for Texas governor
        "https://www.youtube.com/watch?v=fofytJnb0Sk",  # Race for Governor: Wilson, Waguespack
        "https://www.youtube.com/watch?v=y6zVqSWj2_0",  # Georgia governor debate: Stacey Abrams vs. Brian Kemp
        "https://www.youtube.com/watch?v=eNBkJa2swuY",  # Louisiana Governor's Debate - The Runoff
        "https://www.youtube.com/watch?v=yX4qzKMaIqA",  # 1st Televised NOLA Mayoral Debate
        "https://www.youtube.com/watch?v=D49iaqNMWEs",  # Race for Governor: Wilson, Waguespack (New Orleans)
        "https://www.youtube.com/watch?v=p7_3Ri5bW4U",  # North Dakota U.S. House Debate | KFGO
        "https://www.youtube.com/watch?v=nXQoQO3iJOo",  # Face to Face: North Dakota US Senate Debate
        "https://www.youtube.com/watch?v=o-MDyvTirzc",  # Face to Face: North Dakota Republican Primary Debate 2024
        "https://www.youtube.com/watch?v=wt6_Y8g2IYo",  # Face to Face: North Dakota U.S. Senate Debate 2024
        "https://www.youtube.com/watch?v=4GUOEVW0CD4",  # Face to Face: North Dakota US House Debate
        "https://www.youtube.com/watch?v=ESuIRp-x0_g",  # Election 2022: U.S. Senate Candidates Debate
        "https://www.youtube.com/watch?v=hd4YDHZcz0s",  # 2016 Debate for the State: U.S. Senate Race
        "https://www.youtube.com/watch?v=o3ZinMENzVA",  # North Dakota Gubernatorial Debate - October 3rd, 2016
        "https://www.youtube.com/watch?v=GxHBMR3HN-4",  # 2022 California Gubernatorial Debate: Newsom vs. Dahle
        "https://www.youtube.com/watch?v=z-Ld4FJTra0",  # U.S. Senate Debate at Cal State LA
        "https://www.youtube.com/watch?v=7AVucpSPXR0",  # California Counts: U.S. Senate Debate
        "https://www.youtube.com/watch?v=xXF08O2z8BU",  # Newsom, Dahle meet in only gubernatorial debate
        "https://www.youtube.com/watch?v=qHV9opwgmLU",  # 1993 Canadian Federal Election Debate
        "https://www.youtube.com/watch?v=F95ifWpk_7I",  # 2000 Canadian Federal Election Debate
        "https://www.youtube.com/watch?v=myug1U3uklQ",  # California Governor Recall Debate
    ]
    
    print("Political Debate Video Downloader")
    print("=" * 70)
    print(f"Preparing to download {len(youtube_urls)} political debate videos")
    print("These videos will be used for multimodal analysis in the Ruben_Diploma pipeline")
    print("=" * 70)
    
    downloader = ParallelYouTubeDownloader(max_workers=3, output_dir="input/videos")
    
    try:
        successful, failed = downloader.download_multiple(youtube_urls)
        
        print("\n" + "=" * 70)
        if failed:
            print(f"{len(failed)} downloads failed. Check the errors above.")
            print(f"{len(successful)} videos successfully downloaded and ready for processing!")
        else:
            print("All downloads completed successfully!")
            print("Ready to run the analysis pipeline with: python run_pipeline.py")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")

if __name__ == "__main__":
    main()
